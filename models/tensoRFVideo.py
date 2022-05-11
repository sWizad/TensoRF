# tensoRF video model
# TODO: 1. support time coarse to fine
# TODO: 2. verify that it can run well with crop image
# TODO: 3. try harder dataset
import torch
import torch.nn.functional as F
import numpy as np

from utils import printlog
from .tensoRF import TensorVMSplit
from .tensorBase import raw2alpha, AlphaGridMask

class TensoRFVideo(TensorVMSplit):
    """
    a typical tensoRF but add the video dimension into it
    """
    def __init__(self, *args, **kwargs):
        print("Model: TensoRFVideo")
        self.max_t = kwargs['max_t'] if 'max_t' in kwargs else 1
        #self.t_keyframe = kwargs['t_keyframe']
        self.max_upsampling = len(kwargs['upsamp_list'])
        self.upsampling_cnt = 0 
        if self.max_t > kwargs['t_keyframe']:
            self.keyframe_upsampling = (torch.round(torch.exp(torch.linspace(np.log(np.floor(self.max_t / kwargs['t_keyframe'])), np.log(self.max_t), self.max_upsampling+1))).long()).tolist()
            self.keyframe_initial = self.keyframe_upsampling[0]
            self.keyframe_upsampling = self.keyframe_upsampling[1:]
        else:
            self.keyframe_initial = self.max_t
            self.keyframe_upsampling = [self.max_t]
        super(TensorVMSplit, self).__init__(*args, **kwargs)

    def init_svd_volume(self, res, device):
        # density_time is list of 3 contain shape #[1,n_component,num_frame,1]
        self.density_plane, self.density_line, self.density_time = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line, self.app_time = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef, time_coeff = [], [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
            """
            time_coeff.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
            """
            time_coeff.append(torch.nn.Parameter(scale * torch.randn((1, n_component[i], self.keyframe_initial, 1)))) #TODO: Vec id shouldn't duplicate
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device), torch.nn.ParameterList(time_coeff).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz}, {'params': self.density_time, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz}, {'params': self.app_time, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network},
                          ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line) + self.vectorDiffs(self.density_time) + self.vectorDiffs(self.app_time)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx])) + torch.mean(torch.abs(self.density_time[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3  + reg(self.density_time[idx]) * 1e-3
        return total
    
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3  + reg(self.app_time[idx]) * 1e-3
        return total


    def sample_time(self, time_sampled, is_train):
        # normalize 
        time_sampled = ((time_sampled + 0.5) / self.max_t) * 2.0 - 1.0
        # apply randomization on training to prevent overfit 

        if is_train:
            rand_val = torch.rand_like(time_sampled) * 2.0 - 1.0 #random frame from -1 to 1
            rand_val = rand_val / self.max_t # shifting not more than 1 frame 
            time_sampled = torch.clip(time_sampled + rand_val, -1.0, 1.0)
        return time_sampled

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        """
        @params rays_chunk - ray information shape [batch_size,7] conist of rays_o [0,1,2], rays_d [3,4,5] and frame_id [6]
        """
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        # prepare normalized [-1,1] time sample
        time_sampled = self.sample_time(rays_chunk[:,6], is_train=is_train)
        time_sampled = time_sampled[...,None].expand(-1, xyz_sampled.shape[1])

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], time_sampled[ray_valid])
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask], time_sampled[app_mask])
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight
    
    def compute_densityfeature(self, xyz_sampled, time_sampled):
        """
        @params xyz_sampled #(n,3)
        @params time_sampled #(n,), range[-1,1]
        """
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2) #[3, 1304014, 1, 2]
        
        coordinate_time = torch.stack([time_sampled, time_sampled, time_sampled])
        coordinate_time = torch.stack((torch.zeros_like(coordinate_time), coordinate_time), dim=-1).detach().view(3, -1, 1, 2) #[3, 1304014, 1, 2]


        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            time_coef_point = F.grid_sample(self.density_time[idx_plane], coordinate_time[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])

        sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point * time_coef_point, dim=0)
        #sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled, time_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        coordinate_time = torch.stack([time_sampled, time_sampled, time_sampled])
        coordinate_time = torch.stack((torch.zeros_like(coordinate_time), coordinate_time), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_coef_point,line_coef_point, time_coef_point = [],[],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            time_coef_point.append(F.grid_sample(self.app_time[idx_plane], coordinate_time[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))

        plane_coef_point, line_coef_point, time_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point), torch.cat(time_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point * time_coef_point).T)
        #return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, time_coef, res_target):
        time_id = self.upsampling_cnt if self.upsampling_cnt < len(self.keyframe_upsampling) else len(self.keyframe_upsampling) - 1
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            time_coef[i] = torch.nn.Parameter(F.interpolate(time_coef[i].data, size=(self.keyframe_upsampling[time_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef, time_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line, self.app_time = self.up_sampling_VM(self.app_plane, self.app_line, self.app_time, res_target)
        self.density_plane, self.density_line, self.density_time = self.up_sampling_VM(self.density_plane, self.density_line, self.density_time, res_target)
        self.upsampling_cnt += 1

        self.update_stepSize(res_target)
        printlog(f'upsamping to {res_target}')
    
    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        # we cannot create AlphaMask (OccupencyGrid) 
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = torch.zeros_like(dense_xyz[...,0])
        for t in torch.linspace(-1,1, self.max_t):
            time_sampled = t.expand(dense_xyz[0].view(-1,3).shape[0]).to(dense_xyz.device)
            for i in range(gridSize[2]):
                alpha[i] += self.compute_alpha(dense_xyz[i].view(-1,3), time_sampled, self.distance_scale*self.aabbDiag).view((gridSize[1], gridSize[0]))
        alpha = alpha.clamp(0,1)[None,None]


        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    def compute_alpha(self, xyz_locs, time_sampled, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled, time_sampled[alpha_mask])
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)
        elif self.fea2denseAct == "abs":
            return torch.abs(density_features)
        else:
            raise NotImplementedError("Not avaible feature2density")

    def normalize_coord(self, xyz_sampled):
        aabb = self.aabb.to(xyz_sampled.device)
        invaabbSize = self.invaabbSize.to(xyz_sampled.device)
        return (xyz_sampled-aabb[0]) * invaabbSize - 1
    
    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        aabb = self.aabb.to(rays_pts.device)
        mask_outbbox = ((aabb[0] > rays_pts) | (rays_pts > aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

class TensoRFVideoStaticColor(TensoRFVideo):
    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line, self.density_time = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line, self.app_time = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Sequential(
            torch.nn.Linear(sum(self.app_n_comp), 3),
            torch.nn.Sigmoid()
        ).to(device) #output only 3 color to create static color instead