# tensoRF video model
# TODO: 1. support time coarse to fine
# TODO: 2. verify that it can run well with crop image
# TODO: 3. try harder dataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
#from matplotlib import pyplot as plt 

from utils import printlog
from .tensoRF import TensorVMSplit
from .tensorBase import raw2alpha, AlphaGridMask

MATH_PI = torch.acos(torch.zeros(1)).item() * 2

class TensoRF5dSigma(TensorVMSplit):
    """
    a typical tensoRF but add the video dimension into it
    """
    def __init__(self, *args, **kwargs):
        print("Model: TensoRF5DSigma")
        self.viewdir_bounds = self.find_viewdir_boundary(kwargs['train_dataset'])
        super(TensorVMSplit, self).__init__(*args, **kwargs)

    def init_svd_volume(self, res, device):
        # density_time is list of 3 contain shape #[1,n_component,num_frame,1]
        self.density_plane, self.density_line, self.density_viewdir = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line, _ = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device) #we omit self.viewdir
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef, viewdir_coeff = [], [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))
            viewdir_coeff.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  
        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device), torch.nn.ParameterList(viewdir_coeff).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz}, {'params': self.density_viewdir, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network},
                          ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)# + self.vectorDiffs(self.density_time) + self.vectorDiffs(self.app_time)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx])) + torch.mean(torch.abs(self.density_viewdir[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 + reg(self.density_line[idx]) * 1e-3  + reg(self.density_viewdir[idx]) * 1e-2 
        return total
    
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 + reg(self.app_line[idx]) * 1e-3#  + reg(self.app_viewdir[idx]) * 1e-2
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

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            #norm_dir = norm_spherical(viewdirs)
            norm_dir = self.norm_sterographic(viewdirs)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], norm_dir[ray_valid])
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        if app_mask.any():
            app_features = self.compute_appfeature(xyz_sampled[app_mask])
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
    
    def compute_densityfeature(self, xyz_sampled, viewdirs):
        """
        @params xyz_sampled #(n,3)
        @params viewdirs #(n,3)
        """
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2) #torch.Size([3, 1332531, 1, 2])
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2) #[3, 1304014, 1, 2]
        
        
        coordinate_viewdir = viewdirs[None,:,None,:].expand(3,-1,-1,-1)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            viewdir_coef_point = F.grid_sample(self.density_viewdir[idx_plane], coordinate_viewdir[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
 
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point * viewdir_coef_point, dim=0) #remove detach incase need grad
            #sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))            

        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, viewdirs_coef, res_target):
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',align_corners=True))
            line_coef[i] = torch.nn.Parameter(F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            if viewdirs_coef is not None:
                viewdirs_coef[i] = torch.nn.Parameter(F.interpolate(viewdirs_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',align_corners=True))

        return plane_coef, line_coef, viewdirs_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line, _, = self.up_sampling_VM(self.app_plane, self.app_line, None, res_target)
        self.density_plane, self.density_line, self.density_viewdir = self.up_sampling_VM(self.density_plane, self.density_line, self.density_viewdir, res_target)

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
        #SPH_RES = torch.max(torch.tensor(gridSize)).item()
        SPH_RES = 64
        phis = torch.linspace(-1,1, SPH_RES)
        thetas = torch.linspace(-1,1, SPH_RES)
        phis, thetas = torch.meshgrid(phis, thetas, indexing='ij')
        spherical_dirs = torch.cat([phis[...,None],thetas[...,None]],dim=-1).view(-1,2)
        for sph_dir in tqdm(spherical_dirs):
            for i in range(gridSize[2]):
                xyz_sampled = dense_xyz[i].view(-1,3)
                sph_dir_rand = sph_dir[None].expand(xyz_sampled.shape[0], -1) + (((torch.rand(xyz_sampled.shape[0], 2) + 1.0) / 2.0) / SPH_RES)
                sph_dir_rand = sph_dir_rand.to(xyz_sampled.device)
                alpha[i] += self.compute_alpha(xyz_sampled, sph_dir_rand, self.distance_scale*self.aabbDiag).view((gridSize[1], gridSize[0]))
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

    def compute_alpha(self, xyz_locs, norm_spherical, length=1):

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled, norm_spherical[alpha_mask])
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

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

    def find_viewdir_boundary(self, dataset):
        CENTER_CAMID = 8 #note this should be varies depend on dataset
        self.center_c2w = torch.from_numpy(dataset.poses[CENTER_CAMID]).float()
        dataset.is_sampling = False
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        
        viewdirs = []
        for i,ray_batch in enumerate(dataloader):
            rays_d = ray_batch['rays'][:,:,3:6].view(-1,3)
            rays_d = rays_d / (torch.norm(rays_d,dim=1)+ torch.finfo(rays_d.dtype).eps)[...,None].expand(-1,3)
            viewdirs.append(rays_d)
        viewdirs = torch.cat(viewdirs)
        viewdirs = (self.center_c2w[:3,:3] @ viewdirs.t()).t()
        sph = sterographic(viewdirs)
        view_bound = torch.tensor([
            [torch.min(sph[...,0]), torch.max(sph[...,0])],
            [torch.min(sph[...,1]), torch.max(sph[...,1])]
        ])
        dataset.is_sampling = True #set back to true to avoid mess-up with other dataset called
        return view_bound

    def norm_sterographic(self, viewdirs):
        viewdirs_len =  torch.norm(viewdirs,dim=-1)[...,None]
        if len(viewdirs_len.shape) == 2:
            viewdirs_len = viewdirs_len.expand(-1, 3)
        elif len(viewdirs_len.shape) == 3:
            viewdirs_len = viewdirs_len.expand(-1, -1, 3)
        viewdirs = viewdirs / (viewdirs_len + torch.finfo(viewdirs.dtype).eps)
        viewdirs = viewdirs.view(-1, 3)
        viewdirs = (self.center_c2w[:3,:3].to(viewdirs.device) @ viewdirs.t()).t()
        viewdirs = viewdirs.view(*viewdirs_len.shape)
        sph = sterographic(viewdirs)
        sph[...,0] = (sph[...,0] - self.viewdir_bounds[0,0]) / (self.viewdir_bounds[0,1] - self.viewdir_bounds[0,0])
        sph[...,1] = (sph[...,1] - self.viewdir_bounds[1,0]) / (self.viewdir_bounds[1,1] - self.viewdir_bounds[1,0])
        sph = (sph * 2.0) - 1.0 # rescale to [-1,-1]
        sph = torch.clip(sph, -1.0, 1.0)
        return sph


def norm_spherical(cartesian):
    """
    @params cartesian coordinate
    @return  normalize spherical coordinate in [-1,1]
    """
    cartesian_len = torch.norm(cartesian, dim=-1)[...,None]
    if len(cartesian.shape) == 2:
        cartesian_len = cartesian_len.expand(-1, 3)
    elif len(cartesian.shape) == 3:
        cartesian_len = cartesian_len.expand(-1, -1, 3)

    cartesian = cartesian / cartesian_len
    spherical = cartesian2spherical(cartesian)
    normalized = spherical / MATH_PI
    return normalized

def cartesian2spherical(cartesian):
    """
    convert a cartesian coordiante to spherical coordinate
    Note: we assume the input already normalize so, we return on phi and theta
    @param cartesian coodinate #(n_ray,3)
    @return spherical coorinate #(n_ray,2) consist of phi and theta
    @see https://keisan.casio.com/exec/system/1359533867
    """
    eps = torch.finfo(cartesian.dtype).eps #smallest possible value for avoid divde by zero
    x = cartesian[...,0]
    y = cartesian[...,1]
    z = cartesian[...,2]
    # note that
    #x = cartesian[...,1]
    #y = cartesian[...,0]
    #z = cartesian[...,2]
    theta = torch.atan2(y,x)
    phi = torch.atan2(torch.sqrt(y**2 + x**2), z + eps)
    return torch.cat([theta[..., None], phi[..., None]],dim=-1)

def sterographic(XYZ):
    X = XYZ[...,0:1]
    Y = XYZ[...,1:2]
    Z = XYZ[...,2:3]
    eps =  torch.finfo(XYZ.dtype).eps
    diver = 1 - Z + eps
    x = X / diver
    y = Y / diver
    combine = torch.cat([x,y],dim=-1)
    return combine