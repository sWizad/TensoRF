# NeRFVideo, 6-D input NeRF to show how well it can render the video
import torch
import torch.nn.functional as F

from utils import printlog
from .tensorBase import TensorBase, raw2alpha, positional_encoding, AlphaGridMask


class NeRF(TensorBase):
    def __init__(self, *args, **kwargs):
        if 'max_t' in kwargs and kwargs['max_t'] > 1:
            raise Exception("--num_frames must be removed from setting file or set it to 1")
        self.nerf_hidden, self.pos_pe, self.view_pe = kwargs['featureC'], kwargs['pos_pe'], kwargs['view_pe']
        print("NeRF: train NeRF to verify everything is work as expected")
        super().__init__(*args, **kwargs) #initail super class

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        print("lr_init_spatialxyz: ", lr_init_spatialxyz)
        print("lr_init_network: ", lr_init_network)
        grad_vars = [
            {'params': self.sigma_front_net.parameters(), 'lr': lr_init_network},
            {'params': self.sigma_back_net.parameters(), 'lr': lr_init_network},
            {'params': self.sigma_layer.parameters(), 'lr': lr_init_network},
            {'params': self.color_net.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def init_svd_volume(self, res, device):
        # neural network for handleing
        # predefine variable 
        hidden_size =  self.nerf_hidden
        pos_pe = (self.pos_pe * 2) * 3 # input is x,y,z
        view_pe = ((self.view_pe * 2) * 3) #input is vx,vy,vz

        self.sigma_front_net = torch.nn.Sequential(
            torch.nn.Linear(pos_pe, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ).to(device)
        self.sigma_back_net = torch.nn.Sequential(
            torch.nn.Linear(pos_pe + hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
        ).to(device)
        self.sigma_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1),
            AbsoluteAcivation()
        ).to(device)
        self.color_net = torch.nn.Sequential(
            torch.nn.Linear(view_pe + hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid(),
        ).to(device)
        #apply datapararell
        """
        self.sigma_front_net = torch.nn.DataParallel(self.sigma_front_net)
        self.sigma_back_net = torch.nn.DataParallel(self.sigma_back_net)
        self.sigma_layer = torch.nn.DataParallel(self.sigma_layer)
        self.color_net = torch.nn.DataParallel(self.color_net)
        """
    def feature2density(self, sigma_feature):
        sigma = self.sigma_layer(sigma_feature)[...,0]
        return sigma

    def compute_densityfeature(self, xyz_sampled):
        sampled = positional_encoding(xyz_sampled, self.pos_pe)
        x = self.sigma_front_net(sampled) #([132700, 128])
        x = torch.cat([x, sampled], dim=-1) #torch.Size([132700, 164])
        x = self.sigma_back_net(x)
        return x

    def compute_appfeature(self, xyz_sampled, sigma_features, view_directions):
        sampled = positional_encoding(view_directions, self.view_pe)
        x = torch.cat([sampled, sigma_features], dim=-1)
        x = self.color_net(x)
        return x

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
    
    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
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
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask_ray = weight[ray_valid]  > self.rayMarch_weight_thres

        if app_mask_ray.any():
            xyz_r = xyz_sampled[ray_valid][app_mask_ray]
            viewdir_r = viewdirs[ray_valid][app_mask_ray]
            sigma_features_r = sigma_feature[app_mask_ray]
            app_features = self.compute_appfeature(xyz_r, sigma_features_r, viewdir_r)
            valid_rgbs = self.renderModule(xyz_r, viewdir_r, app_features)
            temp_rgb =  torch.zeros_like(rgb[ray_valid])
            temp_rgb[app_mask_ray] = valid_rgbs
            rgb[ray_valid] = temp_rgb

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[2]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.distance_scale*self.aabbDiag).view((gridSize[1], gridSize[0]))
        alpha = alpha.clamp(0,1)[None,None]


        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMaskMultiDevice(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMaskMultiDevice(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])


class AbsoluteAcivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.abs(input)

class AlphaGridMaskMultiDevice(AlphaGridMask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_alpha(self, xyz_sampled):
        device = xyz_sampled.device 
        xyz_sampled = self.normalize_coord(xyz_sampled.to(self.aabb.device)) #move data to first device
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals.to(device) #copy value back to each distributed device