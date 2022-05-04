# NeRFVideo, 6-D input NeRF to show how well it can render the video
import torch
import torch.nn.functional as F

from utils import printlog
from .tensorBase import TensorBase, raw2alpha, positional_encoding

class NeRFVideo(TensorBase):
    def __init__(self, *args, **kwargs):
        self.max_t = kwargs['max_t'] if 'max_t' in kwargs else 1
        self.nerf_hidden, self.pos_pe, self.view_pe = kwargs['featureC'], kwargs['pos_pe'], kwargs['view_pe']
        print("NeRFVideo: A modification version of NeRF to handle video")
        super().__init__(*args, **kwargs) #initail super class

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
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
        pos_pe = (self.pos_pe * 2) * 4 # input is x,y,z,t
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

    def feature2density(self, sigma_feature):
        sigma = self.sigma_layer(sigma_feature)[...,0]
        return sigma

    def compute_densityfeature(self, xyz_sampled, time_sampled):
        sampled = torch.cat([xyz_sampled, time_sampled[...,None]],dim=-1)
        sampled = positional_encoding(sampled, self.pos_pe)
        x = self.sigma_front_net(sampled)
        x = torch.cat([x, sampled], dim=-1)
        x = self.sigma_back_net(x)
        return x

    def compute_appfeature(self, xyz_sampled, time_sampled, sigma_features, view_directions):
        sampled = positional_encoding(view_directions, self.view_pe)
        x = torch.cat([sampled, sigma_features], dim=-1)
        x = self.color_net(x)
        return x

    
    def sample_time(self, time_sampled, is_train):
        # normalize 
        time_sampled = (time_sampled / self.max_t) * 2.0 - 1.0
        # apply randomization on training to prevent overfit 
        
        if is_train:
            rand_val = torch.rand_like(time_sampled) * 2.0 - 1.0 #random frame from -1 to 1
            rand_val = rand_val / self.max_t # shifting not more than 1 frame 
            time_sampled = time_sampled + rand_val
        return time_sampled

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

        """        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
        """

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

        app_mask_ray = weight[ray_valid]  > self.rayMarch_weight_thres

        if app_mask_ray.any():
            xyz_r = xyz_sampled[ray_valid][app_mask_ray]
            time_r = time_sampled[ray_valid][app_mask_ray]
            viewdir_r = viewdirs[ray_valid][app_mask_ray]
            app_features = self.compute_appfeature(xyz_r, time_r, sigma_feature[app_mask_ray], viewdir_r)
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

class AbsoluteAcivation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.abs(input)
