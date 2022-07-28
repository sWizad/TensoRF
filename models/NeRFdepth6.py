import torch
import torch.nn.functional as F
import numpy as np

from utils import printlog
from .NeRF import AbsoluteAcivation
from .tensoRFdepth import TensoRFdepth
from .tensorBase import raw2alpha, AlphaGridMask, positional_encoding

class NeRFdepth6(TensoRFdepth):

    def __init__(self, *args, **kwargs):
       
        self.nerf_hidden, self.pos_pe, self.view_pe = kwargs['featureC'], kwargs['pos_pe'], kwargs['view_pe']
        print("NeRFdepth: Visualize power of nerf view depenedent effect")
        super().__init__(*args, **kwargs) #initail super class

    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        print("lr_init_spatialxyz: ", lr_init_spatialxyz)
        print("lr_init_network: ", lr_init_network)
        grad_vars = [
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

        self.color_net = torch.nn.Sequential(
            torch.nn.Linear(view_pe + pos_pe, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 3),
            torch.nn.Sigmoid(),
        ).to(device)
        
   

    def compute_appfeature(self, xyz_sampled, sigma_features, view_directions):
        view_ped = positional_encoding(xyz_sampled, self.view_pe)
        pos_ped =  positional_encoding(view_directions, self.pos_pe)
        x = torch.cat([view_ped, pos_ped], dim=-1)
        x = self.color_net(x)
        return x

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        depths = rays_chunk[:, 6:7]
        if ndc_ray:
            raise NotImplementError("Unsupported NDC format")
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples, depths = depths)
            #dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            dists = z_vals
            
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
 

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            #sigma_features = self.compute_densityfeature(xyz_sampled[ray_valid])
            sigma_features = None
            app_features = self.compute_appfeature(xyz_sampled[ray_valid], sigma_features, viewdirs[ray_valid])
            rgb[ray_valid] = self.renderModule(xyz_sampled[ray_valid], viewdirs[ray_valid], app_features)
            

        #we foce alpha to be one anywhere that deepth exist
        alpha = torch.zeros_like(z_vals)
        weight = torch.zeros_like(z_vals)
        bg_weight = torch.zeros_like(z_vals)
        alpha[ray_valid] = 1.0
        weight[ray_valid] = 1.0 
        bg_weight[~ray_valid] = 1.0 

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])

        
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]

        return rgb_map, depth_map # rgb, sigma, alpha, weight, bg_weight

class NeRFdepth8(NeRFdepth6):
    def init_svd_volume(self, res, device):
        # neural network for handleing
        # predefine variable 
        hidden_size =  self.nerf_hidden
        pos_pe = (self.pos_pe * 2) * 3 # input is x,y,z
        view_pe = ((self.view_pe * 2) * 3) #input is vx,vy,vz

        self.color_net = torch.nn.Sequential(
            torch.nn.Linear(view_pe + pos_pe, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 3),
            torch.nn.Sigmoid(),
        ).to(device)