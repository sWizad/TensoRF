import torch
import torch.nn.functional as F
import numpy as np

from utils import printlog
from .tensoRF import TensorVMSplit
from .tensorBase import raw2alpha, AlphaGridMask

class TensoRFdepth(TensorVMSplit):


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
        #np_xyz = xyz_sampled.cpu().detach().numpy()
        #print(np.max(np_xyz,(0,1)))
        #print(np.min(np_xyz,(0,1)))
        #pdb.set_trace()

        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid


        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)


        #we foce alpha to be one anywhere that deepth exist
        alpha = torch.zeros_like(z_vals)
        weight = torch.zeros_like(z_vals)
        bg_weight = torch.zeros_like(z_vals)
        alpha[ray_valid] = 1.0
        weight[ray_valid] = 1.0 
        bg_weight[~ray_valid] = 1.0 

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

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1, depths=None):
        #IT ALWAY HAS SINGLE SAMPLE         
        # Expceted rays_pts [4096,1,3] / depths [4096,1] / mask_outbbox [4096,1]
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * depths[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)
        return rays_pts, depths, ~mask_outbbox

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        return self.aabb

    @torch.no_grad()
    def shrink(self, new_aabb):
        pass

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        accept_mask = all_rays[:, 6] < 1e5
        return all_rays[accept_mask], all_rgbs[accept_mask]