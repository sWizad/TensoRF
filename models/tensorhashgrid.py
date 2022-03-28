# TensorHashGrid.py - triplane model 


import torch
import torch.nn.functional as F

import tinycudann as tcnn
from .tensoRF import TensorVMSplit, raw2alpha
import numpy as np

class TensorHashGrid(TensorVMSplit):
    def __init__(self, *args, **kwargs):
        print("Model: TensorHashGrid ")
        self.grid_level = kwargs['grid_level']
        self.grid_feature_per_level = kwargs['grid_feature_per_level']
        self.grid_hash_log2 = kwargs['grid_hash_log2']
        self.grid_base_resolution = kwargs['grid_base_resolution']
        self.grid_level_scale = kwargs['grid_level_scale']

        super().__init__(*args, **kwargs)



    def init_svd_volume(self, res, device):
        # define network component
        encoders = []        
        num_density = np.sum(np.array(self.density_n_comp))
        num_apperance = np.sum(np.array(self.app_n_comp))
        for i in range(num_density + num_apperance):
            encoders.append(
                tcnn.Encoding(
                    n_input_dims=2,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 1,
                        "n_features_per_level": 1, #must be 1,2,4,8
                        "log2_hashmap_size": np.ceil(np.log(90000) / np.log(2)), #self.grid_hash_log2,
                        "base_resolution": 300, #grid_resolution
                        "per_level_scale": 1,
                    },
                ).to(device)
            )
            

        self.density_encoders = torch.nn.ModuleList(encoders[:num_density])
        self.apperance_encoders = torch.nn.ModuleList(encoders[num_density:])
        self.basis_mat = torch.nn.Linear(num_apperance, self.app_dim, bias=False, device=device)

    def get_optparam_groups(self, lr_init = 0.01, lr_basis = 0.001):
        # lr_init_spatialxyz = lr_init
        # lr_init_network = lr_basis
        grad_vars = [
            # HahsGrid parameters
            {'name': 'density_encoders', 'params': self.density_encoders.parameters(), 'lr': lr_init},
            {'name': 'apperance_encoders', 'params': self.apperance_encoders.parameters(), 'lr': lr_init},
            # apperant feature network
            {'name': 'basis_mat', 'params': self.basis_mat.parameters(), 'lr': lr_basis}
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_basis}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):
        """
        @params xyz_sampled scale in [-1,1] #[num_ray, 3]
        @return sigma_feature #[num_ray, num_level]
        """
        xyz_sampled = (xyz_sampled + 1.0) / 2.0 #scale to [0,1]
        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        shift_density = 0
        for i in range(3):
            coordinate_plane = xyz_sampled[..., self.matMode[i]]
            for j in range(self.density_n_comp[i]):
                idx = shift_density + j
                plane_coef_point = self.density_encoders[idx](coordinate_plane)
                plane_coef_point = plane_coef_point.type(xyz_sampled.dtype)
                sigma_feature = sigma_feature + torch.sum(plane_coef_point)
            shift_density += self.density_n_comp[i]

        return sigma_feature.type(xyz_sampled.dtype)

    def compute_appfeature(self, xyz_sampled):
        features = []
        shift_density = 0
        for i in range(3):
            coordinate_plane = xyz_sampled[..., self.matMode[i]]
            for j in range(self.app_n_comp[i]):
                plane_coef_point = self.apperance_encoders[i](coordinate_plane)
                plane_coef_point = plane_coef_point.type(xyz_sampled.dtype)
                features.append(plane_coef_point)  
        features = torch.cat(features,dim=-1)
        app_features = self.basis_mat(features)        
        return app_features

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
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
    def upsample_volume_grid(self, res_target):
        # Hashgrid is not support UPsample size
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')
