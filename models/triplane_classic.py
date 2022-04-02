# a reimplementation of triplane

import torch
import torch.nn.functional as F

import tinycudann as tcnn
from .tensoRF import TensorVMSplit, raw2alpha
import numpy as np

class TriPlaneClassic(TensorVMSplit):
    """
    a drop-in replacement for for VM decompositon where we factor into hashgrid instead
    """
    def __init__(self, *args, **kwargs):
        print("Model: Triplane (Classic) ")
        super(TensorVMSplit, self).__init__(*args, **kwargs)
    
    def init_svd_volume(self, res, device):
        #predefine config
        num_layers = 2
        hidden_dim = 64
        geo_feat_dim = 15
        num_layers_color = 3
        log2_hashmap_size = 19
        hidden_dim_color = 64
        bound = 1
        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        # define network component
        encoders = []
        for i in range(3):
            encoders.append(
                tcnn.Encoding(
                    n_input_dims=2,
                    encoding_config={
                        "otype": "HashGrid",
                        "n_levels": 16,
                        "n_features_per_level": 2,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": 16,
                        "per_level_scale": per_level_scale,
                    },
                ).to(device)
            )
        self.encoder = torch.nn.ModuleList(encoders)
        self.sigma_net = tcnn.Network(
            n_input_dims= 32 * 3,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        ).to(device)
         # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def init_one_svd(self, n_component, gridSize, scale, device):   
        raise NotImplementedError("No need to init SVD") 


    def get_optparam_groups(self, lr_init = 0.01, lr_basis = 0.001):
        # lr_init_spatialxyz = lr_init
        # lr_init_network = lr_basis
        grad_vars = [
            # HahsGrid parameters
            {'name': 'hashgrid_encoder', 'params': self.encoder.parameters(), 'lr': lr_init},
            # Network parameter
            {'name': 'sigma_net', 'params': self.sigma_net.parameters(), 'lr': lr_basis, 'weight_decay': 1e-6},
            {'name': 'color_net', 'params': self.color_net.parameters(), 'lr': lr_basis, 'weight_decay': 1e-6}          
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):
        raise NotImplementedError("Need to implement how to input ")
        return sigma_feature, app_features

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features[...,0]+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features[...,0])
        else:
            raise NotImplementedError()

    def compute_densityfeature(self, xyz_sampled):
        """
        @params xyz_sampled scale in [-1,1] #[num_ray, 3]
        @return sigma_feature #[num_ray]
        """
        xyz_sampled = (xyz_sampled + 1.0) / 2.0 #scale to [0,1]
        features = []
        for i in range(3):
            coordinate_plane = xyz_sampled[..., self.matMode[i]]
            features.append(self.encoder[i](coordinate_plane))
        features = torch.cat(features,dim=-1)
        sigma_feature = self.sigma_net(features)
        return sigma_feature.type(xyz_sampled.dtype)

    def compute_appfeature(self, xyz_sampled, density_features, viewdirs):
        #xyz_sampled = (xyz_sampled + 1.0) / 2.0 #scale to [0,1]
        #xyz_sampled = xyz_sampled[None] #[B, num_ray, 3]
        geo_feat = density_features[...,1:]  #[B, num_ray, 15]
        d = self.encoder_dir(viewdirs)
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        color = torch.sigmoid(h)
        return color.type(viewdirs.dtype)

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        # Hashgrid is not support UPsample size
        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')


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
        #pdb.set_trace()


        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        #app_mask = weight > self.rayMarch_weight_thres
        app_mask_ray = weight[ray_valid]  > self.rayMarch_weight_thres

        if app_mask_ray.any():
            xyz_r = xyz_sampled[ray_valid][app_mask_ray]
            viewdir_r = viewdirs[ray_valid][app_mask_ray]
            app_features = self.compute_appfeature(xyz_r, sigma_feature[app_mask_ray], viewdir_r)
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

    def density_L1(self):
        raise NotImplementedError("HashGrid3D shouldn't call this")

    def TV_loss_density(self, reg):
       raise NotImplementedError("HashGrid3D shouldn't call this")

    def TV_loss_app(self, reg):
       raise NotImplementedError("HashGrid3D shouldn't call this")

    def vectorDiffs(self, vector_comps):
        raise NotImplementedError("HashGrid3D shouldn't call this")

    def vector_comp_diffs(self):
        raise NotImplementedError("HashGrid3D shouldn't call this")
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        raise NotImplementedError("HashGrid3D shouldn't call this")
