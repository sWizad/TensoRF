# re-implement an idea of sparse hash mlp
# experiement: try multisize hash grid, and same size hashgrid

import torch
import torch.nn.functional as F

try:
    import tinycudann as tcnn
except:
    pass 
from .tensoRF import TensorVMSplit, raw2alpha
import numpy as np


class SparseHashMLP(TensorVMSplit):
    """
    a drop-in replacement for for VM decompositon where we factor into sparse hashMLP instead
    """
    def __init__(self, *args, **kwargs):
        #self.hash_numfeatures = kwargs['hash_numfeatures']
        #self.sparsemlp_depth = [3,4,5,6,7,8,9,10,11]
        #self.sparsemlp_depth = [3,16,16,16,16]
        self.sparsemlp_depth = [3,4,8,16]
        self.sigma_activation = torch.relu
        print("Model: SparseHashMLP - big MLP from sparse hashgrid ")
        super(TensorVMSplit, self).__init__(*args, **kwargs)

    def init_svd_volume(self, res, device):
        # 
        self.weight_hashs = []
        self.bias_hashs = []
        self.weight_inds = [0]
        self.bias_inds = [0]
        num_feature = 3 #define for x,y,z
        base_resolution = 1024
        log2_hashmap_size = 8
        for i in range(len(self.sparsemlp_depth)-1):
            #log2_size = self.hash_logsizes[i]
            #predict weight
            input_size = self.sparsemlp_depth[i]
            output_size = self.sparsemlp_depth[i+1]
            weights, bias = get_weight_and_bias(input_size, output_size, log2_hashmap_size=log2_hashmap_size, base_resolution=base_resolution, device=device)
            self.weight_inds += [self.weight_inds[-1] + len(weights)]
            self.bias_inds += [self.bias_inds[-1] + len(bias)]
            self.weight_hashs += weights 
            self.bias_hashs += bias

        self.weight_hashs = torch.nn.ModuleList(self.weight_hashs)
        self.bias_hashs = torch.nn.ModuleList(self.bias_hashs)
        self.geo_feat_dim = self.sparsemlp_depth[-1] - 1


        # color network
        self.num_layers_color = 3        
        self.hidden_dim_color = 64

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
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": 3,
            },
        )
        """
        self.color_net = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dir.n_output_dims + self.geo_feat_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid()
        ).to(device)
        """
        """
        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": self.hidden_dim_color,
                "n_hidden_layers": self.num_layers_color - 1,
            },
        )
        """
        """
        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "degree": 4,
            },
        )
        self.color_net = torch.nn.Sequential(
            torch.nn.Linear(self.encoder_dir.n_output_dims + self.geo_feat_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3),
            torch.nn.Sigmoid()
        ).to(device)
        """

    def get_optparam_groups(self, lr_init = 0.01, lr_basis = 0.001):
        # lr_init_spatialxyz = lr_init
        # lr_init_network = lr_basis
        grad_vars = [
            # HahsGrid parameters
            {'name': 'weight_hashs', 'params': self.weight_hashs.parameters(), 'lr': lr_init},
            {'name': 'bias_hashs', 'params': self.bias_hashs.parameters(), 'lr': lr_init},
            # Network parameter
            {'name': 'color_net', 'params': self.color_net.parameters(), 'lr': lr_basis, 'weight_decay': 1e-6}          
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars
        
    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features[...,0]+self.density_shift)
        elif self.fea2denseAct == "relu":
            val = density_features[...,0]
            val = F.relu(val)
            return val        
        elif self.fea2denseAct == "abs":
            val = density_features[...,0]
            val = torch.abs(val)
            return val
        else:
            raise NotImplementedError()
    
    def compute_densityfeature(self, xyz_sampled):
        """
        @params xyz_sampled scale in [-1,1] #[num_ray, 3]
        @return sigma_feature #[num_ray]
        """
        xyz_sampled = (xyz_sampled + 1.0) / 2.0 #scale to [0,1]
        x = xyz_sampled # batch, 3
        for i in range(len(self.sparsemlp_depth)-1):
            num_ray = xyz_sampled.shape[0]
            input_size = self.sparsemlp_depth[i]
            output_size = self.sparsemlp_depth[i+1]
            w = get_hash(self.weight_hashs[self.weight_inds[i]:self.weight_inds[i+1]],xyz_sampled).type(xyz_sampled.dtype)
            w = w.view(xyz_sampled.shape[0],output_size, input_size).type(xyz_sampled.dtype)
            b = get_hash(self.bias_hashs[self.bias_inds[i]:self.bias_inds[i+1]],xyz_sampled)
            b = b.view(xyz_sampled.shape[0], output_size, 1)
            x = (w @ x[...,None]) + b
            x = x[..., 0]
            x = self.sigma_activation(x)
        return x.type(xyz_sampled.dtype)
    
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


def get_weight_and_bias(input_size, output_size, log2_hashmap_size=19, base_resolution=1024, device='cpu'):
    """
    generating weight and bias
    weight output size can be reshape to [input_size, output_size]
    bias output size can be reshape to [output_size, 1]
    """
    weight = []
    bias = []
    
    weight_pattern = get_weight_pattern(input_size * output_size)
    bias_pattern = get_weight_pattern(output_size)
    weight = get_weight_by_pattern(weight_pattern, log2_hashmap_size, base_resolution, device)
    bias = get_weight_by_pattern(bias_pattern, log2_hashmap_size, base_resolution, device)
    return weight, bias

def get_weight_by_pattern(patterns, log2_hashmap_size, base_resolution, device):
    weights = []
    for p in patterns:
        weights.append(tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
               "otype": "HashGrid",
               "n_levels": p[0],
               "n_features_per_level": p[1],
               "log2_hashmap_size": log2_hashmap_size, #every table depth has the same size
               "base_resolution": base_resolution,
               "per_level_scale": 1,
            },
        ).to(device))
    return weights


def get_weight_pattern(weight_count):
    MAX_LEVEL = 32
    LATENT_SHAPE = [8,4,2,1]
    pattern = []
    i = MAX_LEVEL
    while weight_count > 0 and i > 0:
        jcnt = 0
        while jcnt < len(LATENT_SHAPE):
            j = LATENT_SHAPE[jcnt]
            if weight_count >= i*j:
                pattern.append([i,j])
                weight_count -= i*j 
                continue
            jcnt += 1
        i -= 1
    return pattern

def get_hash(encoder_list,xyz_sampled):
    output = []
    num_ray = xyz_sampled.shape[0]
    for encoder in encoder_list:
        data = encoder(xyz_sampled)
        output.append(data.view(num_ray, -1))
    output = torch.cat(output, dim=-1)
    output = output.view(num_ray, -1)
    return output

"""
def get_hashgrid_pattern(weight_count)
    pattern = []
    while weight_count > 0:
        for b in [8,4,2,1]:
            if weight_count > b:
                pattenr.append(b)
                weight_count -= b
                break
    return pattern
"""