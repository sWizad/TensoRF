from .tensoRF import *
import pdb

class AlphaSphereMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume, origin, sph_box = [-1.4, 1.4], sph_frontback = [10, 300]):
        super(AlphaSphereMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)
        
        self.origin   = origin
        self.sph_box  = sph_box
        self.sph_frontback = sph_frontback

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def sample_direct_alpha(self, normalize_xyz):
        alpha_vals = F.grid_sample(self.alpha_volume, normalize_xyz.view(1,-1,1,1,3), align_corners=True).view(-1)
        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        radias = torch.norm(xyz_sampled - self.origin, dim=-1)
        #r_depth = (radias - self.sph_frontback[0]) / (self.sph_frontback[1] - self.sph_frontback[0]) *2 -1
        r_depth = (self.sph_frontback[0]/radias-1)/(self.sph_frontback[0]/self.sph_frontback[1]-1)
        r_depth = r_depth * 2 - 1

        unit_sphere_point = (xyz_sampled - self.origin) / ( radias[...,None] + 1e-8)

        xy_point = (torch.asin(unit_sphere_point[...,:2]) - self.sph_box[0]) / (self.sph_box[1] - self.sph_box[0]) *2 -1

        return torch.concat([xy_point, r_depth[...,None]],-1)
        #return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1

class TensorSph(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorSph, self).__init__(aabb, gridSize, device, **kargs)
        if 'origin' in kargs:
            self.origin = kargs['origin']
        print('--------------------Spherical TENSOR----------------------')
    
    def set_origin(self,origin,sph_box,sph_frontback):
        self.origin = torch.Tensor(origin[:,0]).cuda()
        self.sph_box = sph_box
        self.sph_frontback = sph_frontback
        self.near_far = sph_frontback  #This may wrong in furture
        '''
        if False:
            self.sph_box = [-1., 1.]
            self.sph_frontback = [10, 300]
            self.near_far = [10,300]
        else:
            self.sph_box = [-1.8, 1.8]
            self.sph_frontback = [1, 6]
            self.near_far = [1,6]
        '''


    def compute_appfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        #
        plane_coef_point,line_coef_point = [],[]
        view_coef_point0, view_coef_point1 = [], []
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)

        return self.basis_mat((plane_coef_point * line_coef_point).T)

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_direct_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
            

        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)

        if alpha_mask.any():
            sigma_feature = self.compute_densityfeature(xyz_locs[alpha_mask])
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        

        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])

        return alpha

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):

        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        dense_xyz = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, gridSize[0]),
            torch.linspace(-1, 1, gridSize[1]),
            torch.linspace(-1, 1, gridSize[2]),
        ), -1).to(self.device)
        #dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[2]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.distance_scale*self.aabbDiag).view((gridSize[1], gridSize[0]))
        alpha = alpha.clamp(0,1)[None,None]


        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaSphereMask(self.device, self.aabb, alpha, self.origin, self.sph_box, self.sph_frontback)
        #pdb.set_trace()
        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    def normalize_coord(self,xyz_sampled):
        radias = torch.norm(xyz_sampled - self.origin, dim=-1)
        #r_depth = (radias - self.sph_frontback[0]) / (self.sph_frontback[1] - self.sph_frontback[0]) *2 -1
        r_depth = (self.sph_frontback[0]/radias-1)/(self.sph_frontback[0]/self.sph_frontback[1]-1)
        r_depth = r_depth * 2 - 1

        unit_sphere_point = (xyz_sampled - self.origin) / ( radias[...,None] + 1e-8)
        #xy_point = (unit_sphere_point[...,:2] - self.sph_box[0]) / (self.sph_box[1] - self.sph_box[0]) *2 -1

        xy_point = (torch.asin(unit_sphere_point[...,:2]) - self.sph_box[0]) / (self.sph_box[1] - self.sph_box[0]) *2 -1
        #pdb.set_trace()

        return torch.concat([xy_point, r_depth[...,None]],-1)

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        #interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        interpx = (1/torch.linspace(1, near / far,  N_samples) * near).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):

        # sample points
        viewdirs = rays_chunk[:, 3:6]
        if True:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        #view = viewdirs.cpu().numpy()
        ray_valid = torch.ones_like(ray_valid)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid
            #pdb.set_trace()

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            #xyz_sampled = self.spherical_mapping(xyz_sampled)
            #pdb.set_trace()
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid])

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