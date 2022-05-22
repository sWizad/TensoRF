import torch
from .tensoRF import TensorVMSplit
import torch.nn.functional as F


class ExplicitGrid(TensorVMSplit):

    def __init__(self, aabb, gridSize, device, **kargs):
        super(ExplicitGrid, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        scale_volume = 0.1
        scale_app = 1
        self.density_volume = torch.nn.Parameter(scale_volume * torch.randn((1, 1, self.gridSize[2], self.gridSize[0], self.gridSize[1]), device=device)) #NCDHW
        self.app_volume = torch.nn.Parameter(scale_app * (torch.randn((1, 3, self.gridSize[2], self.gridSize[0], self.gridSize[1]), device=device) * 2.0 - 1.0)) #NCDHW
 
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [
            {'name':'density_volume','params':self.density_volume, 'lr': lr_init_spatialxyz},
            {'name':'app_volume','params':self.app_volume, 'lr': lr_init_spatialxyz},
        ]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars


    def vector_comp_diffs(self):
        raise NotImplementError("Explicit grid0oesnt support vector comp diff")
    
    def density_L1(self):
        return torch.mean(torch.abs(self.density_volume))
    
    def TV_loss_density(self, reg):
        return 1e-2 * volumeTV(self.density_volume)
        
    def TV_loss_app(self, reg):
        return 1e-2 * volumeTV(self.app_volume)

    def compute_densityfeature(self, xyz_sampled):
        """
        @params xyz_sampled: the xyz in [-1,1] shape #[num_ray,3]
        @return sigma_feature: the sigma before acitvate function shape #[num_ray]
        """
        grid = xyz_sampled[None,None,None,:,:] #NDHW3
        sigma_feature = F.grid_sample(self.density_volume, grid, align_corners=True) #NCDHW
        sigma_feature = sigma_feature[0,0,0,0,:] #[num_ray]
        return sigma_feature


    def compute_appfeature(self, xyz_sampled):
        """
        @params xyz_sampled: the xyz in [-1,1] shape #[num_ray,3]
        @return rgb_feature: the rgb_feature in [0,1] shape #[num_ray,3]
        """
        grid = xyz_sampled[None,None,None,:,:] #NDHW3
        rgb_feature = F.grid_sample(self.app_volume, grid, align_corners=True) #NCDHW
        rgb_feature = rgb_feature[0,:,0,0,:].permute(1,0) #[num_ray,3]
        return rgb_feature



    @torch.no_grad()
    def up_sampling_VM(self, volume, res_target):
        volume = F.interpolate(volume.data, size=(res_target[2], res_target[0], res_target[1]), mode='trilinear', align_corners=True)
        volume = torch.nn.Parameter(volume)
        return volume

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_volume = self.up_sampling_VM(self.density_volume, res_target)
        self.app_volume = self.up_sampling_VM(self.app_volume, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        density_volume = self.density_volume.data[...,t_l[2]:b_r[2], t_l[0]:b_r[0], t_l[1]:b_r[1]]
        app_volume = self.app_volume.data[...,t_l[2]:b_r[2], t_l[0]:b_r[0], t_l[1]:b_r[1]]
        
        self.density_volume = torch.nn.Parameter(density_volume)
        self.app_volume = torch.nn.Parameter(app_volume)

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




def volumeTV(x):
    """
    calculate total variation for 3 dimension
    @params 5D tensor that formatted for grid_sample which is BCDHW format
    """
    batch_size, channel_size, depth_size, height_size, width_size = x.shape
    count_h = torch.prod(torch.tensor(x[:,:,:,1:,:].shape))
    count_w = torch.prod(torch.tensor(x[:,:,:,:,1:].shape))
    count_d = torch.prod(torch.tensor(x[:,:,1:,:,:].shape))
    h_tv = torch.pow((x[:,:,:,1:,:]-x[:,:,:,:h_x-1,:]),2).sum()
    w_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:w_x-1]),2).sum()
    d_tv = torch.pow((x[:,:,:,:,1:]-x[:,:,:,:,:w_x-1]),2).sum()
    tv = (h_tv / count_h) + (w_tv / count_w) + (d_tv / count_d)
    return 2*tv*batch_size
