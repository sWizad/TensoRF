# hashgrid based model
import torch
import torch.nn.functional as F

import tinycudann as tcnn

from .tensoRF import TensorVMSplit

class HashGridDecomposition(TensorVMSplit):
    """
    a drop-in replacement for for VM decompositon where we factor into hashgrid instead
    """
    def __init__(self, *args, **kwargs):
        print("Model: HashGridDecomposition")
        super(TensorVMSplit, self).__init__(*args, **kwargs)
    
    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        print(self.density_line)
        print("Stop at initial")
        exit()


    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def get_optparam_groups(self, lr_init = 0.02, lr_basis = 0.001):
        raise NotImplementedError("Need to support hashgrid_soon") 
        # lr_init_spatialxyz = lr_init
        # lr_init_network = lr_basis
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):
        raise NotImplementedError("Need to implement how to input ")
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        """
        @params xyz_sampled scale in [-1,1] #[num_ray, 3]
        @return sigma_feature #[num_ray]
        """
        raise NotImplementedError("Need to implement how to input ")
        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        return app_features

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
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        """
        raise NotImplementedError()
        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
        """


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

    def feature2density(self, density_features):
        raise NotImplementedError()
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def density_L1(self):
        raise NotImplementedError("HashGrid shouldn't call this")

    def TV_loss_density(self, reg):
       raise NotImplementedError("HashGrid shouldn't call this")

    def TV_loss_app(self, reg):
       raise NotImplementedError("HashGrid shouldn't call this")

    def vectorDiffs(self, vector_comps):
        raise NotImplementedError("HashGrid shouldn't call this")

    def vector_comp_diffs(self):
        raise NotImplementedError("HashGrid shouldn't call this")
    
    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        raise NotImplementedError("HashGrid shouldn't call this")

    
# experiment 1: Given same number of parameters. Hashgrid or VM perform better


# experiment 2: Given only same number of parameters. Hashgrid or VM perform better?
#class HashGridDecompositionMatchResolution(HashGridDecomposition):
#    def __init__(self, *args, **kwargs):
#        print("Model: HashGridDecompositionMatchResolution")
#        super(TensorVMSplit, self).__init__(*args, **kwargs)

