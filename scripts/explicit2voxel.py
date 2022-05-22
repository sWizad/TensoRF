# convert tensorf checkpoint to a voxel numpy for  debugging
# currently support only density voxel

import argparse
import torch
import numpy as np
import os
from tqdm.auto import tqdm
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ckpt', required=True, type=str, help='checkpoint find location')
parser.add_argument('-o', '--density_file', default='density.npy', type=str, help='density file output path' )
parser.add_argument('-a', '--activation', default='relu', type=str, help='activation function (relu, softplus, etc)')
args = parser.parse_args()

class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1

def load_alpha(ckpt, device = 'cpu'):
    if not 'alphaMask.shape' in ckpt: return None
    length = np.prod(ckpt['alphaMask.shape'])
    alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
    alphaMask = AlphaGridMask(device, ckpt['alphaMask.aabb'].to(device), alpha_volume.float().to(device))
    return alphaMask 

def main():
    with torch.no_grad():
        if not os.path.exists(args.ckpt):
            raise Exception("checkpoint not found!!")
        pth = torch.load(args.ckpt) #kwargs, state_dict, alphaMask.shape, alphaMask.mask alphaMask.aabb
        sigma = pth['state_dict']['density_volume'][0,0].cpu()
        alphaMask = load_alpha(pth)
      
        # Need to filter with alphaMask first
        if alphaMask is not None:
            print("Applying alphamask")
            #torch.Size([1, 1, 140, 235, 211])
            # sigma: torch.Size([471, 786, 706]) [z,x,y]
            grid_x, grid_y, grid_z = torch.meshgrid(
                torch.linspace(-1,1,sigma.shape[1]),
                torch.linspace(-1,1,sigma.shape[2]),
                torch.linspace(-1,1,sigma.shape[0]),
                indexing='ij'
            )
            grid = torch.cat([ grid_x[...,None], grid_y[...,None], grid_z[...,None]], dim=-1) #torch.Size([786, 706, 471, 3])
            #grid = grid.permute(2,1,0,3)
            grid = grid.permute(2,0,1,3)
            grid = grid[None]
            mask_vals = F.grid_sample(alphaMask.alpha_volume.permute(0,1,2,4,3), grid, align_corners=True)[0,0]
            sigma[mask_vals == 0.0] = 0.0 #clear out sigma that mask
            #sigma = alphaMask.alpha_volume[0,0]
            

        print("FINISH CONVERT :O")
        if args.activation == 'relu':
            sigma = torch.relu(sigma)
        elif args.activation == 'softplus':
            density_shift = -10
            sigma = F.softplus(sigma+density_shift)
        else:
            raise Exception("Unknow activation type")
        print('saving density...')
        np.save(args.density_file, sigma.numpy())

        

    

if __name__ == "__main__":
    main()