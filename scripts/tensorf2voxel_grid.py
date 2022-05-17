# convert tensorf checkpoint to a voxel numpy for  debugging
# currently support only density voxel

import argparse
import torch
import numpy as np
import os
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ckpt', required=True, type=str, help='checkpoint find location')
parser.add_argument('-o', '--density_file', default='density.npy', type=str, help='density file output path' )
parser.add_argument('-a', '--activation', default='relu', type=str, help='activation function (relu, softplus, etc)')
args = parser.parse_args()

def main():
    with torch.no_grad():
        if not os.path.exists(args.ckpt):
            raise Exception("checkpoint not found!!")
        pth = torch.load(args.ckpt) #kwargs, state_dict, alphaMask.shape, alphaMask.mask alphaMask.aabb
        #pth['state_dict'] #['denisity_plane.N','denisity_line.N', 'app_plane.N', 'app_line.N'] where N = 0.1.2, also, increading baissi_mat and renderModuleMLP
        density_plane, density_line = [], []
        for i in range(3):
            # we don't mine to use CPU right
            density_plane.append(pth['state_dict']['density_plane.{}'.format(i)].cpu())
            density_line.append(pth['state_dict']['density_line.{}'.format(i)].cpu()) 
            print(density_plane[i].shape)
        
        del pth #free GPU memory
        
        # we multiply grid to geter for faster export
        sigma = torch.zeros(density_plane[0].shape[2], density_plane[0].shape[3], density_line[0].shape[2])
        H,W,D = sigma.shape[0], sigma.shape[1], sigma.shape[2]

        # combine from X axis 
        plane_coeff = density_plane[0][0].view(-1, H, W, 1).expand(-1,-1,-1,D)
        line_coeff = density_line[0][0,:,None,None,:,0].expand(-1,H,W,-1)
        combi = torch.sum(plane_coeff * line_coeff,dim=0)
        sigma = sigma + combi

        #combine from Y Axis
        plane_coeff = density_plane[1][0].view(-1, D, W, 1).expand(-1,-1,-1,H)
        line_coeff = density_line[1][0,:,None,None,:,0].expand(-1,D,W,-1) 
        combi = torch.sum(plane_coeff * line_coeff,dim=0) #DWH
        sigma = sigma + combi.permute(2,1,0)

        #combine from Z Axis
        plane_coeff = density_plane[2][0].view(-1, D, H, 1).expand(-1,-1,-1,W)
        line_coeff = density_line[2][0,:,None,None,:,0].expand(-1,D,H,-1) 
        combi = torch.sum(plane_coeff * line_coeff,dim=0) #DHW
        sigma = sigma + combi.permute(1,2,0)

        print("FINISH CONVERT")
        if args.activation == 'relu':
            sigma = torch.relu(sigma)
        else:
            raise Exception("Unknow activation type")
        print('saving density...')
        sigma = sigma.permute(2,0,1) #Webgl use CHW format
        np.save(args.density_file, sigma.numpy())

        

    

if __name__ == "__main__":
    main()