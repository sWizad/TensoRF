# convert tensorf checkpoint to a voxel numpy for  debugging
# currently support only density voxel

import argparse
import torch
import os
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--ckpt', required=True, type=str, help='checkpoint find location')
parser.add_argument('-o', '--density_file', default='density.npy', type=str, help='density file output path' )
parser.add_argument('-a', '--activation', default='relu.npy', type=str, help='activation function (relu, softplus, etc)')
args = parser.parse_args()

def main():
    if not os.path.exists(args.ckpt):
        raise Exception("checkpoint not found!!")
    pth = torch.load(args.ckpt) #kwargs, state_dict, alphaMask.shape, alphaMask.mask alphaMask.aabb
    #pth['state_dict'] #['denisity_plane.N','denisity_line.N', 'app_plane.N', 'app_line.N'] where N = 0.1.2, also, increading baissi_mat and renderModuleMLP
    density_plane, density_line = [], []
    for i in range(3):
        # we don't mine to use CPU right
        density_plane.append(pth['state_dict']['density_plane.{}'.format(i)].cpu())
        density_line.append(pth['state_dict']['density_line.{}'.format(i)].cpu()) 
    del pth #free GPU memory
    grid_shape = (density_plane[0].shape[2], density_plane[0].shape[3], density_line[0].shape[2])
    total_grid = torch.prod(torch.tensor(grid_shape))
    divder = [grid_shape[0] *grid_shape[1], grid_shape[1], 1]
    sigma = torch.zeros(*grid_shape)
    for i in tqdm(range(total_grid.item())):
        x = i // divder[0]
        y = i // divder[1] % grid_shape[2]
        z = i % grid_shape[2]
        for k in range(3):
            plane_coff = density_plane[k][0,:,x,y]
            line_coeff = density_line[k][0,:,z,0]
            sigma[x,y,z] = sigma[x,y,z] + torch.sum(plane_coff * line_coeff)
    if args.activation == 'relu':
        sigma = torch.relu(sigma)
    else:
        raise Exception("Unknow activation type")
    print('saving density...')
    np.save(args.density_file, sigma.numpy())



        

    

if __name__ == "__main__":
    main()