# design for predict the image without load the rgb into dataset for memory efficient rendering

import torch
import os, sys
import numpy as np
from opt import config_parser
from utils import colored_hook
from dataLoader import dataset_dict
from renderer import *
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import skimage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def chuncked_render(model, rays, white_bg=True, data_parallel=False):
    """
    @params rays #[rays, 6]
    @return rgbs, depths #[rays, 3] / #[rays, 1]
    """
    chucked_size = 200*200*4
    rgbs, depths = [], []
    if data_parallel:
        model = torch.nn.DataParallel(model)
    for chunck in torch.split(rays, chucked_size):
        rgb_map, depth_map = model(chunck.cuda(), is_train=False, white_bg=white_bg, ndc_ray=False, N_samples=-1)
        rgbs.append(rgb_map)
        depths.append(depth_map)
    rgbs = torch.cat(rgbs, dim=0)
    depths = torch.cat(depths, dim=0)
    return rgbs, depths


def main(args):
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train)
    white_bg = test_dataset.white_bg

    if not os.path.exists(args.ckpt):
        print('require ckpt path to render image!!')
        return

    # load checkpoint    
    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']     
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    W,H = test_dataset.img_wh
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)
    render_dir = os.path.join(args.basedir, args.expname,'render') 
    depth_dir  = os.path.join(args.basedir, args.expname,'render_depth') 
    os.makedirs(render_dir,exist_ok=True)
    os.makedirs(depth_dir,exist_ok=True)
    with torch.no_grad():
        for i, sample_batched in enumerate(tqdm(dataloader)):
            if sample_batched['rays'].shape[0] != 1: raise Exception('currently support only batch_size = 1')
            # render image
            rgbs, depths = chuncked_render(tensorf, torch.squeeze(sample_batched['rays']), white_bg=white_bg, data_parallel=False)
            rgbs = rgbs.view(H, W, 3).cpu().numpy()
            depths = depths.view(H, W).cpu().numpy()
            # save image
            skimage.io.imsave(os.path.join(render_dir,f"{i:04d}.png"),skimage.img_as_ubyte(rgbs))
            np.save(os.path.join(depth_dir,f"{i:04d}.npy"), depths)
        

    

if __name__ == '__main__':
    sys.excepthook = colored_hook(os.path.dirname(os.path.realpath(__file__)))
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20121202)
    np.random.seed(20121202)
    args = config_parser()
    main(args)