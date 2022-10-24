import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np

from .ray_utils import get_ray_directions,  get_rays


class BlenderPredictDataset(Dataset):
    def __init__(self, datadir, downsample=1.0, **kwargs):
        print("======================================")
        print("====== BlenderPredictDataset =========")
        print("======================================")
        self.root_dir = datadir
        self.split = 'test'
        self.near_far = [2.0,6.0]
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.white_bg = True
        self.read_meta()
        
    def read_meta(self):
        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)
        
        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)

        # load ray direction
        poses = []
        for frame in self.meta['frames']:
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            poses.append(torch.FloatTensor(pose))
        self.poses = poses

    def __len__(self):
        # return number of image
        return len(self.poses)

    def __getitem__(self, idx):
        rays_o, rays_d = get_rays(self.directions, self.poses[idx])  # both (h*w, 3)
        rays = torch.cat([rays_o, rays_d], -1)
         # (h*w, 6)
        return {
            'rays': rays
        }