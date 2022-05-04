# experiment on croping meta dataset
import torch
import numpy as np 
import os
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm

from .meta import MetaVideoDataset, image_preprocess, center_poses, normalize, get_spiral, average_poses
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender

class MetaCropVideoDataset(MetaVideoDataset):
    def __init__(self, *args, **kwargs):
        print("MetaCropVideoDataset")
        #self.crop_info = [460, 360,  660, 560] #corner top-left, bottom-right
        self.crop_shift = [200, 200] #xy
        self.crop_info = np.array([
            [360, 460],
            [460, 490],
            [430, 490],
            [0, 0],
            [390, 480],
            [390, 480],
            [350, 500],
            [330, 480],
            [320, 470],
            [270, 480],
            [250, 500],
            [220, 420],
            [180, 360],
            [400, 370],
            [400, 400],
            [360, 400],
            [370, 380],
            [0, 0],
            [340, 380],
            [410, 460],
            [380,420],
        ])
        super().__init__(*args, **kwargs)

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        hwf = poses[:, :, -1]

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        #N_views, N_rots = 30, 2
        N_views, N_rots = self.max_t, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]   
        num_cameras = 21 #len(self.poses)# we fix number of input camera instead
        img_list = i_test if self.split != 'train' else list(set(np.arange(num_cameras)) - set(i_test))
        self.all_rays = []
        self.all_rgbs = []

        missing_view = 0
        image_paths = []
        for i, view in enumerate(img_list):
            if not os.path.exists(os.path.join(self.root_dir,f'frames/c{view:02d}')): 
                missing_view += 1
                continue #skip missing input view
            c2w = torch.FloatTensor(self.poses[view - missing_view])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            rays_o = rays_o.view(H,W,3)[self.crop_info[view, 1]:self.crop_info[view, 1] + self.crop_shift[1] , self.crop_info[view, 0]:self.crop_info[view,0] + self.crop_shift[0]].reshape((-1,3))
            rays_d = rays_d.view(H,W,3)[self.crop_info[view, 1]:self.crop_info[view, 1] + self.crop_shift[1] , self.crop_info[view, 0]:self.crop_info[view,0] + self.crop_shift[0]].reshape((-1,3))
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            for t in range(self.max_t):
                image_paths += [os.path.join(self.root_dir,f'frames/c{view:02d}/f{t+1:04d}.png')] #change to view instead, sometimes i is not consequtive [i_test]
                self.all_rays += [torch.cat([rays_o, rays_d, t*torch.ones_like(rays_o[...,0:1])], 1)]  # (h*w, 7) 
        
        # pararell loading and resizeing files
        print("reading {} images files...".format(self.split))
        image_preprocess_fn = partial(image_preprocess, self.downsample, self.img_wh)
        with Pool(os.cpu_count()) as p:
            self.all_rgbs = list(tqdm(p.imap(image_preprocess_fn, image_paths), total=len(image_paths)))

        self.directions = self.directions
        self.img_wh = np.array([self.crop_shift[1], self.crop_shift[0]])

        # this conversation should in image_preprocess. however, it freeze the process for no reason, so i doing it in main thread instead
        for i in range(len(self.all_rgbs)):
            self.all_rgbs[i] = self.transform(self.all_rgbs[i])
            self.all_rgbs[i] = self.all_rgbs[i][:, self.crop_info[view, 1]:self.crop_info[view, 1] + self.crop_shift[1] , self.crop_info[view, 0]:self.crop_info[view,0] + self.crop_shift[0]]
            self.all_rgbs[i] = self.all_rgbs[i].reshape((3, -1)).permute(1, 0)  # (h*w, 3) RGB


        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        
       