# Shiny dataset but enable 

import os
import glob
import numpy as np
import torch
from PIL import Image

from .shiny import ShinyDataset, center_poses, normalize, get_spiral, average_poses
from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender


class ShinyFewDataset(ShinyDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(args[0][0], split=args[1]['split'], downsample=args[1]['downsample'], is_stack=args[1]['is_stack'], ndc_ray=args[1]['ndc_ray'], max_t=args[1]['max_t'], hold_every=args[1]['hold_every'])

    def get_image_ids(self):
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))
        if self.split == 'train':
            img_list = img_list[::self.pick_train_every]
        print("NUM_IMAGE: {} / {} images".format(self.split, len(img_list)))
        return img_list
        
    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        # load full resolution image then resize
        if self.split in ['train', 'test']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        #poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        poses = poses_bounds[:, :-2].reshape(-1, 3, 4)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        intrinsic_arr = np.load(os.path.join(self.root_dir, 'hwf_cxcy.npy')) 
        hwf = poses[:, :, -1]
        # Step 1: rescale focal length according to training resolution
        #H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        H, W ,self.focal = intrinsic_arr[:3,0]
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)
        #pdb.set_trace()

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()
        if hasattr(self, 'override_near'):
            near_original = self.override_near 

        scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor

        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, rads_scale=0.1, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)

        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        img_list = self.get_image_ids()

        # use first N_images-1 to train, the LAST is val
        self.all_rays = []
        self.all_rgbs = []
        for i in img_list:
            image_path = self.image_paths[i]
            c2w = torch.FloatTensor(self.poses[i])

            img = Image.open(image_path).convert('RGB')
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
            self.all_rgbs += [img]
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
                # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)


class ShinyFew1(ShinyFewDataset):
    def __init__(self, *args, **kwargs):
        self.self.override_near = 40
        self.pick_train_every = 1
        super().__init__(args,kwargs)

class ShinyFew5(ShinyFewDataset):
    def __init__(self, *args, **kwargs):
        self.self.override_near = 40
        self.pick_train_every = 5
        super().__init__(args,kwargs)

class ShinyFew10(ShinyFewDataset):
    def __init__(self, *args, **kwargs):
        self.self.override_near = 40
        self.pick_train_every = 10
        super().__init__(args,kwargs)

class ShinyFew15(ShinyFewDataset):
    def __init__(self, *args, **kwargs):
        self.self.override_near = 40
        self.pick_train_every = 15
        super().__init__(args,kwargs)

class ShinyFew20(ShinyFewDataset):
    def __init__(self, *args, **kwargs):
        self.self.override_near = 40
        self.pick_train_every = 20
        super().__init__(args,kwargs)

class ShinyFern400(ShinyFewDataset):
    def __init__(self, *args, **kwargs):
        self.pick_train_every = 20
        super().__init__(args,kwargs)
        self.near_far = [0.0, 1.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def get_image_ids(self):
        TOTAL_FILE = 420
        TOTAL_ORIGINAL_FILE = 20
        i_test = np.arange(0, TOTAL_ORIGINAL_FILE, self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(TOTAL_FILE)) - set(i_test)) 
        print("NUM_IMAGE: {} / {} images".format(self.split, len(img_list)))
        return img_list


















