import torch
from torch.utils.data import Dataset
import glob, json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation, Slerp
import pdb

from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.eul
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def webGLspiralPath(ref_rotation, ref_translation, dmin, dmax, total_frame = 120, spin_radius = 10, total_spin = 1):
  spin_speed = 2*np.pi / total_frame * total_spin
  render_poses = []
  # matrix conversation helper
  def dcm_to_4x4(r,t):
    camera_matrix = np.zeros((4,4),dtype=np.float32)
    camera_matrix[:3,:3] = r
    if len(t.shape) > 1:
      camera_matrix[:3,3:4] = t
    else:
      camera_matrix[:3,3] = t
    camera_matrix[3,3] = 1.0
    return camera_matrix

  for i in range(total_frame):
    anim_time = spin_speed * i
    leftright = np.sin(anim_time) * spin_radius / 500.0
    updown = np.cos(anim_time) * spin_radius / 500.0
    r = ref_rotation
    t = ref_translation
    cam = dcm_to_4x4(r,t)
    dist = (dmin + dmax) / 2.0
    translation_matrix = dcm_to_4x4(np.eye(3), np.array([0,0, -dist]))
    translation_matrix2 = dcm_to_4x4(np.eye(3), np.array([0,0, dist]))
    euler_3x3 = Rotation.from_euler('yxz', [leftright, updown, 0])
    try:
        euler_3x3 = euler_3x3.as_matrix() #newer version of scipy
    except:
        euler_3x3 = euler_3x3.as_dcm() #compatibility to old version
    euler_4x4 = dcm_to_4x4(euler_3x3, np.array([0.0,0.0,0.0]))
    output = translation_matrix2 @ euler_4x4 @  translation_matrix @ cam
    output = output.astype(np.float32)
    r = output[:3, :3]
    t = output[:3, 3:4]
    #render_poses[i] = output#{'r': r, 't': t}
    render_poses.append(output)
  return render_poses

def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    #up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    #dt = 0.75
    #close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    #focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    #zdelta = near_fars.min() * .2
    #tt = c2ws_all[:, :3, 3]
    #rads = np.percentile(np.abs(tt), 90, 0) * 0
    #render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    render_poses = webGLspiralPath(c2w[:,:3],c2w[:,3],near_fars.min(),near_fars.max(), total_frame=N_views)
    #pdb.set_trace()
    return np.stack(render_poses)

def get_ray_directions_blender2(H, W, K):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]+0.5
    i, j = grid.unbind(-1)
    directions = torch.stack([i, j, torch.ones_like(i)], -1)
    K_inv = torch.Tensor(np.linalg.inv(K))
    directions = K_inv @ directions[...,None]
    return directions[...,0]

def inv_fisheye(fishpoint, depth, radial_distortion):
    """inverse fisheye"""
    radias = torch.sqrt(fishpoint[...,0] * fishpoint[...,0] + fishpoint[...,1] * fishpoint[...,1]) #[H,W]
    theta = torch.zeros_like(radias)
    for i in range(3):
        theta2 = theta * theta
        f = theta*(1.0 + theta2*(radial_distortion[0] + theta2 * radial_distortion[1])) - radias
        fp = 1.0 + theta2*(3*radial_distortion[0] + 5*theta2 * radial_distortion[1])
        theta = theta - f/fp
    ras = torch.tan(theta) * depth  #[H,W]
    output = torch.stack([fishpoint[...,0] * ras / (radias + 1e-5),
                        fishpoint[...,1] * ras / (radias + 1e-5),
                        torch.ones_like(ras) * depth], -1)
    return output

def get_rays(directions, c2w):
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o

class DeepvvDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        #         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        self.near_far = [0, 2.0]
        #self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        #self.scene_bbox = torch.tensor([[-15.0, -15.0, -15.0], [15.0, 15.0, 15.0]])
        self.scene_bbox = torch.tensor([[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):

        #with open('data/deepview_video/01_Welder/models.json') as f:
        with open(os.path.join(self.root_dir,'models.json')) as f:
            view_list = json.load(f)
        H, W = view_list[0]['height'],  view_list[0]['width']
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.near_fars = np.array([0.1,10])

        self.all_rays = []
        self.all_rgbs = []
        self.poses = []
        self.directions = None
        #i_test = np.arange(0, self.poses.shape[0], self.hold_every)
        i_test = [i for i in range(0, len(view_list), self.hold_every)]
        #i_train = [i if i not in i_test for i in range(len(view_list)) ]
        new_list = []
        for i, view in enumerate(view_list):
            if self.split != 'train' and i in i_test:
                new_list.append(view)
            elif self.split == 'train' and i not in i_test:
                new_list.append(view)

        #pdb.set_trace()
        for i, view in enumerate(new_list):
            image_path = os.path.join(self.root_dir,f'frames/c{i+1:02d}/f0001.png')
            img = Image.open(image_path).convert('RGB')

            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (3, h, w)

            img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

            self.all_rgbs += [img]
            #
            
            R = Rotation.from_rotvec(view['orientation']).as_matrix()
            t = np.array(view['position'])[:, np.newaxis]
            c2w = np.concatenate((R, -np.dot(R, t)), axis=1)
            c2w = torch.FloatTensor(c2w)
            K = np.array([[view['focal_length']*self.img_wh[0]/W, 0.0, view['principal_point'][0]*self.img_wh[0]/W],
                          [0.0, view['focal_length']*self.img_wh[1]/H, view['principal_point'][1]*self.img_wh[1]/H],
                          [0.0, 0.0, 1.0]])
            radial_distortion = np.array(view['radial_distortion'])
            
            directions2 = get_ray_directions_blender2(self.img_wh[1], self.img_wh[0], K)
            output1 = inv_fisheye(directions2,1.0,radial_distortion)#.view(-1, 3)
            output2 = inv_fisheye(directions2,2.0,radial_distortion)#.view(-1, 3)
            #output3 = inv_fisheye(directions2,3.0,radial_distortion)#.view(-1, 3)
            out1 = (c2w[:,:3].T @ (output1 - c2w[:,3])[...,None]).view(-1, 3)
            out2 = (c2w[:,:3].T @ (output2 - c2w[:,3])[...,None]).view(-1, 3)
            #out3 = (c2w[:,:3].T @ (output3 - c2w[:,3])[...,None]).view(-1, 3)
            rays_d = out2 - out1
            rays_o = out1 - rays_d

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)        
            #if i == 2 : break
            tt = rays_o[0].numpy()
            #pdb.set_trace()
            c2w2 = np.concatenate((R, tt[:,None]), axis=1)
            self.poses.append(c2w2)
            if self.directions is None: 
                self.directions = output1 - c2w[None,:,3]
                self.c2w = c2w2
                #principal_point = view['principal_point']
                #principal_point = [principal_point[0]*self.img_wh[0]/W, principal_point[1]*self.img_wh[1]/H]
                #self.directions = get_ray_directions_blender(self.img_wh[1], self.img_wh[0], principal_point)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        #pdb.set_trace()
        self.poses = np.stack(self.poses,0)
        #poses, self.pose_avg = center_poses(self.poses, self.blender2opencv)
        #self.render_path = get_spiral(self.poses, self.near_fars, rads_scale=0.1, N_views=30)
        self.render_path = np.eye(4)
        self.render_path[:3,:] = self.c2w
        self.render_path = np.stack([i/10*self.c2w+(1-i/10)*c2w2 for i in range(11)])
        #self.render_path = np.repeat(self.render_path[None],5,axis=0)
        #pdb.set_trace()


    def read_meta_old(self):       
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
        i_test = np.arange(0, self.poses.shape[0], self.hold_every)  # [np.argmin(dists)]
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))

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
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        if True: self.poses = self.poses[img_list]

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample