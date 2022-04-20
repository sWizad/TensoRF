import torch
from torch.utils.data import Dataset
import glob, json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation, Slerp
import pdb
from tqdm.auto import tqdm
from skimage import io, img_as_ubyte

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
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)



class MetaDynamicDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, ndc_ray=False, max_t=1, **kwargs):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.ndc_ray = ndc_ray
        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()
        self.max_t = max_t
        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        #         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        self.near_far = [0, 2.0]
        #self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        #self.scene_bbox = torch.tensor([[-15.0, -15.0, -15.0], [15.0, 15.0, 15.0]])
        self.scene_bbox = torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

        #sphere parameter
        self.origin = np.array([[0],[0],[0]])
        self.sph_box = [-1, 1]
        self.sph_frontback = [1, 6]
        

    def read_meta(self):

        #with open('data/deepview_video/01_Welder/models.json') as f:
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
        N_views, N_rots = 30, 2
        tt = self.poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)

        self.render_path = get_spiral(self.poses, self.near_fars, N_views=N_views)

        # distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        # val_idx = np.argmin(distances_from_center)  # choose val image as the closest to
        # center image

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
        all_mask = []
        missing_view = 0
        DIFF_THREST = 0.05
        DIFF_NAME = "difference_{}".format(DIFF_THREST)
        DIFF_DIR = os.path.join(self.root_dir, DIFF_NAME)
        os.makedirs(DIFF_DIR, exist_ok=True, mode=0o777)
        for i, view in enumerate(tqdm(img_list)):
            if not os.path.exists(os.path.join(self.root_dir,f'frames/c{view:02d}')): 
                missing_view += 1
                continue #skip missing input view
            #c2w = torch.FloatTensor(self.poses[i])
            c2w = torch.FloatTensor(self.poses[view - missing_view])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            DIFF_DIR_SUB = os.path.join(DIFF_DIR, 'cam{:02d}'.format(view))
            os.makedirs(DIFF_DIR_SUB, exist_ok=True, mode=0o777)
            median_image = Image.open(os.path.join(self.root_dir,'median_{}'.format(self.downsample),'cam{:02d}.png'.format(view))).convert('RGB')
            median_image = self.transform(median_image) #(3, h, w) [0.0-1.0]
            median_image = median_image.view(3, -1).permute(1, 0) #(h*w,3)
            for t in range(self.max_t):
                image_path = os.path.join(self.root_dir,f'frames/c{view:02d}/f{t+1:04d}.png') #change to view instead, sometimes i is not consequtive [i_test]
                img = Image.open(image_path).convert('RGB')
                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                if t == 0:
                    #visual_img = img
                    mask = img[..., 0] >= -9999999
                else: 
                    #visual_img = img.clone()
                    distance = torch.abs(img - median_image)
                    mask = torch.logical_or(distance[...,0] > DIFF_THREST, torch.logical_or(distance[...,1] > DIFF_THREST, distance[...,2] > DIFF_THREST))
                    #not_mask = torch.logical_not(mask)
                    #visual_img[not_mask] = 0.0
                    #visual_img[not_mask,1] = 1.0
                #visual_img = visual_img.permute(1,0).view(3,H,W).permute(1,2,0).numpy()
                #io.imsave(os.path.join(DIFF_DIR_SUB,'f{:04d}.png'.format(t+1)),img_as_ubyte(visual_img))
                all_mask += [mask]
                self.all_rgbs += [img]
                self.all_rays += [torch.cat([rays_o, rays_d, t*torch.ones_like(rays_o[...,0:1])], 1)]  # (h*w, 7) 
                    
        if not self.is_stack:
            # if not is stack, we filter out unused ray on training
            for i in range(len(all_mask)):
                self.all_rgbs[i] = self.all_rgbs[i][all_mask[i]]
                self.all_rays[i] = self.all_rays[i][all_mask[i]]
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
    
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample