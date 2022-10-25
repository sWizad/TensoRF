# DreamFusion Dataset

import torch,cv2
from torch.utils.data import Dataset
import json, random
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T
import numpy as np
from packaging import version as pver

from transformers import CLIPTextModel, CLIPTokenizer, logging
logging.set_verbosity_error()


class DreamFusionDataset(Dataset):
    def __init__(self, datadir="a high quality photo of a pineapple", split='train', iter=20000, **kwargs):
        super().__init__()
        self.split = split
        self.ref_text = datadir
        self.iter = iter
        self.near_far = [0.1, 2.0]
        self.H = 64
        self.W = 64
        self.cx = self.H / 2
        self.cy = self.W / 2
        self.radius_range = [1.0, 1.5]
        self.scene_bbox = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        self.fovy_range = [40, 70]
        self.prepare_word_embed()
    

    def prepare_word_embed(self):
        # load text-related
        print("DreamFusionDataset: Preparing text embedding...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda:0")

        self.text_z = []
        for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
            text = f"{self.ref_text}, {d} view"
            text_z = self.get_text_embeds([text])
            self.text_z.append(text_z.cpu())

        # clear up memory
        del self.tokenizer
        del self.text_encoder
        print("DreamFusionDataset: text embedding DONE")

    def get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.text_encoder.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer([''] * len(prompt), padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.text_encoder.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    
    def __len__(self):
        return self.iter

    def __getitem__(self, idx):
        if self.split == "train":
            poses, dirs = rand_poses(1, "cpu", radius_range=self.radius_range, return_dirs=True, angle_overhead=30, angle_front=60, jitter=False)
            # random focal
            fov = random.random() * (self.fovy_range[1] - self.fovy_range[0]) + self.fovy_range[0]
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])

        else:
            # circle pose
            phi = (idx / self.iter) * 360
            poses, dirs = circle_poses("cpu", radius=self.radius_range[1] * 1.2, theta=60, phi=phi, return_dirs=True, angle_overhead=30, angle_front=60)

            # fixed focal
            fov = (self.fovy_range[1] + self.fovy_range[0]) / 2
            focal = self.H / (2 * np.tan(np.deg2rad(fov) / 2))
            intrinsics = np.array([focal, focal, self.cx, self.cy])
        
        rays = get_rays(poses, intrinsics, self.H, self.W, -1)
        rays = torch.cat([rays['rays_o'], rays['rays_d']],dim=-1).view(-1,6) 
        
        sample = {
            'rays': rays,
            'text_embed': self.text_z[dirs[0]]
        }
        
        return sample

"""
Utiliy function
Taken from stable-dreamfusion
@see https://github.com/ashawkey/stable-dreamfusion/blob/5d9dde6013eb43bcfe5ebd0e4d11548eb2c4f597/nerf/utils.py
"""    

def rand_poses(size, device, radius_range=[1, 1.5], theta_range=[0, 100], phi_range=[0, 360], return_dirs=False, angle_overhead=30, angle_front=60, jitter=False):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, pi]
        phi_range: [min, max], should be in [0, 2 * pi]
    Return:
        poses: [size, 4, 4]
    '''

    theta_range = np.deg2rad(theta_range)
    phi_range = np.deg2rad(phi_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    
    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    targets = 0

    # jitters
    if jitter:
        centers = centers + (torch.rand_like(centers) * 0.2 - 0.1)
        targets = targets + torch.randn_like(centers) * 0.2

    # lookat
    forward_vector = safe_normalize(targets - centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    
    if jitter:
        up_noise = torch.randn_like(up_vector) * 0.02
    else:
        up_noise = 0

    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1) + up_noise)

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs

def circle_poses(device, radius=1.25, theta=60, phi=0, return_dirs=False, angle_overhead=30, angle_front=60):

    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    thetas = torch.FloatTensor([theta]).to(device)
    phis = torch.FloatTensor([phi]).to(device)

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - safe_normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    if return_dirs:
        dirs = get_view_direction(thetas, phis, angle_overhead, angle_front)
    else:
        dirs = None
    
    return poses, dirs

def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = safe_normalize(directions)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis
    res[(phis < front)] = 0
    res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi + front))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')