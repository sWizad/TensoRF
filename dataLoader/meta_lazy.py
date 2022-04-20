# this is MetaVideo dataset that implementa lazy load
import torch
import torchvision
from torch.utils.data import Dataset
import glob, json
import numpy as np
import os
import socket, shutil, subprocess
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation, Slerp
import pdb
from tqdm.auto import tqdm
from multiprocessing import Pool
from functools import partial
from skimage import img_as_float, img_as_ubyte, io
import time


class MetaVideoLazyDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, ndc_ray=False, max_t=1, **kwargs):
        print('MetaVideoLazyDataset')
        #TODO: need to config proper variable
        # initialize variable 
        self.root_dir = datadir
        self.downsample = downsample

        self.prepare_memmap()

        self.ndc_ray = ndc_ray
        self.split = split
        self.hold_every = hold_every
        self.is_stack = is_stack
        self.define_transforms()
        self.max_t = max_t
        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False
        self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
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

    def prepare_memmap(self):
        # copy or create memmap if not existed.
        # we use memmap instead of the original dataset for faster file seeking
        if not os.path.exists(os.path.join(self.root_dir,'rgb_{}.memmap'.format(self.downsample))) or not os.path.exists(os.path.join(self.root_dir,'video_meta_{}.json'.format(self.downsample))):
            print("#TODO: NEED TO BE REMOVE")
            print("EXIT: SAFEGUARD BEFORE MEMMAP")
            exit()
            self.create_memmap()
        if not os.path.exists(os.path.join(self.root_dir,'median_{}.npy'.format(self.downsample))):
            self.create_median()
        print("ANOTHER SAFEGUARD")
        exit()
        # copy memmap to local instead of using Network attach storage
        hostname = socket.gethostname()
        cache_path = None
        if hostname.lower().startswith("vision"): #vision cluster
            cache_path = "/data/cache/meta_video_dataset/"
        if hostname.lower().startswith("ist-gpu"): #ist cluster
            cache_path = "/scratch/vision/cache/meta_video_dataset/"
        if cache_path is not None:
            os.makedirs(cache_path, mode=0o777, exist_ok=True)
            memmap_datapath_abs = os.path.abspath(memmap_datapath).lower().replace("/","_")
            dst_path = os.path.join(cache_path, memmap_datapath_abs)
            shutil.copy2(memmap_datapath, dst_path)
            memmap_datapath = dst_path

    def create_median(self):
        print('creating median image...')
        start_time = time.time()
        with open(os.path.join(self.root_dir,'video_meta_{}.json'.format(self.downsample)), 'r') as f:
            meta = json.load(f)
        num_threads = 10 # please manually set the number of threads on the machine that has Out-of-memory problem
        fn = partial(median_process, self.root_dir, self.downsample, len(meta['video']))
        with Pool(num_threads) as p: 
            images = p.map(fn, enumerate(meta['video']))
            np.save(os.path.join('median_{}.npy'.format(self.downsample)), np.concatenate([img[None] for img in images],axis=0))
        print('======= break XD =====')
        exit()
 
    def create_memmap(self):
        print('creating memmap...')
        start_time = time.time()
        # check pre-require software
        if shutil.which('ffmpeg') is None:
            raise Exception("FFmpeg is require to pre-process video.")
        if shutil.which('ffprobe') is None:
            raise Exception("FFprobe is missing. we recommend to install FFmpeg from APT which include FFprobe")
        # list video first
        video_files = sorted([f for f in os.listdir(self.root_dir) if f.endswith(".mp4")])
        num_inputview = len(video_files)
        fn = partial(memmap_process, self.root_dir, self.downsample, num_inputview)
        num_threads = 8 # please manually set the number of threads on the machine that has Out-of-memory problem
        raw_meta = get_video_metadata(os.path.join(self.root_dir, video_files[0]))
        num_frames = int(raw_meta['streams'][0]['nb_read_packets'])
        height =  int(raw_meta['streams'][0]['height'] / self.downsample)
        width =  int(raw_meta['streams'][0]['width'] / self.downsample)
        memfile_path = os.path.join(self.root_dir,'rgb_{}.memmap'.format(self.downsample))     
        create_memfile(memfile_path, (num_inputview*num_frames*height*width,3))
        with Pool(num_threads) as p:
            a = list(enumerate(video_files[:2]))          
            video_metadata = p.map(fn, list(enumerate(video_files)))
            with open(os.path.join(self.root_dir, "video_meta_{}.json".format(self.downsample)), 'w') as f:
                json.dump({
                    "video": video_metadata
                },f,indent=4)
            print("created memmap in " ,time.time() - start_time, "seconds...")

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
        img_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))) - set(i_test))
        """
        self.all_rays = []
        self.all_rgbs = []

        print("Loading cameras...")
        for i, view in enumerate(tqdm(img_list)):
            c2w = torch.FloatTensor(self.poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
            # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
            for t in range(self.max_t):
                image_path = os.path.join(self.root_dir,f'frames/c{view:02d}/f{t+1:04d}.png') #change to view instead, sometimes i is not consequtive [i_test]

                img = Image.open(image_path).convert('RGB')
                if self.downsample != 1.0:
                    img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]
                self.all_rays += [torch.cat([rays_o, rays_d, t*torch.ones_like(rays_o[...,0:1])], 1)]  # (h*w, 7) 

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)
        """

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        # @return total number of pixel
        # we can fine all pixel by count height*width*num_of_frame*num_video
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample

def get_video_metadata(file):
    #cmd = 'ffprobe -v error -select_streams v -i {} -show_entries stream=width,height  -of json'.format(file)
    cmd = 'ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=width,height,nb_read_packets -of json {}'.format(file)
    output = subprocess.check_output(
        cmd,
        shell=True, # Let this run in the shell
        stderr=subprocess.STDOUT
    )
    return json.loads(output)


def median_process(directory, downsample, num_cams, frame_data):
    name = frame_data[1]['name']
    print(name)
    cam_id = frame_data[0]
    height = frame_data[1]['height']
    width = frame_data[1]['width']
    num_frames = frame_data[1]['num_frames']
    memfile_path = os.path.join(directory,'rgb_{}.memmap'.format(downsample))
    fp = np.memmap(memfile_path, dtype=np.ubyte, mode='r', offset=0, shape=(num_cams*num_frames*height*width,3), order='C') #flat line of
    start_offset = cam_id * num_frames*height*width
    end_offset = (cam_id+1) * num_frames*height*width
    pixels = fp[start_offset:end_offset].reshape((num_frames,height*width,3))
    pixels = np.median(pixels,axis=0)
    median_dir = os.path.join(directory, 'median_{}'.format(downsample))
    os.makedirs(median_dir, exist_ok=True)
    print(os.path.join(median_dir,'{}.png'.format(name)))
    io.imsave(os.path.join(directory,'median','{}.png'.format(name)), pixels.reshape(height,width,3)) # write median image for visualization
    return pixels




def memmap_process(directory, downsample, num_cams, input_file):
    file_id = input_file[0]
    input_file = input_file[1]
    video_path = os.path.join(directory, input_file)
    video_meta = get_video_metadata(video_path)
    height = int(video_meta['streams'][0]['height'] / downsample)
    width = int(video_meta['streams'][0]['width'] / downsample)
    input_name = ".".join(str(input_file).split('.')[:-1])
    frame_dir = os.path.join(directory, "{}_{}".format(input_name, downsample))
    print(frame_dir)
    os.makedirs(frame_dir, mode=0o777, exist_ok=True)
    subprocess.check_output(
        "ffmpeg -i {} -vf scale={}:{} {}".format(video_path, width, height, str(os.path.join(frame_dir,'frame_%04d.png'))),
        shell=True, 
        stderr=subprocess.STDOUT
    )
    img_files = sorted([os.path.join(frame_dir,f) for f in os.listdir(frame_dir) if f.endswith('.png')])
    num_frames = len(img_files)
    #fp = np.memmap(os.path.join(directory,'{}_res{}.memmap'.format(input_name,downsample)), dtype=np.ubyte, mode='w+', offset=0, shape=(num_frames*height*width,3), order='C') #flat line of
    
    memfile_path = os.path.join(directory,'rgb_{}.memmap'.format(downsample))
    fp = np.memmap(memfile_path, dtype=np.ubyte, mode='r+', offset=0, shape=(num_cams*num_frames*height*width,3), order='C') #flat line of
    #offset = 0
    offset = file_id*num_frames*height*width
    for i,f in enumerate(img_files):
        img = io.imread(f)
        img = img[...,:3] # only RGB can continue
        img = img_as_ubyte(img)
        img = img.reshape((-1,3))
        fp[offset:offset+img.shape[0]] = img
        offset += img.shape[0]
        os.remove(f)
    fp.flush() #write content to file
    shutil.rmtree(frame_dir)
    return {
        'id': int(input_name[3:]),
        'name': input_name,
        'num_frames': num_frames,
        'height': height,
        'width': width
    }

def create_memfile(f,s):
    if not os.path.exists(f):
        fp = np.memmap(f, dtype=np.ubyte, mode='w+', offset=0, shape=s, order='C') #flat line of
            