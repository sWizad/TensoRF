
import numpy as np 
import os 
from skimage import io
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm
import json

SOURCE_DIR = 'data/meta_3d_video/flame_salmon'
TARGET_DIR = 'data/nerfies/flame_salmon_30'

NUM_CAMS = 19
NUM_FRAMES = 1200
NUM_ACTUAL_FRAME = 30
HEIGHT = 768
WIDTH = 1024
IMG_RATIO = 2.640625

def main():
    camera_dir = os.path.join(TARGET_DIR, 'camera')
    image_dir = os.path.join(TARGET_DIR, 'rgb/1x')
    os.makedirs(TARGET_DIR, exist_ok=True, mode=0o777)
    os.makedirs(camera_dir, exist_ok=True, mode=0o777)
    os.makedirs(image_dir, exist_ok=True, mode=0o777)
    cam_info = [(cam, frame) for cam in range(NUM_CAMS) for frame in range(NUM_ACTUAL_FRAME)]
    # extract first 300 images 
    with Pool(16) as p:
        print('saving images...')
        with tqdm(total=len(cam_info)) as pbar:
            for  _ in p.imap_unordered(extract_image, cam_info):
                pbar.update()
    create_dataset_json()
    create_metadata_json()
    create_camera()

def create_camera():
    poses_bounds = np.load(os.path.join(SOURCE_DIR, 'poses_bounds.npy'))
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]
    for cam in tqdm(range(NUM_CAMS)):
        hwf = poses[cam,:,4]
        data = {
            "focal_length": hwf[2] * WIDTH / hwf[1],
            "image_size": [
                WIDTH,
                HEIGHT
            ],
            "orientation": poses[cam,:3,:3].tolist(),
            "pixel_aspect_ratio": 1.0,
            "position": poses[cam,:3,3].tolist(),
            "principal_point": [
                WIDTH / 2.0,
                HEIGHT / 2.0
            ],
            "radial_distortion": [
                0.0,
                0.0,
                0.0
            ],
            "skew": 0.0,
            "tangential": [
                0.0,
                0.0
            ]
        }
        for frame in range(NUM_ACTUAL_FRAME):
            with open(os.path.join(TARGET_DIR,'camera', 'cam{:02d}_{:04d}.json'.format(cam,frame)),'w') as f:
                json.dump(data, f, indent=4)


def create_metadata_json():
    metadata = {}
    for cam in range(NUM_CAMS):
        for frame in range(NUM_ACTUAL_FRAME):
            metadata['cam{:02d}_{:04d}'.format(cam, frame)] = {
                "appearance_id": cam*NUM_ACTUAL_FRAME+frame,
                "camera_id": cam,
                "warp_id": frame
            }
    with open(os.path.join(TARGET_DIR, 'metadata.json'),'w') as f:
        json.dump(metadata, f, indent=4)

def create_dataset_json():
    cam_info = [(cam, frame) for cam in range(NUM_CAMS) for frame in range(NUM_ACTUAL_FRAME)]
    dataset = {
        "count": NUM_CAMS*NUM_ACTUAL_FRAME,
        "num_exemplars": (NUM_CAMS-1)*NUM_ACTUAL_FRAME,
        "ids": ['cam{:02d}_{:04d}'.format(cam, frame) for cam in range(NUM_CAMS) for frame in range(NUM_ACTUAL_FRAME)],
        "train_ids": ['cam{:02d}_{:04d}'.format(cam, frame) for cam in range(1,NUM_CAMS) for frame in range(NUM_ACTUAL_FRAME)],
        "val_ids": ['cam{:02d}_{:04d}'.format(0, frame)  for frame in range(NUM_ACTUAL_FRAME)]
    }
    with open(os.path.join(TARGET_DIR, 'dataset.json'),'w') as f:
        json.dump(dataset, f, indent=4)

def extract_image(info):
    cam_id, frame_id = info
    offset_file = HEIGHT * WIDTH
    offset_frame = cam_id * NUM_FRAMES + frame_id
    memmap_file = os.path.join(SOURCE_DIR, 'rgb_2.640625.memmap')
    rgbs =  np.memmap(memmap_file, dtype=np.ubyte, mode='r', offset=0, shape=(NUM_CAMS*NUM_FRAMES*HEIGHT*WIDTH,3), order='C')
    img = rgbs[offset_frame * offset_file: (offset_frame+1) * offset_file].reshape((HEIGHT, WIDTH, 3))
    io.imsave(os.path.join(TARGET_DIR, 'rgb/1x','cam{:02d}_{:04d}.png'.format(cam_id, frame_id)), img)
    
if __name__ == "__main__":
    main()