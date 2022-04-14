import os 
import shutil 
from multiprocessing import Pool
import subprocess

DATASET_PATH = 'data/meta_3d_video/flame_salmon'

def extract_frames(video_file):
    print(video_file)
    frame_dir = os.path.join(DATASET_PATH, 'frames')
    cam_id = int(video_file.split('.')[0][3:])
    cam_dir = os.path.join(frame_dir,'c{:02d}'.format(cam_id))
    os.makedirs(cam_dir,exist_ok=True,mode=0o777)
    input_path = os.path.join(DATASET_PATH, video_file)
    output_path = os.path.join(cam_dir,'f%04d.png')
    subprocess.check_output(
        "ffmpeg -i {} {}".format(str(input_path), str(output_path)),
        shell=True, 
        stderr=subprocess.STDOUT
    )

def main():
    frame_dir = os.path.join(DATASET_PATH, 'frames')
    os.makedirs(frame_dir,exist_ok=True,mode=0o777)
    video_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith('.mp4')])
    with Pool(8) as p:
        p.map(extract_frames,video_files)

if __name__ == "__main__":
    main()