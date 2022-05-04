import os
import shutil
SOURCE_DIR = "data/meta_3d_video/flame_salmon/frames"
TARGET_DIR = "data/meta_3d_video/flame_salmon/frames_ones"

def main():
    os.makedirs(TARGET_DIR, mode = 0o777, exist_ok = True) 
    files = sorted(os.listdir(SOURCE_DIR))
    for f in files:
        shutil.copy(os.path.join(SOURCE_DIR,f,'f0001.png'),os.path.join(TARGET_DIR,f + '.png'))

if __name__ == "__main__":
    main()