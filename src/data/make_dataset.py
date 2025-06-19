from dotenv import load_dotenv
import os
import glob

def make_dataset():
    load_dotenv()
    data_dir = "/mnt/g/マイドライブ/kaggle/3d_cnn_video"
    subset_paths  = {}
    for x in ["train", "test", "val"]:
        files = glob.glob(f"{data_dir}/{x}/**/*.avi", recursive=True)
        subset_paths [x] = files
    return subset_paths