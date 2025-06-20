import random
from frames_from_vide_file import *
from pathlib import Path

class FrameGenerator:
    def __init__(self, path, n_frames, training = False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def generate_frame(self):
        # Simulate frame generation logic
        frame_data = f"Frame {self.current_frame} at {self.frame_rate} FPS"
        self.current_frame += 1
        return frame_data
    def get_files_and_class_names(self):
       video_paths = list(self.path.glob('*/*.avi'))
       classes = [p.parent.name for p in video_paths] 
       return video_paths, classes
    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))
        if self.training:
             random.shuffle(pairs)
        for path, name in pairs:
            video_frames = frames_from_video_file(path, self.n_frames) 
            label = self.class_ids_for_name[name] # Encode labels
            yield video_frames, label

def main():
    subset_path = {
        'train': Path('/mnt/g/マイドライブ/kaggle/3d_cnn_video/train'),
        'test':Path('/mnt/g/マイドライブ/kaggle/3d_cnn_video/test'),
        'val': Path('/mnt/g/マイドライブ/kaggle/3d_cnn_video/val')
        }
    n_frames = 10
    frame_generator = FrameGenerator(subset_path['train'], n_frames, training=True)
    for idx, (frames, label) in enumerate(frame_generator()):
        print(f"{idx} Frames shape: {frames.shape}, Label: {label}")

if __name__ == "__main__":
    main()