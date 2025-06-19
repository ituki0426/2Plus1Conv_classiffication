import cv2
import random
from format_frames import * 
from make_dataset import *
import numpy as np
import os
import matplotlib.pyplot as plt

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
  """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames to be created per video file.
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, 224, 224, 3).
  """
  # Read each video frame by frame
  result = []
  src = cv2.VideoCapture(str(video_path))  

  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

  need_length = 1 + (n_frames - 1) * frame_step

  if need_length > video_length:
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)

  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  ret, frame = src.read()
  
  result.append(format_frames(frame, output_size))

  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]
  return result

def main():
    path = make_dataset()['train'][0]
    frames = frames_from_video_file(path, 10)
    print(f"type of frames: {type(frames)}")
    print(f"Shape of frames: {frames.shape}")

    # 保存先フォルダの作成（存在しなければ作成）
    save_dir = "../../img"
    os.makedirs(save_dir, exist_ok=True)

    # フレームを1枚ずつ保存
    for i in range(frames.shape[0]):
        filename = os.path.join(save_dir, f"frame_{i+1:02}.png")
        # frames[i] の値が 0〜1 の float の場合でも保存可能
        plt.imsave(filename, frames[i])
        print(f"Saved: {filename}")


if __name__ == "__main__":
  main()