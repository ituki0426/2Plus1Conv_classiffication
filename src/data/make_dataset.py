from dotenv import load_dotenv
import os
import glob
import tensorflow as tf
from data.FrameGenerator import FrameGenerator
from pathlib import Path

def make_dataset():
    subset_path = {
        'train': Path('/mnt/g/マイドライブ/kaggle/3d_cnn_video/train'),
        'test':Path('/mnt/g/マイドライブ/kaggle/3d_cnn_video/test'),
        'val': Path('/mnt/g/マイドライブ/kaggle/3d_cnn_video/val')
    }
    n_frames = 10
    batch_size = 8
    output_signature = (
    tf.TensorSpec(shape=(10, 224, 224, 3), dtype=tf.float32),  # フレーム列
    tf.TensorSpec(shape=(), dtype=tf.int16)                        # ラベル
    )
    train_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_path['train'], n_frames, training=True),
        output_signature = output_signature
    )
    train_ds = train_ds.batch(batch_size)
    val_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_path['val'], n_frames, training=False),
        output_signature = output_signature
    )
    val_ds = val_ds.batch(batch_size)
    test_ds = tf.data.Dataset.from_generator(
        FrameGenerator(subset_path['test'], n_frames, training=False),
        output_signature = output_signature
    )
    test_ds = test_ds.batch(batch_size)
    return train_ds, val_ds, test_ds

def main():
    train_ds, val_ds, test_ds = make_dataset()

if __name__ == "__main__":
    main()