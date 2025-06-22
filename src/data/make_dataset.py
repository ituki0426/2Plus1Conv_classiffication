from dotenv import load_dotenv
import os
import glob
import tensorflow as tf
from data.FrameGenerator import FrameGenerator
from pathlib import Path

def make_dataset():
    subset_path = {
        'train': Path(os.getenv('TRAIN_PATH')),
        'test':Path(os.getenv('TEST_PATH')),
        'val': Path(os.getenv('VAL_PATH'))
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
    print("train_ds:")
    for batch_idx, (frames, labels) in enumerate(train_ds.take(1)):  # 1バッチだけ確認
        print(f"Batch {batch_idx}:")
        print(f"  Frames shape: {frames.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels.numpy()}")
    

if __name__ == "__main__":
    main()