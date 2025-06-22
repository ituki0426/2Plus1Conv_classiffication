import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization,ReLU
from tensorflow.keras import Sequential
from Conv2Plus1D import *

class ResidualMain(Layer):
    def __init__(self,filters,kernel_size):
        super().__init__()
        self.seq = Sequential([
            Conv2Plus1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same'
            ),
            LayerNormalization(),
            ReLU(),
            Conv2Plus1D(
                filters=filters,
                kernel_size=kernel_size,
                padding='same'
            ),
            LayerNormalization()
        ])
    def call(self,x):
        return self.seq(x)
def main():
    input_tensor = tf.random.normal((2, 10, 64, 64, 3))  # 例: 2本の動画クリップ
    model = ResidualMain(filters=8,kernel_size=(3,3))
    output = model(input_tensor)
    assert output.shape  == (2, 10, 64, 64, 8), "出力形状が正しくありません"
    print("✅ テスト通過")

if __name__ == "__main__":
    main()