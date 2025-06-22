import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization,Conv3D
from tensorflow.keras import Sequential


class Conv2Plus1D(Layer):
    def __init__(self, filters, kernel_size,padding ):
        super().__init__()
        self.seq = Sequential([
            Conv3D(
                filters=filters, 
                kernel_size=(1, kernel_size[1], kernel_size[2]),
                padding=padding
            ),
            Conv3D(
                filters=filters, 
                kernel_size=(kernel_size[0], 1, 1),
                padding=padding
            )
        ])

    def call(self, x):
        return self.seq(x)

def main():
    x = tf.random.normal((2,8,32,32,3))
    model = Conv2Plus1D(
        filters=16,
        kernel_size=(3,3,3),
        padding='same'
    )
    y = model(x)
    print("Input shape",x.shape)
    print("Output shape",y.shape)
    assert y.shape == (2,8,32,32,16), "出力形状が不正です"
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads), "勾配が正しく計算されていません"
    print("✅ Conv2Plus1D テスト通過")

if __name__ == "__main__":
    main()