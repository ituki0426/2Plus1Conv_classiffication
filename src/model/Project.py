import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization
from tensorflow.keras import Sequential

class Project(Layer):
    def __init__(self,units):
        super().__init__()
        self.seq = Sequential([
            Dense(units),
            LayerNormalization()
        ])
    def call(self, x):
        return self.seq(x)

def main():
    x = tf.random.normal((4,10))
    project = Project(units = 8)
    y = project(x)
    print("Input shape : ", x.shape)
    print("Outpu shape : ",y.shape)
    assert y.shape == (4, 8), "出力形状が正しくありません"
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = project(x)
        loss = tf.reduce_mean(y)
    grads = tape.gradient(loss, project.trainable_variables)
    assert all(g is not None for g in grads), "勾配が正しく計算されていません"
    print("✅ テスト通過")

if __name__ == "__main__":
    main()