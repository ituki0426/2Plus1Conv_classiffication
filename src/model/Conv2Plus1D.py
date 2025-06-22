import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization
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
                kernel_size=(1, 1, kernel_size),
                padding=padding
            )
        ])

    def call(self, x):
        return self.seq(x)