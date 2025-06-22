import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization
from tensorflow.keras import Sequential
from ResidualMain import *
def add_residual_block(input,filters,kernel_size):
    out = ResidualMain(filters,kernel_size)(input)