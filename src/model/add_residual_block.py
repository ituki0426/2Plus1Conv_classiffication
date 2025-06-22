import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization,add
from tensorflow.keras import Sequential,Input,Model
from model.ResidualMain import ResidualMain
from model.Project import Project

def add_residual_block(input,filters,kernel_size):
    out = ResidualMain(filters,kernel_size)(input)
    res = input
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)
    return add([res,out])

def main():
    input = Input(shape = (8,32,32,3))
    output_tensor = add_residual_block(input,filters=16, kernel_size=(3, 3, 3))
    model = Model(inputs = input,outputs = output_tensor)
    model.summary()
    x = np.random.randn(2, 8, 32, 32, 3).astype(np.float32)
    y = model.predict(x)
    print("Output shape:", y.shape)
    assert y.shape == (2, 8, 32, 32, 16), "出力形状が不正です"
    print("✅ add_residual_block テスト通過")

if __name__ == "__main__":
    main()