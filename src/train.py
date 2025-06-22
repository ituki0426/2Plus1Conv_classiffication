import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer, Dense,LayerNormalization,ReLU
from tensorflow.keras import Sequential,Model
from data.make_dataset import make_dataset
from model.add_residual_block import add_residual_block
from model.Conv2Plus1D import Conv2Plus1D
from model.ResizeVideo import ResizeVideo

def main():
    HEIGHT = 224
    WIDTH = 224
    input_shape = (None, 10, HEIGHT, WIDTH, 3)
    input = layers.Input(shape=(input_shape[1:]))
    x = input
    x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)
    # Block 1
    x = add_residual_block(x, 16, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)
    # Block 2
    x = add_residual_block(x, 32, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)

    # Block 3
    x = add_residual_block(x, 64, (3, 3, 3))
    x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)

    # Block 4
    x = add_residual_block(x, 128, (3, 3, 3))

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)

    model = Model(input, x)
    train_ds, val_ds, test_ds = make_dataset()
    frames,label = next(iter(train_ds))
    model.build(frames)
    model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001), 
              metrics = ['accuracy'])
    history = model.fit(x = train_ds,
                    epochs = 50, 
                    validation_data = val_ds)
    
if __name__ == "__main__":
    main()