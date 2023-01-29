import random
import cv2
import einops
import numpy as np
import tensorflow as tf
import keras
from keras import layers


height = 224
width = 224

class Conv2Plus1D(layers.Layer):
    def __init__(self, filters, kernel_size, padding, stride):
        super(Conv2Plus1D, self).__init__()
        self.block = keras.Sequential([
            # Spatial Conv
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding,
                         strides=(1, stride, stride)),
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding,
                         strides=(1, 1, 1))
        ])
    def call(self, x):
        return self.block(x)


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size, padding, stride):
        super(ResidualBlock, self).__init__()
        self.ResBlock = keras.Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding=padding, stride=stride),
            layers.ReLU(),
            layers.BatchNormalization(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding=padding, stride=1),
            layers.ReLU()
        ])
        
    def call(self, x):
        return self.ResBlock(x)



class Stage(layers.Layer):
    def __init__(self, filters, kernel_size, padding, stride):
        super(Stage, self).__init__()
        self.stride = stride
        self.resblock = ResidualBlock(filters=filters, kernel_size=kernel_size,
                                     padding=padding, stride=stride)
        if stride > 1:
            self.projector = layers.Conv3D(filters = filters, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                          padding=padding, strides=(1, stride, stride), use_bias = False)
            
        
        
    def call(self, x):
        if self.stride > 1:
            return self.resblock(x) + self.projector(x)
        else:
            return self.resblock(x)


def build_model():
    model_input = keras.Input(shape=(10, 224, 224, 3))
    x = ResidualBlock(filters=8, kernel_size=(3, 7, 7), padding="same", stride=1)(model_input) # N x 10 x 224 x 224 x 8
    x = layers.AveragePooling3D(pool_size=(1, 2, 2), strides=2)(x) # N x 10 x 112 x 112 x 8

    x = Stage(filters=16, kernel_size=(3, 3, 3), padding="same", stride=1)(x)   # N x 10 x 112 x 112 x 16
    x = Stage(filters = 32, kernel_size = (3, 3, 3), padding="same", stride=2)(x)      # N x 10 x 56 x 56 x 32

    x = Stage(filters=64, kernel_size=(3, 3, 3), padding="same", stride=1)(x)      # N x 10 x 56 x 56 x 64
    x = Stage(filters = 32, kernel_size=(3, 3, 3), padding="same", stride=2)(x)   # N x 10 x 28 x 28 x 32

    x = Stage(filters = 16, kernel_size = (3, 3, 3), padding = "same", stride=2)(x)  # N x 10 x 14 x 14 x 16
    x = layers.GlobalAveragePooling3D()(x)         # N x 16
    x = layers.Dense(10)(x)
    model = keras.Model(inputs = model_input, outputs = x)
    model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
             metrics = ["accuracy"])
    return model
