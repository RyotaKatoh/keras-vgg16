from keras.models import Sequential
from keras.layers import Reshape
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, adam
from keras.utils import np_utils
from PIL import Image
import numpy as np
import argparse

import keras_vgg16


def generator():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=224 * 224 * 3))
    model.add(BatchNormalization(mode=2))
    model.add(Activation('relu'))
    model.add(Reshape((3, 224, 224)))

    return model


def generator_on_vgg16(generator, vgg16):
    model = Sequential()
    model.add(generator)

    vgg16.trainable = False
    model.add(vgg16)

    return model


def train(target_class):

    g = generator()
    vgg = keras_vgg16.VGG_16('vgg16_weights.h5')

    gvgg = generator_on_vgg16(g, vgg)
    noise = np.random.uniform(-1, 1, 100)

    gvgg.compile(loss='categorical_crossentropy', optimizer=adam)

    # inputs = np.random.uni

    inputs = np.array([noise])
    outputs = np_utils.to_categorical([target_class], 1000)

    gvgg.fit(inputs, outputs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", type=int)
    args = parser.parse_args()
    return args
