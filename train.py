import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Reshape, Conv2D, MaxPooling2D, LSTM
from keras.layers.normalization import BatchNormalization


def simple_rnn(input_shape=(160, 120, 1)):

    model = Sequential()

    model.add(Conv2D(64, input_shape=input_shape, kernel_size=(
        5, 5), strides=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Reshape((39, 29*64)))

    model.add(LSTM(64))
    model.add(Dense(1))
    model.summary()

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model


class Dataset(object):
    def __init__(self, path):
        data = np.load(path)
        shape = data['arr_0'].shape

        self.x = data['arr_0'].reshape((shape[0], shape[2], shape[1], 1))
        self.y = data['arr_1']


if __name__ == "__main__":
    WIDTH = 160
    HEIGHT = 120

    d = Dataset("data/data.npz")

    model = simple_rnn()

    model.fit(d.x, d.y, epochs=100, validation_split=0.2, batch_size=128)
