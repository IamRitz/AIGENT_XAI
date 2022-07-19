from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense

"""
This file contains code to trains a deep neural network on the CIFAR-10 dataset.
"""

def getData():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print(X_train.shape)
    print(y_train.shape)

    X_train = X_train.reshape((X_train.shape[0], 32*32*3)).astype('float32')
    # X_test.shape[0], X_train[1]*X_train[2]*X_train[3]
    X_test = X_test.reshape((X_test.shape[0], 32*32*3)).astype('float32')

    X_train = X_train / 255
    X_test = X_test / 255

    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, X_test, y_test

def train():
    X_train, y_train, X_test, y_test = getData()
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    num_classes = y_test.shape[1]
    print(num_classes)
    model = Sequential()

    model.add(Dense(50, input_dim = 32 * 32 * 3, activation= 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))
    model.add(Dense(50, activation = 'relu'))


    model.add(Dense(10))
    model.compile(loss = 'MeanSquaredError', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train, y_train, epochs= 50, batch_size = 100)
    model.save("cifar.h5")

train()