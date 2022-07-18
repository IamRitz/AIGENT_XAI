from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from prepareData import getTrainData, getValData

def getData(w, h):
    X_train, y_train = getTrainData(w, h)
    X_test, y_test = getValData(w, h)
    print(X_train.shape)
    print(y_train.shape)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, X_test, y_test

def train():
    w = 128
    h = w
    X_train, y_train, X_test, y_test = getData(w, h)
    # # return
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)
    
    # num_classes = y_test.shape[1]
    # print(num_classes)
    # model = Sequential()

    # model.add(Dense(50, input_dim = w * h * 3, activation= 'relu'))
    # model.add(Dense(50, activation = 'relu'))
    # model.add(Dense(50, activation = 'relu'))
    # model.add(Dense(50, activation = 'relu'))
    # model.add(Dense(10))
    # model.compile(loss = 'MeanSquaredError', optimizer = 'adam', metrics = ['accuracy'])
    # model.fit(X_train, y_train, epochs= 50, batch_size = 64, validation_data=(X_test, y_test))
    # model.save("imagenette_3.h5")
    # return

# train()