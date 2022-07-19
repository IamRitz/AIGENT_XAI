from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np

"""
This file contains code to trains a CNN on the CIFAR-10 dataset.
"""

class SimpleCNN:
	@staticmethod
	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1
		model.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Flatten())
		model.add(Dense(128))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model

def getModel():
    print("[INFO] loading CIFAR-10 dataset...")
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    print("[INFO] compiling model...")
    opt = Adam(lr=1e-3)
    model = SimpleCNN.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])
    
    print("[INFO] training network...")
    model.fit(trainX, trainY,
        validation_data=(testX, testY),
        batch_size=64,
        epochs=10,
        verbose=1)
    
    (loss, acc) = model.evaluate(x=testX, y=testY, verbose=0)
    print("[INFO] loss: {:.4f}, acc: {:.4f}".format(loss, acc))
    return model

model = getModel()
model.save("../Models/cifarModel.h5")