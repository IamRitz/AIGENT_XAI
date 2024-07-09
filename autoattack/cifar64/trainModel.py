import tensorflow as tf
import os
from PIL import Image
import random
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
		model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same",
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



train_img_path = "./train"
test_img_path = "./test"

def getData():
    origin_train_pair_list = []
    origin_test_pair_list = []

    # for i in range(x_train.shape[0]):
    #     pair = []
    #     pair.append(x[i])
    #     pair.append(y[i])
    #     origin_train_pair_list.append(pair)

    folder_name_train = os.listdir(train_img_path)
    folder_name_test = os.listdir(test_img_path)

    NUM_CLASS = len(folder_name_train)

    folder_path_train = []
    folder_path_test = []
    for i in range(NUM_CLASS):
        class_name = folder_name_train[i]
        path = train_img_path 
        path = path + '/' + class_name
        folder_path_train.append(path)

    for i in range(NUM_CLASS):
        class_name = folder_name_test[i]
        path = test_img_path
        path = path + '/' + class_name
        folder_path_test.append(path)
        
    image_label_pair_list_train = []

    #This will decide how many extra images you want to add to the dataset
    NUM_EXTRA_IMAGES = 5000

    #The initial label is 0
    label = 0

    for i in range(NUM_CLASS):
        folder_name = folder_path_train[i]
        image_name_list = os.listdir(folder_name)
        for ids in range(NUM_EXTRA_IMAGES):
            pair = []
            file_path = folder_name + '/' + image_name_list[ids]
            img = np.array(Image.open(file_path))
            pair.append(img)
            lab_list = []
            lab_list.append(label)
            pair.append(lab_list)
            image_label_pair_list_train.append(pair)
        label += 1

    label = 0
    image_label_pair_list_test = []
    for i in range(NUM_CLASS):
        folder_name = folder_path_test[i]
        image_name_list = os.listdir(folder_name)
        for ids in range(1000):
            pair = []
            file_path = folder_name + '/' + image_name_list[ids]
            img = np.array(Image.open(file_path))
            pair.append(img)
            lab_list = []
            lab_list.append(label)
            pair.append(lab_list)
            image_label_pair_list_test.append(pair)
        label += 1
        
    # We shuffle it again
    random.shuffle(image_label_pair_list_train)
    random.shuffle(image_label_pair_list_test)

    origin_train_pair_list.extend(image_label_pair_list_train)
    origin_test_pair_list.extend(image_label_pair_list_test)
    random.shuffle(origin_train_pair_list)
    random.shuffle(origin_test_pair_list)

    image_list_train = []
    label_list_train = []
    image_list_test = []
    label_list_test = []

    for i in range(len(origin_train_pair_list)):
        pair_train = origin_train_pair_list[i]
        image_list_train.append(pair_train[0])
        label_list_train.append(pair_train[1])

    for i in range(len(origin_test_pair_list)):
        pair_test = origin_test_pair_list[i]
        image_list_test.append(pair_test[0])
        label_list_test.append(pair_test[1])


    x_train = np.array(image_list_train)
    y_train = np.array(label_list_train)

    x_test = np.array(image_list_test)
    y_test = np.array(label_list_test)


    # x_train = x_train.reshape(x_train.shape[0], -1)
    # x_test = x_test.reshape(x_test.shape[0], -1)

    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def getModel():
    print("[INFO] loading CIFAR-10 dataset...")
    trainX, trainY, testX, testY = getData()
    trainX = trainX / 255.0
    testX = testX / 255.0
    
    trainX = np.expand_dims(trainX, axis=-1)
    testX = np.expand_dims(testX, axis=-1)
    
    trainY = to_categorical(trainY, 10)
    testY = to_categorical(testY, 10)
    print("[INFO] compiling model...")
    opt = Adam(lr=1e-3)
    model = SimpleCNN.build(width=64, height=64, depth=3, classes=10)
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
