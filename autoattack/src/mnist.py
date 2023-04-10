from argparse import ArgumentParser
from time import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
from autoattack import AutoAttack, utils_tf
import fid

class mnist_loader:
    def __init__(self):

        self.n_class = 10
        self.dim_x   = 28
        self.dim_y   = 28
        self.dim_z   = 1
        self.img_min = 0.0
        self.img_max = 1.0
        self.epsilon = 0.3

    def download(self):
        (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

        trainX = trainX.astype(np.float32)
        testX  = testX.astype(np.float32)

        # ont-hot
        trainY = tf.keras.utils.to_categorical(trainY, self.n_class)
        testY  = tf.keras.utils.to_categorical(testY , self.n_class)

        # get validation sets
        training_size = 55000
        validX = trainX[training_size:,:]
        validY = trainY[training_size:,:]

        trainX = trainX[:training_size,:]
        trainY = trainY[:training_size,:]

        # expand dimesion
        trainX = np.expand_dims(trainX, axis=3)
        validX = np.expand_dims(validX, axis=3)
        testX  = np.expand_dims(testX , axis=3)

        return trainX, trainY, validX, validY, testX, testY

    def get_raw_data(self):
        return self.download()

    def get_normalized_data(self):
        trainX, trainY, validX, validY, testX, testY = self.get_raw_data()
        trainX = trainX / 255.0 * (self.img_max - self.img_min) + self.img_min
        validX = validX / 255.0 * (self.img_max - self.img_min) + self.img_min
        testX  = testX  / 255.0 * (self.img_max - self.img_min) + self.img_min
        trainY = trainY
        validY = validY
        testY  = testY
        return trainX, trainY, validX, validY, testX, testY

def mnist_model():
    # declare variables
    model_layers = [ tf.keras.layers.Input(shape=(28,28,1), name="model/input"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c1"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c2"),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="clf/p1"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c3"),
                        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu", kernel_initializer='he_normal', name="clf/c4"),
                        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name="clf/p2"),
                        tf.keras.layers.Flatten(name="clf/f1"),
                        tf.keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal', name="clf/d1"),
                        tf.keras.layers.Dense(10 , activation=None  , kernel_initializer='he_normal', name="clf/d2"),
                        tf.keras.layers.Activation('softmax', name="clf_output")
                    ]

    # clf_model
    clf_model = tf.keras.Sequential()
    for ii in model_layers:
        clf_model.add(ii)

    clf_model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=['accuracy'])
    clf_model.summary()

    return clf_model

def getAdversarial(threshold):
    tf.compat.v1.keras.backend.clear_session()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
    tf.compat.v1.keras.backend.set_session(sess)
    tf.compat.v1.keras.backend.set_learning_phase(0)

    # load data
    batch_size = 100
    epsilon = mnist_loader().epsilon
    train_X, train_Y, _, _, testX, testY = mnist_loader().get_normalized_data()
    testX, testY = testX[0:threshold], testY[0:threshold]

    # convert to pytorch format
    testY = np.argmax(testY, axis=1)
    torch_testX = torch.from_numpy( np.transpose(testX, (0, 3, 1, 2)) ).float()
    torch_testY = torch.from_numpy( testY ).float()

    # # When you don't have a trained model uncomment the following lines and train and save a new model
    # tf_model = mnist_model()
    # tf_model.fit(train_X, train_Y)
    # tf_model.save('models/mnist_autoattack.h5')

    # If you already have a model, use the line below
    tf_model = tf.keras.models.load_model('models/mnist_autoattack.h5')

    # remove 'softmax layer' and put it into adapter
    atk_model = tf.keras.models.Model(inputs=tf_model.input, outputs=tf_model.get_layer(index=-2).output) 
    # atk_model.summary()
    y_input = tf.placeholder(tf.int64, shape = [None])
    x_input = atk_model.input
    logits  = atk_model.output
    model_adapted = utils_tf.ModelAdapter(logits, x_input, y_input, sess)

    # run attack
    adversary = AutoAttack(model_adapted, norm='Linf', eps=epsilon, version='standard', is_tf_model=True, device='cpu')
    x_adv, y_adv = adversary.run_standard_evaluation(torch_testX, torch_testY, bs=batch_size, return_labels=True)
    np_x_adv = np.moveaxis(x_adv.cpu().numpy(), 1, 3)
    np_x_adv_t = np.transpose(np_x_adv, (0, 3, 1, 2))
    testX_t = np.transpose(testX, (0, 3, 1, 2))
    return testX, np_x_adv, testX_t, np_x_adv_t

def linf(orig, adv):
    val = 0
    diff = [abs(orig[i]-adv[i]) for i in range(len(orig))]
    # print(diff)
    index = np.argmax(diff)
    nonZero = 0
    for i in diff:
        if i!=0:
            nonZero = nonZero + 1

    return diff[index], nonZero

def calculateNorms(orig_o, adv_o, orig, adv):
    model = tf.keras.models.load_model('models/mnist_autoattack.h5')
    l2Total, linfMax, pixels = 0, 0, 0
    count = len(orig)
    for i in range(len(orig)):
        original_array = orig[i][0].flatten()
        adversarial_array = adv[i][0].flatten()

        l2 = np.linalg.norm(np.array(original_array)-np.array(adversarial_array))
        l2Total += l2
        lin, pixelsModified = linf(original_array, adversarial_array)
        pixels += pixelsModified
        if lin>linfMax:
            linfMax = lin

    fid_calc = fid.calculate_fid(model, orig_o, adv_o)
    ps, success = fid.calculate_ps(model, orig_o, adv_o)

    print("For ", count, "Images:")
    print("Average L2 distance was: ", (l2Total)/count)
    print("Highest L-inf distance was: ", linfMax)
    print("Average number of pixels modified was: ", (pixels)/count)
    print("FID:", fid_calc)
    print("Pielou Measure is:", ps)
    print("Attack was succesfull on:", success)
    return (l2Total)/count, linfMax, (pixels)/count

if __name__ == '__main__':
    threshold = 1000
    t1 = time()
    orig, adv, orig_t, adv_t = getAdversarial(threshold)
    t2 = time()
    print("\n\n######################################################################")
    print("####################### PRINTING STATISTICS ##########################")
    print("######################################################################\n")
    print("Average time taken to generate images:", (t2-t1)/len(orig), " seconds.")
    l2, linf, pixels = calculateNorms(orig, adv, orig_t, adv_t)
    print()