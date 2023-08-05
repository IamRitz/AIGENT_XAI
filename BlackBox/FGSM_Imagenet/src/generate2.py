from time import time
from pielouMeasure import PielouMeaure
from FID import calculate_fid, getImages
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
import numpy as np
from tensorflow.keras.losses import MSE
import tensorflow as tf
from PIL import Image
import numpy as np
from prepareData import getTrainData, getValData, getSingleImage

"""
This file generates the adversarial images for a subset of the Imagenet dataset 
using the Blackbox attack described in: https://arxiv.org/pdf/2001.11137.pdf
"""

def getData(imageName, label, w, h):
    # X_test, y_test = getValData(w, h)
    X_test, y_test = getSingleImage(imageName, label, w, h)
    return X_test, y_test

def getModel(imageName, label):
    w, h = 128, 128
    print("[INFO] loading dataset...")
    X_test, y_test = getData(imageName, label, w, h)
    testX = np.expand_dims(X_test, axis=-1)
    testY = to_categorical(y_test, 10)
    model = tf.keras.models.load_model('../Models/imagenette_3.h5')
    return model, testX, testY

def generate_image_adversary(model, image, label, eps=2 / 255.0):
  w, h = 128, 128
  image = tf.cast(np.array(image).reshape(1, w*h*3), tf.float32)
  
  with tf.GradientTape() as tape:
    tape.watch(image)
    pred = model(image)
    loss = MSE(label, pred)
    gradient = tape.gradient(loss, image)
    signedGrad = tf.sign(gradient)
    adversary = (image+ (signedGrad*eps)).numpy()
    return adversary

def convertToMtarix(array, m, n, channels):
    for i in range(m*n*channels):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m,n, channels))

def show(pixelMatrix, w, h, channels):
    pixelMatrix = convertToMtarix(pixelMatrix[0], w, h, channels)
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    return im

def generateAdversarial(model, testX, testY, folderSuffix):
    countOriginal = [0]*10
    countAdversary = [0]*10
    avConf = 0
    totalL2 = 0
    totalLinf = 0
    avCount = 0
    success = 0
    w, h = 128, 128
    parentDir = "../Images/"

    for i in range(1):
        image = testX[i]
        original_array = np.array(image).reshape(1, w*h*3)
        label = testY[i]
        adversary = generate_image_adversary(model, image.reshape(1, w, h, 3), label, eps=13/255)
        pred = model.predict(adversary.reshape(1, w*h*3))
        adversary = adversary.reshape(w, h, 3)
        adversarial_array = np.array(adversary).reshape(1, w*h*3)
        
        originalImage = show(original_array, w, h, 3)
        adversarialImage = show(adversarial_array, w, h, 3)

        originalLabel = np.argmax(label)
        adversaryLabel = np.argmax(pred)

        originalImage.save(parentDir+"/Image2Orig_"+str(i)+".jpg")
        adversarialImage.save(parentDir+"/Image2Adv_"+str(i)+".jpg")

        print(originalLabel, adversaryLabel)
        print("..........................................")
    return avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary

def displayResults():
    folderSuffix = ""
    w, h = 128, 128
    imageName, label = "Image_91.jpg", 1
    model, textX, textY = getModel(imageName, label)
    t1 = time()
    avConf, totalL2, totalLinf, avCount, success, countOriginal, countAdversary = generateAdversarial(model, textX, textY, folderSuffix)
    t2 = time()

displayResults()