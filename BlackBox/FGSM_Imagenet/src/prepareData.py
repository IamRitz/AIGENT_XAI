import os
from re import I
from PIL import Image
from numpy import asarray
import numpy as np
import cv2
import numpy as np
import random
import csv

"""
This file prepares images in the required format.
"""

def convertChannel(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = np.zeros_like(img)
    img2[:,:,0] = gray
    img2[:,:,1] = gray
    img2[:,:,2] = gray
    cv2.imwrite(path, img2)

def convertImage(image, w, h):
    img = Image.open(image)
    img = img.resize((w, h))
    if len(np.shape(img))==2:
        convertChannel(image)
        img = Image.open(image)
        img = img.resize((w, h))
    numpydata = asarray(img).reshape(w*h*3)/255
    return numpydata

def convertToMtarix(array, m, n, channels):
    print(np.shape(array))
    for i in range(128*128*3):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m, n, channels))

def showing(pixelMatrix, m, n, channels):
    print(np.shape(pixelMatrix))
    pixelMatrix = convertToMtarix(pixelMatrix, 128, 128, 3)
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    return im

def getTrainData(w, h):
    labels = {}
    i = 0
    for path in os.listdir("../data/train"):
        labels[path] = i
        i = i+1
    
    X_train = []
    Y_train = []

    print("Fetching data...")
    for folder in os.listdir("../data/train"):
        i = 0
        print("Loading for:", folder)
        for path in os.listdir("../data/train/"+folder):
            img = convertImage("../data/train/"+folder+"/"+path, w, h)
            X_train.append(img)
            Y_train.append([labels[folder]])
            i += 1
            if i>=40:
                break
        
    temp = list(zip(X_train, Y_train))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return np.array(res1), np.array(res2)

def getValData(w, h):
    labels = {}
    i = 0
    for path in os.listdir("../data/val"):
        labels[path] = i
        i = i+1

    X_test = []
    Y_test = []
    for folder in os.listdir("../data/val"):
        for path in os.listdir("../data/val/"+folder):
            img = convertImage("../data/val/"+folder+"/"+path, w, h)
            X_test.append(img)
            Y_test.append(labels[folder])

    temp = list(zip(X_test, Y_test))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return np.array(res1), np.array(res2)

def getSingleImage(imageName, label, w, h):
    X_test = []
    Y_test = []
    img = convertImage("../Images/"+imageName, w, h)
    X_test.append(img)
    Y_test.append(label)

    temp = list(zip(X_test, Y_test))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return np.array(res1), np.array(res2)


