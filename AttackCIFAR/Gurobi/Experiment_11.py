from cProfile import label
from csv import reader
from time import time

from numpy import genfromtxt
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
from extractNetwork import extractNetwork
import random
import numpy as np
import os
import gurobipy as gp
import z3
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
To supress the tensorflow warnings. 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
Setting verbosity of tensorflow to minimum.
"""
from findModificationsLayerK import find as find
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
from modificationDivided import find as find2
from labelNeurons import labelNeurons
from gurobipy import GRB


def loadModel():
    model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
    return model


def getData():
    f1 = open('MNISTdata/inputs.csv', 'r')
    f1_reader = reader(f1)
    model = loadModel()
    i=0
    for row in f1_reader:
        inp = [float(x) for x in row]
        prediction = model.predict([inp])
        out = np.argmax(prediction)
        if out==6:
            print(i,",")
            # break
        i=i+1

getData()
        