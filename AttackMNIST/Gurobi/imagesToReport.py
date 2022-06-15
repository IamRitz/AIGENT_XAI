from GurobiAttack import *
"""
This file displays image whose m*n pixels are given.
"""
from csv import reader
from math import ceil
from PIL import Image
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""

def convertToMtarix(array, m, n):
    for i in range(m*n):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m,n))

def show(pixelMatrix, w, h):
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='L')
    im = im.resize((56, 56))
    # im.show()
    return im

def attack():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=15
    m, n = 28, 28
    for i in range(count):
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        # print()
        t1 = time()
        success, original, adversarial, true_label, adversarial_label = generateAdversarial(sat_in)
        # print(success)
        if success==1:
            L2_norm = np.linalg.norm(np.array(original)-np.array(adversarial))
            print("...........................................................................................")
            print("Attack successful.")
            print("L2-norm is: ",L2_norm)
            t2 = time()
            print("Time taken in this iteration:", (t2-t1), "seconds.")
            print("...........................................................................................")
            """
            Now, here we will generate images for original and adversarial image.
            """
            mat1 = convertToMtarix(original, m, n)
            image_original = show(mat1, m, n)

            mat2 = convertToMtarix(adversarial, m, n)
            image_adversarial = show(mat2, m, n)

            image_original.save("OriginalImages/Image_"+str(i)+".jpg")
            image_adversarial.save("AdversarialImages/Image_"+str(i)+".jpg")
            # break


        
        # break
        # if i==0:
        #     break

attack()