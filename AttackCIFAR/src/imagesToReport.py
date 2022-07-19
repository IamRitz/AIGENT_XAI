from GurobiAttack import *
"""
This file displays image whose m*n pixels are given.
"""
from PIL import Image
import numpy as np
from PIL import Image
from csv import reader
from time import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
To supress the tensorflow warnings. 
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
Setting verbosity of tensorflow to minimum.
"""
from findModificationsLayerK import find as find

def convertToMtarix(array, m, n, channels):
    for i in range(32*32*3):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m, n, channels))

def show(pixelMatrix, m, n, channels):
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    # im = im.resize((m, n, channels))
    # im.show()
    return im

def attack():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=15
    m, n, channels = 32, 32, 3
    for i in range(count):
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        # print()
        t1 = time()
        success, original, adversarial, true_label, adversarial_label, L2_norm, linf, k = generateAdversarial(sat_in)
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
            mat1 = convertToMtarix(original, m, n, channels)
            image_original = show(mat1, m, n, channels)

            mat2 = convertToMtarix(adversarial, m, n, channels)
            image_adversarial = show(mat2, m, n, channels)

            image_original.save("OriginalImages/Image_"+str(i)+".jpg")
            image_adversarial.save("AdversarialImages/Image_"+str(i)+".jpg")
            # break

attack()