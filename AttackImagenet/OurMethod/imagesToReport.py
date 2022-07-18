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

def convertToMtarix(array, m, n, channels):
    for i in range(128*128*3):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m, n, channels))

def showing(pixelMatrix, m, n, channels):
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    # im = im.resize((m, n, channels))
    # im.show()
    return im

def attack():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=0
    m, n, channels = 128, 128, 3
    for i in range(count):
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        # true_output = 
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
            # print(np.shape(original))
            mat1 = convertToMtarix(sat_in, m, n, channels)
            image_original = showing(mat1, m, n, channels)

            mat2 = convertToMtarix(adversarial, m, n, channels)
            image_adversarial = showing(mat2, m, n, channels)
            print("Saving to..")
            image_original.save("OriginalImages/Image_"+str(i)+".jpg")
            image_adversarial.save("AdversarialImages/Image_"+str(i)+".jpg")

attack()