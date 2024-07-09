from attackMethod2 import *
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

"""
What this file does? 
It generates adversarial images using the attack algorithm and functions described in attack.py
The generated images are saved in an appropriate sub-folder in Images folder.
"""

def convertToMtarix(array, m, n, channels):
    for i in range(64*64*3):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m, n, channels))

def show(pixelMatrix, m, n, channels):
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='RGB')
    return im

def attack():
    inputs, outputs, count = getData()
    # read data from an image and flatten and normalize it

    print("Number of inputs in consideration: ",len(inputs))
    m, n, channels = 64, 64, 3
    folderSuffix = "_cifar64_XAI"

    for i in [16, 68]:
    # for i in range(count):
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        t1 = time()
        success, original, adversarial, true_label, adversarial_label, k = generateAdversarial_XAI(sat_in)
        if success==1:
            print(f"True Label{i}: ", true_label)
            print(f"Adv Label{i}: ", adversarial_label)
            print("...........................................................................................")
            print("Attack successful.")
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

            # image_original.save("../Images/OriginalImages"+folderSuffix+"/Image_"+str(i)+".jpg")
            # image_adversarial.save("../Images/AdversarialImages"+folderSuffix+"/Image_"+str(i)+".jpg")

            # image_original.save(f"./img_orig_{true_label}.jpg")
            # image_adversarial.save("../Images/AdversarialImages"+folderSuffix+"/Image_"+str(i)+".jpg")
            # break

attack()