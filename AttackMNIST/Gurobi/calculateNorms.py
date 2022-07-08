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
from PIL import Image
from numpy import asarray
from scipy import stats


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

def getImages():
    paths = []
    count = 0
    folderCount = 2
    for path in os.listdir("AdversarialImages"+str(folderCount)):
        paths.append(str(path))
        # print(path)
        count = count+1
    
    l2Total = 0
    linfTotal = 0
    totalPixelsModified = []
    for i in range(count):
        original = Image.open('OriginalImages'+str(folderCount)+'/'+paths[i])
        original = original.resize((28, 28))
        original_array = asarray(original).reshape(28*28)
        original_array = original_array/255.0
        # print(original_array)

        adversarial = Image.open('AdversarialImages'+str(folderCount)+'/'+paths[i])
        adversarial = adversarial.resize((28, 28))
        adversarial_array = asarray(adversarial).reshape(28*28)
        adversarial_array = adversarial_array/255.0
        # print(adversarial_array)

        l2 = np.linalg.norm(np.array(original_array)-np.array(adversarial_array))
        l2Total += l2
        lin, pixelsModified = linf(original_array, adversarial_array)
        linfTotal += lin
        totalPixelsModified.append(pixelsModified)
        # print("For image:", i, " L2 norm is: ", l2, " L-inf norm is: ", lin)
        # break
    
    print("For ", count, "Images:")
    print("Average L2 distance was: ", (l2Total)/count)
    print("Average L-inf distance was: ", (linfTotal)/count)
    print("Number of pixels modified (Mode): ", stats.mode(totalPixelsModified))
    print("Number of pixels modified (Median): ", np.mean(totalPixelsModified))

getImages()