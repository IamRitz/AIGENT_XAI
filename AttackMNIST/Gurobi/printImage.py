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

def getData():
    inputs = []
    outputs = []
    f1 = open('MNISTdata/inputs.csv', 'r')
    f1_reader = reader(f1)
    stopAt = 5
    f2 = open('MNISTdata/outputs.csv', 'r')
    f2_reader = reader(f2)
    i=0
    for row in f1_reader:
        inp = [float(x) for x in row]
        inputs.append(inp)
        i=i+1
        if i==stopAt:
            break
        # break
    i=0
    for row in f2_reader:
        out = [float(x) for x in row]
        outputs.append(out)
        i=i+1
        if i==stopAt:
            break
        # break

    return inputs, outputs, len(inputs)

def convertToMtarix(array, m, n):
    for i in range(m*n):
        array[i] = 255*array[i]
    matrix = np.array(array)
    return matrix.reshape((m,n))

def show(pixelMatrix, w, h):
    data = np.array(pixelMatrix)
    im = Image.fromarray(data.astype(np.uint8), mode='L')
    im = im.resize((512, 512))
    im.show()

inputs, outputs, count = getData()
for i in range(count):
    image = inputs[i]
    imageMatrix = convertToMtarix(image, 28, 28)
    show(imageMatrix, 28, 28)
    break