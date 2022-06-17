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
import argparse
from time import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobipy as grb
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
"""
Finds minimal modification only in Layer 0 for the toy example given by Madhukar Sir so that Output1 is greater than Output 0.
"""
def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/testdp1_2_2opModifiedZ3.nnet'
    model = obj.constructModel(fileName=file)
    print(type(model))
    # print(model.summary())
    return model

model = loadModel()
predict = model.predict([[-1, -1, -1, -1]])
print(predict)