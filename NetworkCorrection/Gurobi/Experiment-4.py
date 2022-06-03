import sys
# sys.path.append('../')
import numpy as np
import tensorflow as tf
import uuid
from ExperimentDumy import find

import argparse
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
import keras
# tf.compat.v1.disable_eager_execution()
"""
What this file does?
Creates a new model with the help of epsilon found by findCorrection.py
Epsilon is an array where each index of epsilon i.e epsilon[i] tells the amount by which w[i] of layer n-2 has to be changed.
That means for ayer n-2, the weights will be modified as: w[i] = w[i]+epsilon[i]
It modifies the 3rd last layer of the network by modifiying it's weights using the above epsilon.
"""
def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/testdp1_2_2op.nnet'
    model = obj.constructModel(fileName=file)
    print(type(model))
    print(model.summary())
    return model

def getInputs():
    inp = [-1, -1, -1, -1]
    return [inp]

def getOutputs():
    out = [1, -1]
    return [out]

def getEpsilons():
    model = loadModel()
    # inp = getmnist()
    inp = getInputs()

    num_inputs = len(inp)
    # print(model.summary())
    # sample_output = model.predict([inp])
    sample_output = getOutputs()
    true_label = (np.argmax(sample_output))
    num_outputs = len(sample_output[0])

    print(true_label)

    all_epsilons = find(5, model, inp[0], true_label, num_inputs, num_outputs, 1)
    
    return all_epsilons


# load_model_name = 'ACASXU_2_9'
# model_name = 'ACASXU_2_9_3'
model = loadModel()
epsilon = getEpsilons()
# print(epsilon)
"""
Change the name of the epsilon file according to what was generated in findCorrection.py
"""
print("Model loaded")
# ACASXU_2_9_0to04.vals.npy
weights = model.get_weights()
# print(weights)

print(len(epsilon))
print(len(weights))

# print(np.shape(epsilon))
# print(np.shape(weights))

weights[2] = weights[2]+ epsilon[0]

# weights = weights + epsilon
# # print(weights)
# # print(epsilon)
# """
# Got weights of the original model in line 32 and the epsilon in line 31 and now calculated new model weights in line 34.
# """
model.set_weights(weights)
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# utils.save_model('../Models/{}_corrected_mine.json'.format(model_name), '../Models/{}_corrected_mine.h5'.format(model_name), model)
# utils.saveModelAsProtobuf(model, '{}_corrected_mine'.format(model_name))

# sub_model, last_layer = utils.splitModel(model)
# # print("1")
# utils.saveModelAsProtobuf(last_layer, 'last.layer.{}_corrected_mine'.format(model_name))
# print("2")
sat_in = getInputs()
print(sat_in)
sat_out = getOutputs()

prediction = model.predict(sat_in)
print(prediction)
print(np.argmax(prediction, axis=1))
# np.save('../data/{}.prediction_mine'.format(model_name), prediction)    
