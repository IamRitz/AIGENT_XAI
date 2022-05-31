import sys
# sys.path.append('../')
import numpy as np
import tensorflow as tf
import uuid
from WatermarkRemoval import utils
import argparse

tf.compat.v1.disable_eager_execution()
"""
What this file does?
Creates a new model with the help of epsilon found by findCorrection.py
Epsilon is an array where each index of epsilon i.e epsilon[i] tells the amount by which w[i] of layer n-2 has to be changed.
That means for ayer n-2, the weights will be modified as: w[i] = w[i]+epsilon[i]
It modifies the 3rd last layer of the network by modifiying it's weights using the above epsilon.
"""


parser = argparse.ArgumentParser()
parser.add_argument('--load_model', default='ACASXU_2_9', help='the name of the model')
parser.add_argument('--model', default='ACASXU_2_9_0', help='the name of the model')
args = parser.parse_args()

load_model_name = args.load_model
model_name = args.model


# load_model_name = 'ACASXU_2_9'
# model_name = 'ACASXU_2_9_3'
model = utils.load_model('../Models/{}.json'.format(load_model_name), '../Models/{}.h5'.format(load_model_name))
epsilon = np.load('../data/{}to04.vals.npy'.format(model_name))
"""
Change the name of the epsilon file according to what was generated in findCorrection.py
"""
# ACASXU_2_9_0to04.vals.npy
weights = model.get_weights()
print(weights[-2])
weights[-2] = weights[-2] + epsilon
print(weights[-2])
print(epsilon)
"""
Got weights of the original model in line 32 and the epsilon in line 31 and now calculated new model weights in line 34.
"""
model.set_weights(weights)
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

utils.save_model('../Models/{}_corrected.json'.format(model_name), '../Models/{}_corrected.h5'.format(model_name), model)
utils.saveModelAsProtobuf(model, '{}_corrected'.format(model_name))

sub_model, last_layer = utils.splitModel(model)
# print("1")
utils.saveModelAsProtobuf(last_layer, 'last.layer.{}_corrected'.format(model_name))
# print("2")

datafile = open('../data/inputs.csv')
sat_in = np.array([[float(x) for x in line.split(',')] for line in datafile])
print(sat_in)
datafile.close()
datafile = open('../data/outputs.csv')
sat_out = np.array([[float(x) for x in line.split(',')] for line in datafile])
datafile.close()

prediction = model.predict(sat_in)
print(prediction)
print(np.argmin(prediction, axis=1))
np.save('../data/{}.prediction'.format(model_name), prediction)    
