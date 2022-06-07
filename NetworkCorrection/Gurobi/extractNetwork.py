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
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

"""
This file converts a neural network saved in nnet format to a tensorflow model.
"""
class extractNetwork:
    def extractModel(self, model, layer_count):
        # print(model.summary())
        print("Extracting till layer: ", layer_count)
        weights = model.get_weights()
        # print(len(weights))
        modifiedModel = Sequential()
        i = 0
        
        input_shape = np.shape(weights[i])[0]
        num_nodes = np.shape(weights[i])[1]
        if layer_count>1:
            modifiedModel.add(Dense(num_nodes, input_dim = input_shape, activation= 'relu'))
        else:
            modifiedModel.add(Dense(num_nodes, input_dim = input_shape))
        i = i + 2
        while i<(2*layer_count)-2:
            print("Adding hidden layer.")
            num_nodes = np.shape(weights[i])[1]
            modifiedModel.add(Dense(num_nodes, activation= 'relu'))
            i = i + 2
        if layer_count>1:
            num_nodes = np.shape(weights[i])[1]
            modifiedModel.add(Dense(num_nodes))

        weights_to_set = []
        for i in range(0, 2*layer_count, 2):
            weights_to_set.append(np.array(weights[i]))
            weights_to_set.append(np.array(weights[i+1]))
        
        # print((modifiedModel.get_weights()))
        # print(modifiedModel.summary())
        # print(len(weights_to_set))
        modifiedModel.set_weights(weights_to_set)
        
        # print("Model retreived.")
        return modifiedModel

    def printActivations(self, model):
        for i, layer in enumerate (model.layers):
            print (i, layer)
            try:
                print ("    ",layer.activation)
            except AttributeError:
                print('   no activation attribute')