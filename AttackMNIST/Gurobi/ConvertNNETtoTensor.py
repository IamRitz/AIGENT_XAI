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

class ConvertNNETtoTensorFlow:
    def read_nnet(self, file_name):
            with open(file_name) as f:
                line = f.readline()
                cnt = 1
                while line[0:2] == "//":
                    line = f.readline()
                    cnt += 1
                # numLayers does't include the input layer!
                numLayers, inputSize, outputSize, maxLayersize = [int(x) for x in line.strip().split(",")[:-1]]
                line = f.readline()

                # input layer size, layer1size, layer2size...
                layerSizes = [int(x) for x in line.strip().split(",")[:-1]]

                line = f.readline()
                symmetric = int(line.strip().split(",")[0])

                line = f.readline()
                inputMinimums = [float(x) for x in line.strip().split(",")[:-1]]

                line = f.readline()
                inputMaximums = [float(x) for x in line.strip().split(",")[:-1]]

                line = f.readline()
                means = [float(x) for x in line.strip().split(",")[:-1]]

                line = f.readline()
                ranges = [float(x) for x in line.strip().split(",")[:-1]]

                weights = []
                biases = []
                for layernum in range(numLayers):

                    previousLayerSize = layerSizes[layernum]
                    currentLayerSize = layerSizes[layernum + 1]
                    # weights
                    weights.append([])
                    biases.append([])
                    # weights
                    for i in range(currentLayerSize):
                        line = f.readline()
                        aux = [float(x) for x in line.strip().split(",")[:-1]]
                        weights[layernum].append([])
                        for j in range(previousLayerSize):
                            weights[layernum][i].append(aux[j])
                    # biases
                    for i in range(currentLayerSize):
                        line = f.readline()
                        x = float(line.strip().split(",")[0])
                        biases[layernum].append(x)
                return weights, biases

    def constructModel(self, fileName):
        w, biases = self.read_nnet(fileName)
        model = Sequential()
        i = 0
        input_shape = np.shape(w[i])[1]
        num_nodes = np.shape(w[i])[0]
        model.add(Dense(num_nodes, input_dim = input_shape, activation= 'relu'))
        i = i + 1
        while i<(len(biases))-1:
            num_nodes = np.shape(w[i])[0]
            model.add(Dense(num_nodes, activation= 'relu'))
            i = i + 1
        num_nodes = np.shape(w[i])[0]
        model.add(Dense(num_nodes))

        weights = []
        for i in range(len(biases)):
            weights.append(np.array(w[i]).T)
            weights.append(np.array(biases[i]))
        
        model.set_weights(weights)
        
        # print("Model retreived.")
        return model

    def predict(self, model, inputToModel, actual_output):
        predicted_output = model.predict(np.array([inputToModel]))
        return predicted_output
