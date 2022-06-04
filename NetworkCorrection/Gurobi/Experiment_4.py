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
from Experiment_1 import find
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow

# tf.compat.v1.disable_eager_execution()
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""
def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/testdp1_2_2op.nnet'
    model = obj.constructModel(fileName=file)
    # print(type(model))
    # print(model.summary())
    return model

def getInputs():
    inp = [-1, -1, -1, -1]
    return [inp]

def getOutputs():
    out = [1, -1]
    return [out]

def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 1
        # print(len(loaded_model.layers))
        for layer in loaded_model.layers:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # print(w)
            result = np.matmul(input,w)+b
            # print(l)
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            print(w, b, result)
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        print(neurons)
        return neurons

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

    all_epsilons = find(1000, model, inp[0], true_label, num_inputs, num_outputs, 1)
    
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

# print(len(epsilon))
# print(len(weights))

# print("Weights before change:")
# print(model.get_weights())
weights[0] = weights[0]+ np.array(epsilon[0])
print("__________________________________________\n",weights[0].T,"\n__________________________________________")
model.set_weights(weights)
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# print("Weights after change:")
# print(model.get_weights())

sat_in = getInputs()
print(sat_in)
sat_out = getOutputs()

prediction = model.predict(sat_in)
print("Final prediction: ",prediction)
print(np.argmax(prediction, axis=1))

get_neuron_values_actual(model, sat_in[0], 2)