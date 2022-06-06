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
from Experiment_2 import find
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow

# tf.compat.v1.disable_eager_execution()
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""
def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/ACASXU_run2a_1_6_batch_2000.nnet'
    model = obj.constructModel(fileName=file)
    # print(type(model))
    # print(model.summary())
    return model

def getInputs():
    inp = [0.6399288845, 0.0, 0.0, 0.475, -0.475]
    return inp

def getOutputs():
    output_1 = [-0.0203966, -0.01847511, -0.01822628, -0.01796024, -0.01798192]
    output_2 = [-0.01942023, -0.01750685, -0.01795192, -0.01650293, -0.01686228]
    output_3 = [ 0.02039307, 0.01997121, -0.02107569, 0.02101956, -0.0119698 ]
    return output_3

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
    num_outputs = len(sample_output)

    print(true_label)

    all_epsilons = find(10, model, inp, true_label, num_inputs, num_outputs, 1)
    
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
# print("__________________________________________\n",epsilon[0],"\n__________________________________________")
model.set_weights(weights)
model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# print("Weights after change:")
# print(model.get_weights())

sat_in = getInputs()
print(sat_in)
sat_out = getOutputs()

prediction = model.predict([sat_in])
print("Final prediction: ",prediction)
print(np.argmax(prediction[0]))

# get_neuron_values_actual(model, sat_in, 2)