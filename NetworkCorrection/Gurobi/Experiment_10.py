# from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
# from extractNetwork import extractNetwork

# def loadModel():
#     obj = ConvertNNETtoTensorFlow()
#     file = '../Models/ACASXU_run2a_1_2_batch_2000.nnet'
#     model = obj.constructModel(fileName=file)
#     # print(type(model))
#     # print(model.summary())
#     return model

# # o1 = extractNetwork()
# # originalModel = loadModel()
# # print(originalModel.summary())
# # print(o1.printActivations(originalModel))
# # modifiedModel = o1.extractModel(originalModel, 4)
# # print(modifiedModel.summary())
# # print(o1.printActivations(modifiedModel))


import random
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
import keras
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
Setting verbosity of tensorflow to minimum.
"""
from findModificationsLayerK import find
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow

# tf.compat.v1.disable_eager_execution()
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""
def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/ACASXU_run2a_1_2_batch_2000.nnet'
    model = obj.constructModel(fileName=file)
    # print(type(model))
    # print(model.summary())
    return model

# def loadModel():
#     json_file = open('../Models/ACASXU_2_9.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = keras.models.model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights("../Models/ACASXU_2_9.h5")
#     return loaded_model

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
            # print(w, b, result)
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        # print(neurons)
        return neurons

def getEpsilons(layer_to_change):
    model = loadModel()
    inp = getInputs()

    num_inputs = len(inp)
    sample_output = getOutputs()
    num_outputs = len(sample_output)
    true_label = np.argmax(model.predict([inp]))
    expected_label = random.randint(0, num_outputs-1)
    while true_label==expected_label:
        expected_label = random.randint(0, num_outputs-1)

    # expected_label = 0
    print(true_label, expected_label)
    all_epsilons = find(100, model, inp, true_label, num_inputs, num_outputs, 1, layer_to_change)
    
    return all_epsilons

def updateModel():
    num_layers = 7
    layer_to_change = int(num_layers/2)
    model = loadModel()
    epsilon = getEpsilons(layer_to_change)

    """
    Change the name of the epsilon file according to what was generated in findCorrection.py
    """
    print("Model loaded")
    # ACASXU_2_9_0to04.vals.npy
    weights = model.get_weights()

    weights[2*layer_to_change] = weights[2*layer_to_change]+ np.array(epsilon[0])

    model.set_weights(weights)
    model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    sat_in = getInputs()
    print(sat_in)
    sat_out = getOutputs()

    prediction = model.predict([sat_in])
    print("Final prediction: ",prediction)
    print(np.argmax(prediction[0]))
    neuron_values = get_neuron_values_actual(model, sat_in, num_layers)
    print(len(neuron_values))
    print(np.shape(neuron_values[layer_to_change]))

updateModel()