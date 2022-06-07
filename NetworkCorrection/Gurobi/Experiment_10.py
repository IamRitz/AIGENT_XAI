from cProfile import label
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
from extractNetwork import extractNetwork
import random
import numpy as np
import os
import z3
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
from findModificationsLayerK import find as find
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
from modificationDivided import find as find2
from labelNeurons import labelNeurons
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""
counter=0

def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/ACASXU_run2a_1_2_batch_2000.nnet'
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
    expected_label = random.randint(0, 1000)%(num_outputs-1)
    while true_label==expected_label:
        expected_label = random.randint(0, 1000)%(num_outputs-1)

    # expected_label = 0
    print(true_label, expected_label)
    all_epsilons = find(10, model, inp, true_label, num_inputs, num_outputs, 1, layer_to_change)
    
    return all_epsilons, inp

def predict(epsilon, layer_to_change):
    print("predicting for: ", layer_to_change)
    model = loadModel()

    """
    Change the name of the epsilon file according to what was generated in findCorrection.py
    """
    # layer_to_change = 0
    weights = model.get_weights()

    weights[2*layer_to_change] = weights[2*layer_to_change]+ np.array(epsilon[0])

    model.set_weights(weights)
    model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    sat_in = getInputs()
    # print(sat_in)
    prediction = model.predict([sat_in])
    print("Final prediction: ",prediction)
    print(np.argmax(prediction[0]))
    return model

def updateModel():
    num_layers = 7
    layer_to_change = int(num_layers/2)
    # layer_to_change = 0
    model = loadModel()
    originalModel = model
    print("Layer to change in this iteration:", layer_to_change)
    epsilon, inp = getEpsilons(layer_to_change)
    sat_in = getInputs()
    # print("Model loaded")
    # ACASXU_2_9_0to04.vals.npy
    tempModel = predict(epsilon, layer_to_change)
    print("...........................................................................................")
    print("Dividing Network.")
    print("...........................................................................................")
    """
    Now we have modifications in the middle layer of the netwrok.
    Next, we will run a loop to divide the network and find modifications in lower half of the network.
    """
    o1 = extractNetwork()
    o3 = labelNeurons()
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[layer_to_change-1]
    # all_epsilons2 = epsilon

    while layer_to_change>0:
        print("Extracting model till layer: ", layer_to_change)
        extractedNetwork = o1.extractModel(originalModel, layer_to_change)
        print(len(extractedNetwork.get_weights()))
        layer_to_change = int(layer_to_change/2)
        print("Applying modifications to: ", layer_to_change)
        
        epsilon = find2(10, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases)

        tempModel = predict(epsilon, layer_to_change)
        phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        neuron_values_1 = phases[layer_to_change-1]
        print("...........................................................................................")
        print("Dividing Network.")
        print("...........................................................................................")
    return extractedNetwork, neuron_values_1,  epsilon 

def ReLU(input):
        return np.vectorize(lambda y: z3.If(y>=0, y, z3.RealVal(0)))(input)

def add(m, expr):
    global counter
    m.assert_and_track(expr, "Constraint_: "+str(counter))
    counter = counter + 1

def Z3Attack(inputs, model, outputs):
    delta_max = 100000
    m = z3.Solver()  
    m.set(unsat_core=True)  
    delta = z3.RealVector('delta',len(inputs))
    input_vars = z3.RealVector('input_vars',len(inputs))

    for i in range(len(inputs)):
        add(m, z3.And(input_vars[i]>=inputs[i]-delta[i], input_vars[i]<=inputs[i]+delta[i]))
        add(m, z3.And(delta[i]>=0, delta[i]<=delta_max))
    
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    out = w.T @ input_vars + b
    # print(out)
    layer_output = ReLU(out)
    
    for i in range(len(outputs)):
        if outputs[i]>0:
            add(m, out[i]==outputs[i])
        else:
            add(m, out[i]<=0)
        # add(m, layer_output[i]==outputs[i])
   
    solution = m.check()
    print(solution)
    if solution==z3.sat:
        print("SAT")
    else:
        print(m.unsat_core())
        
    return 0

def generateAdversarial():
    extractedModel, neuron_values_1, epsilon = updateModel()
    print("Finally, we have layer 0 modifications.")
    tempModel = predict(epsilon, 0)
    
    sat_in = getInputs()
    num_layers = int(len(tempModel.get_weights())/2)
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[0]
    # for p in neuron_values_1:
    #     print(p)
    """
    Now, I have all the epsilons which are to be added to layer 0. 
    Left over task: Find delta such that input+delta can give the same effect as update model
    We want the outputs of hidden layer 1 to be equal to the values stored in neuron_values_1
    """
    Z3Attack(sat_in, extractedModel, neuron_values_1)
    # print(all_epsilons)

generateAdversarial()