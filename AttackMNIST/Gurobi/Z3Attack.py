from cProfile import label
from csv import reader
from time import time

from numpy import genfromtxt
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
from extractNetwork import extractNetwork
import random
import numpy as np
import os
import gurobipy as gp
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
from gurobipy import GRB
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""
counter=0

def loadModel():
    # obj = ConvertNNETtoTensorFlow()
    # file = '../Models/ACASXU_run2a_1_6_batch_2000.nnet'
    # model = obj.constructModel(fileName=file)
    # print(type(model))
    # print(model.summary())
    model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
    # print(model.summary())
    return model

def getData():
    inputs = []
    outputs = []
    f1 = open('MNISTdata/inputs.csv', 'r')
    f1_reader = reader(f1)
    stopAt = 500
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

# def getInputs():
    inputs, outputs, count = getData()
    return inputs[0]

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

def getEpsilons(layer_to_change, inp):
    model = loadModel()
    # inp = getInputs()

    num_inputs = len(inp)
    # print(type(inp), np.shape(inp))
    sample_output = model.predict(np.array([inp]))[0]
    # print(sample_output)
    # num_outputs = len(sample_output)
    true_label = np.argmax(sample_output)
    num_outputs = len(sample_output)
    # expected_label = random.randint(0, 1000)%(num_outputs-1)
    # while true_label==expected_label:
    #     expected_label = random.randint(0, 1000)%(num_outputs-1)
    expected_label = sample_output.argsort()[-2]
    print("Attack label is:", expected_label)
    # expected_label = 4
    # print(true_label, expected_label)
    all_epsilons = find(10, model, inp, expected_label, num_inputs, num_outputs, 1, layer_to_change)
    
    return all_epsilons, inp

def predict(epsilon, layer_to_change, sat_in):
    # print("predicting for: ", layer_to_change)
    model = loadModel()

    """
    Change the name of the epsilon file according to what was generated in findCorrection.py
    """
    # layer_to_change = 0
    weights = model.get_weights()

    weights[2*layer_to_change] = weights[2*layer_to_change]+ np.array(epsilon[0])

    model.set_weights(weights)
    model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # sat_in = getInputs()
    # print(sat_in)
    prediction = model.predict([sat_in])
    # print("Final prediction: ",prediction)
    print("Prediction: ",np.argmax(prediction[0]))
    return model

def updateModel(sat_in):
    model = loadModel()
    num_layers = int(len(model.get_weights())/2)
    layer_to_change = int(num_layers/2)
    # layer_to_change = 0
    print("Modifying weights for weight layer:",layer_to_change)
    originalModel = model
    # print("Layer to change in this iteration:", layer_to_change)
    epsilon, inp = getEpsilons(layer_to_change, sat_in)
    # sat_in = getInputs()
    # print("Model loaded")
    # ACASXU_2_9_0to04.vals.npy
    tempModel = predict(epsilon, layer_to_change, sat_in)
    # print("...........................................................................................")
    # print("Dividing Network.")
    # print("...........................................................................................")
    """
    Now we have modifications in the middle layer of the netwrok.
    Next, we will run a loop to divide the network and find modifications in lower half of the network.
    """
    o1 = extractNetwork()
    o3 = labelNeurons()
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[layer_to_change]

    # phases_original = get_neuron_values_actual(originalModel, sat_in, num_layers)
    # neuron_values_2 = phases[layer_to_change-1]

    # for i in range(len(neuron_values_1)):
    #     print(neuron_values_1[i], neuron_values_2[i])
    #     print("\n\n")
    # all_epsilons2 = epsilon
    # extractedNetwork = o1.extractModel(originalModel, 1)
    # layer_to_change = 0
    while layer_to_change>0:
        # print("##############################")
        
        print("Extracting model till layer: ", layer_to_change+1)
        extractedNetwork = o1.extractModel(originalModel, layer_to_change+1)
        # print(extractedNetwork.summary())
        # print("Number of weights: ",len(extractedNetwork.get_weights())/2)
        # print(len(extractedNetwork.get_weights()))
        layer_to_change = int(layer_to_change/2)
        # print("Applying modifications to: ", layer_to_change)
        
        epsilon = find2(10, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases)

        tempModel = predict(epsilon, layer_to_change, sat_in)
        phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        neuron_values_1 = phases[layer_to_change]
        # print("...........................................................................................")
        # print("Dividing Network.")
        # print("...........................................................................................")
    return extractedNetwork, neuron_values_1,  epsilon 

def ReLU(input):
        return np.vectorize(lambda y: z3.If(y>=0, y, z3.RealVal(0)))(input)

def add(m, expr):
    global counter
    # if counter==19 or counter==27 or counter==22 or counter==35 or counter==37 or counter==49:
    #     return
    m.assert_and_track(expr, "Constraint_: "+str(counter)+":"+str(expr))
    counter = counter + 1

def Z3Attack(inputs, model, outputs):
    delta_max = 10
    m = z3.Solver()  
    m.set(unsat_core=True)  
    delta = z3.RealVector('delta',len(inputs))
    input_vars = z3.RealVector('input_vars',len(inputs))

    for i in range(len(inputs)):
        add(m, z3.And(input_vars[i]>=inputs[i]-delta[i], input_vars[i]<=inputs[i]+delta[i]))
        # add(m, input_vars[i]!= 0)
        add(m, z3.And(delta[i]>=0, delta[i]<=abs(inputs[i])))
    
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    out = w.T @ input_vars + b
    # print(out)
    layer_output = ReLU(out)
    tolerance = 10

    for i in range(len(outputs)):
        if outputs[i]>0:
            add(m, out[i]<=outputs[i]+tolerance)
            add(m, out[i]>=outputs[i]-tolerance)
        else:
            add(m, out[i]<=tolerance)
        # add(m, layer_output[i]==outputs[i])
   
    solution = m.check()
    # print(solution)
    # print(inputs)
    ad_inp = []
    if solution==z3.sat:
        # print(out)
        solution = m.model()
        dictionary = sorted ([(d, solution[d]) for d in solution], key = lambda x: str(x[0]))
        i = 1
        for x in dictionary:
            if "Constraint" in str(x[0]):
                continue
            r = x[1]
            # print(x, r, type(r))
            val = 0
            # print(r)
            if z3.is_algebraic_value(r):
                r = r.approx(20)
            val = float(r.numerator_as_long())/float(r.denominator_as_long())
            # print(x[0], val)
            if "input" in str(x[0]):
                ad_inp.append(val)

    else:
        # print(m.unsat_core())
        print("UNSAT")
    
    # print(ad_inp)
    return ad_inp


def findMetric(sat_in, ad_inp):
    lInf, sumDist = 0, 0 
    for i in range(len(sat_in)):
        sumDist = sumDist + abs(sat_in[i]-ad_inp[i])
        if abs(sat_in[i]-ad_inp[i])>lInf:
            lInf = abs(sat_in[i]-ad_inp[i])
    return lInf, sumDist

def generateAdversarial(sat_in):
    try:
        extractedModel, neuron_values_1, epsilon = updateModel(sat_in)
    except:
        print("UNSAT. Could not find a minimal modification by divide and conquer.")
        return 0
    
    # print("Finally, we have layer 0 modifications.")
    tempModel = predict(epsilon, 0, sat_in)
    
    
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
    ad_inp = Z3Attack(sat_in, extractedModel, neuron_values_1)
    # # ad_inp = [1,0,-1, 2]
    originalModel = loadModel()
    true_output = originalModel.predict([sat_in])
    true_label = np.argmax(true_output)
    print("True Label is: ", true_label)
    # print(true_output)
    if len(ad_inp)>0:
        ad_output = originalModel.predict([ad_inp])
        print("Output after attack: ", ad_output)
        predicted_label = np.argmax(ad_output)
        print("Label after attack: ", predicted_label)
        vals = get_neuron_values_actual(originalModel, ad_inp, num_layers)
        ch = 0
        max_shift = 0
        for i in range(len(vals[0])):
            ch = ch + abs(vals[0][i]-neuron_values_1[i])
            if abs(vals[0][i]-neuron_values_1[i])>max_shift:
                max_shift = abs(vals[0][i]-neuron_values_1[i])
            # print(vals[0][i], neuron_values_1[i])
        print("Overall shift : ", ch)
        print("Maximum shift: ", max_shift)
        linf, sumDist = findMetric(sat_in, ad_inp)
        if predicted_label!=true_label:
            print(linf, sumDist)
            print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
            # print(sat_in)
            return 1
    return 0
    
def attack():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=15
    # indexes = [15 ,
    #     29 ,
    #     38 ,
    #     42 ,
    #     52 ,
    #     71 ,
    #     79 ,
    #     84 ,
    #     91 ,
    #     96 ]
    indexes = [13 ,
        18 ,
        32 ,
        36 ,
        39 ,
        62 ,
        66 ,
        73 ,
        83 ,
        90 ,
        93]
    # count=0
    # for i in range(784):
    #     # print(indexes[i])
    #     f = 0
    #     c = inputs[indexes[0]][i]
    #     for j in range(0, 50):
    #         if inputs[j][i]!=c:
    #             f=-1
    #     if f==0:
    #         print(i, c)
    #         count=count+1
    # print(count)
    for i in range(count):
        print("...........................................................................................")
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        print()
        t1 = time()
        success = generateAdversarial(sat_in)
        t2 = time()
        print("Time taken in this iteration:", (t2-t1), "seconds.")
        print("...........................................................................................")
        # break
        # if i==5:
        #     break

attack()