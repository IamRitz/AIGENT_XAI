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
# from labelNeurons import labelNeurons
from gurobipy import GRB
import random
"""
What this file does?
Calls any particular Experiment file to get the epsilons generated.
Updates the original network with the epsilons and generates a comparison between original and modified network.
"""
counter=0

def loadModel():
    model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/cifar.h5')
    return model

def getData():
    inputs = []
    outputs = []
    f1 = open('../data/inputs.csv', 'r')
    f1_reader = reader(f1)
    stopAt = 500
    f2 = open('../data/outputs.csv', 'r')
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
    # print("Modifying weights for weight layer:",layer_to_change)
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
    # o3 = labelNeurons()
    # phases1 = get_neuron_values_actual(originalModel, sat_in, num_layers)
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[layer_to_change]
    # for i in range(len(neuron_values_1)):
    #     print(phases[layer_to_change-1][i], phases1[layer_to_change-1][i])
    while layer_to_change>0:
        # print("##############################")
        
        # print("Extracting model till layer: ", layer_to_change+1)
        extractedNetwork = o1.extractModel(originalModel, layer_to_change+1)
        # print(extractedNetwork.summary())
        # print("Number of weights: ",len(extractedNetwork.get_weights())/2)
        # print(len(extractedNetwork.get_weights()))
        layer_to_change = int(layer_to_change/2)
        # print("Applying modifications to: ", layer_to_change)
        
        epsilon = find2(100, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases)

        tempModel = predict(epsilon, layer_to_change, sat_in)
        phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        neuron_values_1 = phases[layer_to_change]
        # print("....................................")
        # print("Dividing Network.")
        # print("....................................")
    return extractedNetwork, neuron_values_1,  epsilon 

def GurobiAttack(inputs, model, outputs, k):
    print("Launching attack with Gurobi.")
    tolerance = 10
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    x = []
    input_vars = []
    max_change = m.addVar(lb = 0, ub = tolerance, vtype=GRB.CONTINUOUS, name="max_change")

    changes = []

    for i in range(len(inputs)):
        changes.append(m.addVar(lb=-10, ub=10, vtype=GRB.CONTINUOUS))
        x.append(m.addVar(vtype=GRB.BINARY))
        m.addConstr(changes[i]-tolerance*x[i]<=0)
        m.addConstr(changes[i]+tolerance*x[i]>=0)

    for i in range(len(inputs)):
        input_vars.append(m.addVar(lb=-10, ub=10, vtype=GRB.CONTINUOUS))
        # m.addConstr(input_vars[i]+changes[i]>=inputs[i])
        # m.addConstr(input_vars[i]-changes[i]<=inputs[i])
        m.addConstr(input_vars[i]-changes[i]==inputs[i])
    
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    result = np.matmul(input_vars,w)+b

    tr = 3
    for i in range(len(result)):
        if outputs[i]<=0:
            m.addConstr(result[i]<=tr)
            # z = z+1
        else:
            m.addConstr(result[i]-outputs[i]<=tr)
            # p=p+1
    
    sumX = gp.quicksum(x)
    # max_sum = m.addVar(lb=0,ub=k,vtype=GRB.CONTINUOUS, name="max_sum")
    # m.addConstr(sumX-max_sum<=0)
    m.addConstr(sumX-k<=0)

    expr = gp.quicksum(changes)
    epsilon_max_2 = m.addVar(lb=0,ub=150,vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.addConstr(expr>=0)
    m.addConstr(expr-epsilon_max_2<=0)
    m.update()
    
    m.optimize()
    if m.Status == GRB.INFEASIBLE:
        print("Adversarial example not found.")
        return []
    # m.write("abc.lp")
    # print(sumX)
    modifications = []
    for i in range(len(changes)):
        modifications.append(float(changes[i].X))
    return modifications

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
        return 0, [], [], -1, -1, 0, 0, -1

    # return 0, [], [], -1, -1, 0, 0
    tempModel = predict(epsilon, 0, sat_in)
    
    num_layers = int(len(tempModel.get_weights())/2)
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[0]
    """
    Now, I have all the epsilons which are to be added to layer 0. 
    Left over task: Find delta such that input+delta can give the same effect as update model
    We want the outputs of hidden layer 1 to be equal to the values stored in neuron_values_1
    """
    originalModel = loadModel()
    true_output = originalModel.predict([sat_in])
    true_label = np.argmax(true_output)
    # print("True Label is: ", true_label)
    # print(true_output)
    k = 12
    change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
    
    # for i in range(len(change))
    if len(change)>0:
        for j in range(8):
            ad_inp2 = []
            for i in range(len(change)):
                ad_inp2.append(change[i]+sat_in[i])

            ad_output = originalModel.predict([ad_inp2])
            # print(ad_output)
            predicted_label = np.argmax(ad_output)
            # print(predicted_label)
            vals = get_neuron_values_actual(originalModel, ad_inp2, num_layers)
            ch = 0
            max_shift = 0
            for i in range(len(vals[0])):
                ch = ch + abs(vals[0][i]-neuron_values_1[i])
                if abs(vals[0][i]-neuron_values_1[i])>max_shift:
                    max_shift = abs(vals[0][i]-neuron_values_1[i])
            linf, sumDist = findMetric(sat_in, ad_inp2)
            
            if predicted_label!=true_label:
                L2_norm = np.linalg.norm(np.array(sat_in)-np.array(ad_inp2))
                print("L2-norm is: ", L2_norm, "L-inf norm is: ", linf, "Total change is: ", sumDist)
                print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
                print("This was:", k, "pixel attack.")
                # print("Original Input:")
                return 1, sat_in, ad_inp2, true_label, predicted_label, L2_norm, linf, k
            else:
                k = k*2
                print("Changing k to:", k)
                change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
    return 0, [], [], -1, -1, 0, 0, -1

def attack():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=0
    counter_inputs = [0]*10
    counter_outputs = [0]*10
    l2 = 0
    linfTotal = 0
    adversarial_count = 0
    model = loadModel()
    ks = []

    for i in range(count):
        print("###########################################################################################")
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        t= np.argmax(model.predict([sat_in]))
        print("True label is:", t)
        print()
        t1 = time()
        success, original, adversarial, true_label, predicted_label, L2_norm, linf, k = generateAdversarial(sat_in)
        if success==1 and counter_inputs[true_label]<30:
            counter_inputs[true_label] = counter_inputs[true_label] + 1
            counter_outputs[predicted_label] = counter_outputs[predicted_label] + 1
        if success==1:
            l2 = l2 + L2_norm
            linfTotal = linfTotal + linf
            adversarial_count = adversarial_count + 1
            ks.append(k)
            # break
        t2 = time()
        print("Time taken in this iteration:", (t2-t1), "seconds.")
        print("###########################################################################################")
        # break
        # if i==5:
        #     break
    
    print("Attack was successful on:", adversarial_count," images.")
    print(counter_inputs)
    print(counter_outputs)
    print("Average L-inf norm:", linfTotal/adversarial_count)
    print("Average L-2 norm:", l2/adversarial_count)
    print("Mean k value:",np.mean(ks))
    print("Median k value:",np.median(ks))


t1 = time()
attack()
t2 = time()
print("TIME TAKEN IN GENERATION OF ABOVE EXAMPLES: ", (t2-t1), "seconds.")