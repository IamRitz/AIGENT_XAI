import sys
import copy
from csv import reader
from time import time
from label import labelling
from PielouMesaure import PielouMeaure
from extractNetwork import extractNetwork

sys.path.append( "./Marabou/" )
sys.path.append( "./XAI/" )

import verif_property
from draw import *
import minExp
import helper
import numpy as np
import os
import gurobipy as gp
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
from findModificationsLayerK import find as find
from modificationDivided import find as find2
from gurobipy import GRB
from scipy import stats
import networkx as nx
# from memory_profiler import profile
"""
What this file does?
Find modification in intermediate layers and converts that modification into an adversarial input.
This file implements our algorithm as described in the paper.
"""

counter=0

MODEL_PATH = '../Models/mnist.h5'

no_exp = 0
no_sig=0
no_pair=0
no_ub=0

def loadModel():
    model = tf.keras.models.load_model(MODEL_PATH)
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
    i=0
    for row in f2_reader:
        out = [float(x) for x in row]
        outputs.append(out)
        i=i+1
        if i==stopAt:
            break
    return inputs, outputs, len(inputs)


def model_to_graph(model_filename):
    """
    Given the name of the .h5 model file, load it and convert it into a
    NetworkX graph. Returns this NetworkX.

    Return:
    NetworkX directed graph of the loaded .h5 model file.
    """
    # Load the TensorFlow model
    model = tf.keras.models.load_model(model_filename)

    # Initialize weights and biases dictionary
    weights = {}
    biases = {}

    # Extract weights and biases from each layer
    for layer in model.layers:
        layer_weights_biases = layer.get_weights()
        weights[layer.name] = layer_weights_biases[0]
        biases[layer.name] = layer_weights_biases[1]

    keys = list(weights.keys())

    # Size of input layer
    inp_size = len(weights[keys[0]])

    # Size of layers other than input layer in order
    layer_sizes = [len(biases[key]) for key in keys] 

    # Initialize the networkx graph
    G = nx.DiGraph()

    # Construct the input layer
    for i in range(inp_size):
        G.add_node((0, i))

    # Construct the rest of the layers
    for layer_num in range(0, len(keys)):
        sizes = layer_sizes[layer_num]
        for j in range(sizes):
            G.add_node((layer_num+1, j))

    # Add edges for weights
    for layer_id, layer_name in enumerate(keys):
        layer_weights = weights[layer_name]
        for i in range(len(layer_weights)):
            for j in range(layer_sizes[layer_id]):
                G.add_edges_from([((layer_id, i), (layer_id + 1, j), {"weight": layer_weights[i][j]})])

    for layer_id, layer_name in enumerate(keys):
        layer_biases = biases[layer_name]
        for i in range(len(layer_biases)):
            G.nodes[(layer_id+1, i)]["bias"] = layer_biases[i]

    return G, layer_sizes


def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 1
        for layer in loaded_model.layers:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            result = np.matmul(input,w)+b
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        return neurons

def getEpsilons(layer_to_change, inp, labels):
    model = loadModel()
    num_inputs = len(inp)
    sample_output = model.predict(np.array([inp]))[0]
    true_label = np.argmax(sample_output)
    num_outputs = len(sample_output)
    expected_label = sample_output.argsort()[-2]
    all_epsilons = find(10, model, inp, expected_label, num_inputs, num_outputs, 1, layer_to_change, labels)
    
    return all_epsilons, inp

def predict(epsilon, layer_to_change, sat_in):
    model = loadModel()
    weights = model.get_weights()

    weights[2*layer_to_change] = weights[2*layer_to_change]+ np.array(epsilon[0])

    model.set_weights(weights)
    model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    prediction = model.predict([sat_in])
    print("Prediction: ",np.argmax(prediction[0]))
    return model


def remove_till_layer(G, layer_num):
    nodes_to_remove = [node for node in G.nodes if node[0]<layer_num]
    G.remove_nodes_from(nodes_to_remove)
    # Remove isolated nodes if any
    G.remove_nodes_from(list(nx.isolates(G)))

def remove_after_layer(G, layer_num):
    nodes_to_remove = [node for node in G.nodes if node[0]>layer_num]
    G.remove_nodes_from(nodes_to_remove)
    # Remove isolated nodes if any
    G.remove_nodes_from(list(nx.isolates(G)))


def updateModel(sat_in):
    model = loadModel()
    num_layers = int(len(model.get_weights())/2)
    layer_to_change = int(num_layers/2)
    originalModel = model
    sample_output = model.predict(np.array([sat_in]))[0]
    true_output = np.argmax(sample_output)

    print(f"True Label: {true_output}")
    
    # Convert the original model to a networkx graph
    G, layer_sizes = model_to_graph(MODEL_PATH)
    layer_sizes.insert(0, 784)
    
    G_prime = copy.deepcopy(G)
    G_original = copy.deepcopy(G)
    G_another = copy.deepcopy(G)
    
    # Creating conjunction of linear equations
    lin_eqn = [0] * (len(sample_output) + 1)
    lin_eqn[true_output] = 1
    conj_lin_eqn = [[i for i in lin_eqn] for _ in range(len(sample_output)-1)]

    count = 0
    for eqn in conj_lin_eqn:
        if count == true_output:
            count += 1
        eqn[count] = -1
        count += 1


    verif_property.add_property(G, False, conj_lin_eqn)

    G1 = copy.deepcopy(G)
    remove_till_layer(G1, layer_to_change+1)
    remove_till_layer(G_original, layer_to_change+1)

    # We don't need labelling now for min explanation
    labels = labelling(originalModel, true_output, 0.05)

    # We don't want epsilons, rather we would like to get the neurons value
    # at the neuron layer next to the edge layer we want to change.

    phases = get_neuron_values_actual(originalModel, sat_in, num_layers)


    input_layer_values = phases[layer_to_change]
    input_with_indices = [(val, idx) for idx, val in enumerate(input_layer_values)]
    sorted_indices = sorted(input_with_indices, reverse=True)
    sorted_indices = [idx for val, idx in sorted_indices]
    input_features = []
    for i in range(layer_sizes[layer_to_change+1]):
        input_features.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))

    input_lower_bound = [0] * layer_sizes[layer_to_change+1]
    input_upper_bound = [10] * layer_sizes[layer_to_change+1] 
    output_values = None
    E = minExp.XAI()
    ub_exp, lb_exp, pairs = E.explanation(G1, input_features, input_lower_bound, input_upper_bound, output_values)

    # print(f"Upper Bound Exp: {ub_exp}")
    # print(f"Lower Bound Exp: {lb_exp}")
    # print(f"Pairs: {pairs}")

    global no_exp
    global no_sig
    global no_pair
    global no_ub

    if(not E.result_singletons and not E.result_pairs and not E.result_ub):
        no_exp += 1
    if(not E.result_singletons):
        no_sig += 1
    if(not E.result_pairs):
        no_pair += 1
    if(not E.result_ub):
        no_ub += 1

    # print("Result: ", E.result_ub)

    neuron_values_new = {}
    if(E.result_ub):
        max_key = helper.findClassForImage(G_original, E.result_ub[0])
        for node, value in E.result_ub[0]:
            neuron_values_new[node] = value
        print("Max key: ", max_key)
    elif(E.result_singletons):
        # print("Result: ", E.result_singletons)
        max_key = helper.findClassForImage(G_original, E.result_singletons[0])
        for node, value in E.result_singletons[0]:
            neuron_values_new[node] = value
        print("Max key: ", max_key)
    elif(E.result_pairs):
        # print("Result: ", E.result_singletons)
        max_key = helper.findClassForImage(G_original, E.result_pairs[0])
        for node, value in E.result_pairs[0]:
            neuron_values_new[node] = value
        print("Max key: ", max_key)


    # exit(1)

    epsilon, inp = getEpsilons(layer_to_change, sat_in, labels)
    tempModel = predict(epsilon, layer_to_change, sat_in)

    # Now we have modifications in the middle layer of the netwrok.
    # Next, we will run a loop to divide the network and find modifications in lower half of the network.

    o1 = extractNetwork()


    # We don't need to calcuate neuron values as we will already get it from the sub-routine of
    # min explanation, E.result_singletons, E.result_pairs, E.result_ub etc
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)

    # we want to get the neuron_value_1 from the min_exp sub-routine
    neuron_values_1 = phases[layer_to_change] # use results from above to set the vlaues

    neuron_values_final = {}
    while layer_to_change>0:
        extractedNetwork = o1.extractModel(originalModel, layer_to_change+1)
        last_layer_num = layer_to_change+1
        G_new = copy.deepcopy(G_prime)
        remove_after_layer(G_new, layer_to_change+1)
        layer_to_change = int(layer_to_change/2)
        G_prime = copy.deepcopy(G_new)
        remove_till_layer(G_new, layer_to_change+1)

        # for i in range(layer_sizes[last_layer_num]):
        #     G_new.nodes[(last_layer_num, i)]['bias'] = 0
        
        # Again call the explanation sub_routine
        phases2 = get_neuron_values_actual(originalModel, sat_in, num_layers)

        input_layer_values = phases[layer_to_change]
        # input_features = []
        # print("Input: ", input_layer_values)
        # for i in range(layer_sizes[layer_to_change+1]):
        #     input_features.append(((layer_to_change+1, i), input_layer_values[i]))

        input_with_indices = [(val, idx) for idx, val in enumerate(input_layer_values)]
        sorted_indices = sorted(input_with_indices, reverse=True)
        sorted_indices = [idx for val, idx in sorted_indices]
        input_features = []
        for i in range(layer_sizes[layer_to_change+1]):
            input_features.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))


        input_lower_bound = [0] * layer_sizes[layer_to_change+1]
        input_upper_bound = [10] * layer_sizes[layer_to_change+1] 
        output_values = neuron_values_new
        # output_values = {node : 0 for node in neuron_values_new.keys()}
        E = minExp.XAI()
        ub_exp, lb_exp, pairs = E.explanation(G_new, input_features, input_lower_bound, input_upper_bound, output_values)
        print("UB: ", ub_exp)
        print("LB: ", lb_exp)
        print("Pairs: ", pairs)

        if(E.result_ub):
            # print("Result: ", E.result_singletons)
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            remove_till_layer(G_another, layer_to_change+1)
            max_key = helper.findClassForImage(G_another, E.result_ub[0])
            for node, value in E.result_ub[0]:
                neuron_values_final[node[1]] = value
            # print("Max key: ", max_key)

        epsilon = find2(10, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases2, labels)
        tempModel = predict(epsilon, layer_to_change, sat_in)
        phases2 = get_neuron_values_actual(tempModel, sat_in, num_layers)
        neuron_values_1 = phases2[layer_to_change]
    # exit(1)
    # return extractedNetwork, neuron_values_1,  epsilon 
    # epsilon = None
    return extractedNetwork, neuron_values_final,  epsilon 

def GurobiAttack(inputs, model, outputs, k):
    tolerance = 1
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    x = []
    input_vars = []
    changes = []

    for i in range(len(inputs)):
        changes.append(m.addVar(lb=-10, ub=10, vtype=GRB.CONTINUOUS))
        x.append(m.addVar(vtype=GRB.BINARY))
        m.addConstr(changes[i]-tolerance*x[i]<=0)
        m.addConstr(changes[i]+tolerance*x[i]>=0)

    for i in range(len(inputs)):
        input_vars.append(m.addVar(lb=-10, ub=10, vtype=GRB.CONTINUOUS))
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

    sumX = gp.quicksum(x)
    m.addConstr(sumX<=k)

    expr = gp.quicksum(changes)
    epsilon_max_2 = m.addVar(lb=0,ub=150,vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.addConstr(expr>=0)
    m.addConstr(expr-epsilon_max_2<=0)
    m.update()
    
    m.optimize()
    if m.Status == GRB.INFEASIBLE:
        print("Adversarial example not found.")
        return []
    modifications = []
    for i in range(len(changes)):
        modifications.append(float(changes[i].X))
    return modifications

def generateAdversarial(sat_in):
    try:
        extractedModel, neuron_values_1, epsilon = updateModel(sat_in)
    except Exception as e:
        print(e)
        print("UNSAT. Could not find a minimal modification by divide and conquer.")
        return 0, [], [], -1, -1, -1

   

    flag = True
    """
    tempModel below is needed when we are changing the weights in the original attack based on minimal modification.

    Now we want to get the neurons value of the first hidden layer, so that we can skip below and get directly the
    input+delta such that we get the adversarial attack.

    """

    if flag and neuron_values_1:

        tempModel = predict(epsilon, 0, sat_in)
        # 
        num_layers = int(len(tempModel.get_weights())/2)
        # # print("num_layers: ", num_layers)
        # phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        print("NV: ", neuron_values_1)
        # neuron_values_1 = phases[0]
        # print("NV: ", neuron_values_1)

        
        """
        Now, we have all the epsilons which are to be added to layer 0. 
        Left over task: Find delta such that input+delta can give the same effect as update model
        We want the outputs of hidden layer 1 to be equal to the values stored in neuron_values_1
        """
        originalModel = loadModel()
        true_output = originalModel.predict([sat_in])
        true_label = np.argmax(true_output)
        k = 3
        change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
        
        if len(change)>0:
            for j in range(18):
                ad_inp2 = []

                for i in range(len(change)):
                    ad_inp2.append(change[i]+sat_in[i])

                ad_output = originalModel.predict([ad_inp2])
                predicted_label = np.argmax(ad_output)
                vals = get_neuron_values_actual(originalModel, ad_inp2, num_layers)
                ch = 0
                max_shift = 0
                for i in range(len(vals[0])):
                    ch = ch + abs(vals[0][i]-neuron_values_1[i])
                    if abs(vals[0][i]-neuron_values_1[i])>max_shift:
                        max_shift = abs(vals[0][i]-neuron_values_1[i])
                if predicted_label!=true_label:
                    print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
                    print("This was:", k,"pixel attack.")
                    return 1, sat_in, ad_inp2, true_label, predicted_label,  k
                else:
                    k = k*2
                    change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
    return 0, [], [], -1, -1, -1

# @profile
def generate():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=0
    counter_inputs = [0]*10
    counter_outputs = [0]*10
    adversarial_count = 0
    ks = []

    for i in range(count):
        print("###########################################################################################")
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        print()
        t1 = time()
        success, original, adversarial, true_label, predicted_label, k = generateAdversarial(sat_in)
        if success==1 and counter_inputs[true_label]<30:
            counter_inputs[true_label] = counter_inputs[true_label] + 1
            counter_outputs[predicted_label] = counter_outputs[predicted_label] + 1
        if success==1:
            adversarial_count = adversarial_count + 1
            ks.append(k)
        t2 = time()
        print("Time taken in this iteration:", (t2-t1), "seconds.")
        print("###########################################################################################")    
    
    print("Attack was successful on:", adversarial_count," images.")
    # print(counter_inputs)
    # print(counter_outputs)
    print("Number of pixels modified(Mean):",np.mean(ks))
    print("Number of pixels modified(Median):",np.median(ks))
    print("Number of pixels modified(Mode):",stats.mode(ks))
    pm = PielouMeaure(counter_outputs, len(counter_outputs))
    print("Pielou Measure is:", pm)
    return count

    # global no_exp
    # global no_sig
    # global no_pair
    # global no_ub
    # return (no_exp, no_sig, no_pair, no_ub)
