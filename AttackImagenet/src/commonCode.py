import sys
import copy
from csv import reader
from time import time, sleep
from label import labelling
from PielouMesaure import PielouMeaure
from extractNetwork import extractNetwork

sys.path.append( "/home/ritesh/Desktop/MTP2/Marabou/" )
sys.path.append( "./XAI/" )

import verif_property
from draw import *
import minExp
import helper
from importance import get_importance 
import numpy as np
import os
import gurobipy as gp
import slic
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
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# from memory_profiler import profile
"""
What this file does?
Find modification in intermediate layers and converts that modification into an adversarial input.
This file implements our algorithm as described in the paper.
"""

counter=0

# MODEL_PATH = '../Models/FMNIST/fashion_mnist2.h5'
# INP_DATA = '../data/FMNIST/inputs.csv'
# OUT_DATA = '../data/FMNIST/outputs.csv'

MODEL_PATH = '../Models/MNIST/mnist_1.h5'
INP_DATA = '../data/MNIST/inputs.csv'
OUT_DATA = '../data/MNIST/outputs.csv'

# INP_DATA = './failedAttack_fmnist2_inputs.csv'
# OUT_DATA = './failedAttack_fmnist2_outputs.csv'

no_exp = 0
no_sig=0
no_pair=0
no_ub=0

pred_dict = {i: 0 for i in range(10)}

def loadModel():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def getData():
    inputs = []
    outputs = []
    f1 = open(INP_DATA, 'r')
    f1_reader = reader(f1)
    stopAt = 500
    f2 = open(OUT_DATA, 'r')
    f2_reader = reader(f2)
    i=0
    for row in f1_reader:
        inp = [round(float(x), 5) for x in row]
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


def model_to_graph(model_filename, loaded=False):
    """
    Given the name of the .h5 model file, load it and convert it into a
    NetworkX graph. Returns this NetworkX.

    Return:
    NetworkX directed graph of the loaded .h5 model file.
    """
    # Load the TensorFlow model
    if not loaded:
        model = tf.keras.models.load_model(model_filename)
    else:
        model = model_filename

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
    print("True output: ", true_output)
    
    probabilities = tf.nn.softmax(sample_output)

    # largest = np.partition(probabilities, -1)[-1]
    # second_largest = np.partition(probabilities, -2)[-2]

    labels = labelling(originalModel, true_output, 0.05)
    epsilon, inp = getEpsilons(layer_to_change, sat_in, labels)
    tempModel = predict(epsilon, layer_to_change, sat_in)
    """
    Now we have modifications in the middle layer of the netwrok.
    Next, we will run a loop to divide the network and find modifications in lower half of the network.
    """

    o1 = extractNetwork()
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[layer_to_change]
    # print("NV: ", neuron_values_1)
    while layer_to_change>0:
        extractedNetwork = o1.extractModel(originalModel, layer_to_change+1)
        layer_to_change = int(layer_to_change/2)
        epsilon = find2(10, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases, labels)

        tempModel = predict(epsilon, layer_to_change, sat_in)
        phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        neuron_values_1 = phases[layer_to_change]
    return extractedNetwork, neuron_values_1,  epsilon 

def getInputBounds(phases, labels, layer_sizes, layer_to_change, threshold):
    input_lower_bound = [0] * layer_sizes[layer_to_change+1]
    input_upper_bound = [0] * layer_sizes[layer_to_change+1] 
    for i in range(len(phases)):
        if labels[layer_to_change][i] == 1:
            input_lower_bound[i] = phases[layer_to_change][i]
            input_upper_bound[i] = threshold
        elif labels[layer_to_change][i] == -1:
            input_lower_bound[i] = 0
            input_upper_bound[i] = phases[layer_to_change][i]
        else:
            input_lower_bound[i] = 0
            input_upper_bound[i] = threshold

    return input_lower_bound, input_upper_bound

def updateModel_XAI(sat_in):
    model = loadModel()
    num_layers = int(len(model.get_weights())/2)
    layer_to_change = int(num_layers/3)
    originalModel = model
    sample_output = model.predict(np.array([sat_in]))[0]
    true_output = np.argmax(sample_output)

    print(f"True Label: {true_output}")
    
    labels = labelling(originalModel, true_output, 0.05)

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


    verif_property.add_property(G, True, conj_lin_eqn)

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
    pos = []
    neg = []
    # for i in range(layer_sizes[layer_to_change+1]):
    #     input_features.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))
    for i in range(layer_sizes[layer_to_change+1]):
        if (input_layer_values[sorted_indices[i]] > 0):
            pos.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))
        else:
            neg.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))

    input_features.append(pos)
    input_features.append(neg)

    input_lower_bound, input_upper_bound = getInputBounds(phases, labels, layer_sizes, layer_to_change, 20)
    input_lower_bound, input_upper_bound = [0]*layer_sizes[layer_to_change+1], [20]*layer_sizes[layer_to_change+1]

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
    if(E.result_singletons):
        # print("Result: ", E.result_singletons)
        max_key = helper.findClassForImage(G_original, E.result_singletons[0])
        for node, value in E.result_singletons[0]:
            neuron_values_new[node] = value
        print("singleton")
        # print("Max key: ", max_key)
    elif(E.result_ub):
        max_key = helper.findClassForImage(G_original, E.result_ub[0])
        for node, value in E.result_ub[0]:
            neuron_values_new[node] = value
        print("ub")
        # print("Max key: ", max_key)
    elif(E.result_pairs):
        # print("Result: ", E.result_singletons)
        max_key = helper.findClassForImage(G_original, E.result_pairs[0])
        for node, value in E.result_pairs[0]:
            neuron_values_new[node] = value
        print("pair")
        # print("Max key: ", max_key)
    else:
        return -1, None, None



    # Now we have modifications in the middle layer of the netwrok.
    # Next, we will run a loop to divide the network and find modifications in lower half of the network.

    o1 = extractNetwork()


    # We don't need to calcuate neuron values as we will already get it from the sub-routine of
    # min explanation, E.result_singletons, E.result_pairs, E.result_ub etc
    # phases = get_neuron_values_actual(tempModel, sat_in, num_layers)

    # we want to get the neuron_value_1 from the min_exp sub-routine
    # neuron_values_1 = phases[layer_to_change] # use results from above to set the vlaues

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

        pos = []
        neg = []
        for i in range(layer_sizes[layer_to_change+1]):
            if (input_layer_values[sorted_indices[i]] > 0):
                pos.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))
            else:
                neg.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))

        input_features.append(pos)
        input_features.append(neg)

        input_lower_bound, input_upper_bound = getInputBounds(phases, labels, layer_sizes, layer_to_change, 20)
        input_lower_bound, input_upper_bound = [0]*layer_sizes[layer_to_change+1], [20]*layer_sizes[layer_to_change+1]

        output_values = neuron_values_new

        # output_values = {node : 0 for node in neuron_values_new.keys()}
        print("Output vals: ", output_values)
        E = minExp.XAI()
        try:
            ub_exp, lb_exp, pairs = E.explanation(G_new, input_features, input_lower_bound, input_upper_bound, output_values)
        except Exception as e:
            print("Error Occurred")
            return -1, None, None

        neuron_values_final = {}
        if(E.result_ub):
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            remove_till_layer(G_another, layer_to_change+1)
            # max_key = helper.findClassForImage(G_another, E.result_ub[0])
            for node, value in E.result_ub[0]:
                neuron_values_final[node] = value
        elif(E.result_singletons):
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            remove_till_layer(G_another, layer_to_change+1)
            # max_key = helper.findClassForImage(G_another, E.result_singletons[0])
            for (node, value) in E.result_singletons[0]:
                neuron_values_final[node] = value
        elif(E.result_pairs):
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            remove_till_layer(G_another, layer_to_change+1)
            # max_key = helper.findClassForImage(G_another, E.result_pairs[0])
            for (node, value) in E.result_pairs[0]:
                neuron_values_final[node] = value
        else:
            break

        print("Neuron value: ", neuron_values_final)
        if(neuron_values_final):
            neuron_values_new = neuron_values_final
        else:
            return -1, None, None

        # epsilon = find2(10, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases2, labels)
        # tempModel = predict(epsilon, layer_to_change, sat_in)
        # phases2 = get_neuron_values_actual(tempModel, sat_in, num_layers)
        # neuron_values_1 = phases2[layer_to_change]
    # exit(1)
    # return extractedNetwork, neuron_values_1,  epsilon 
    epsilon = None
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


def image_to_bundles(image_data, num_segments=50, comp=10, channel_axis=None):
    # Generate Segments 
    B = slic.Bundle()
    return B.generate_segments2(np.array(image_data), 28, 28, num_segments, comp, channel_axis)


def MarabouAttack(model, inputs, neuron_values, k=2):
    imp_neurons = getImportantNeurons(inputs)
    G, _ = model_to_graph(MODEL_PATH)
    remove_after_layer(G, 1)

    output_values = {}
    for index, value in neuron_values.items():
        # output_values[(1, index)] = value
        output_values[index] = value

    result, new_prediction, k = find_singleton_bundle2(G, imp_neurons, [0]*784, [1]*784, output_values)
    if result:
        adv_inp = []
        for (node, val) in result:
            adv_inp.append(val)
        return 1, adv_inp, new_prediction, k
    else:
        return 0, [], -1, -1



def generate_linear_eqn(num_classes, pred_class):
    # Add linear equations along with bias if any

    # linear_equations_list = [
    #     [1,-1,0,0,0,0,0,0,0,0,0],
    #     [1,0,-1,0,0,0,0,0,0,0,0],
    #     [1,0,0,-1,0,0,0,0,0,0,0],
    #     [1,0,0,0,-1,0,0,0,0,0,0],
    #     [1,0,0,0,0,-1,0,0,0,0,0],
    #     [1,0,0,0,0,0,-1,0,0,0,0],
    #     [1,0,0,0,0,0,0,-1,0,0,0],
    #     [1,0,0,0,0,0,0,0,-1,0,0],
    #     [1,0,0,0,0,0,0,0,0,-1,0]
    # ]

    equation_list = [0] * (num_classes+1)
    equation_list[pred_class] = 1 # this will be same for all equation
    
    linear_equations_list = [[i for i in equation_list] for _ in range(num_classes-1)] # generate all the equations

    count = 0
    for eqn in linear_equations_list: # This loop put -1 in for each equation at each class other than predicted class (for the logic, pred_class - other_class)
        if(count == pred_class):
            count += 1
        eqn[count] = -1
        count += 1

    return linear_equations_list

def sort_bundles(G, image_data, input_features_bundle):
    # Sorts bundles into most important to less important
    imp_neus = get_importance(G,image_data,False)
    img_dict = {}
    img_dict.update({(0, i): val for i, val in enumerate(image_data)})

    imp_neu_dict = {}
    imp_neu_dict.update({neu[1] : val for neu, val in imp_neus})
    bundle_imp = []
    for bundle in input_features_bundle:
        sum = 0
        for neuron in bundle:
            sum += imp_neu_dict[neuron[0][1]]
        bundle_imp.append(sum)

    bundle_imp = enumerate(bundle_imp)
    bundle_imp = sorted(bundle_imp, key = lambda x: x[1], reverse=True)
    # print(bundle_imp)
    inp_f_bundle_sorted = []
    for ind, val in bundle_imp:
        inp_f_bundle_sorted.append(input_features_bundle[ind])

    return inp_f_bundle_sorted


def find_singleton_bundle(G, input_features, input_lower_bound, input_upper_bound, second_largest_data, pred_class_data):
    # Find the single input features that are important 
    # important: removal causes a mis-classification
    E = minExp.XAI()
    # E.lower_conf = True
    # E.second_largest = second_largest_data[0]
    # E.conf_score = second_largest_data[1]
    # E.pred_class = pred_class_data[0]
    # E.pred_value = pred_class_data[1]
    E.input_lb = input_lower_bound
    E.input_ub = input_upper_bound


    orig_features = copy.deepcopy(input_features)
    # orig_features = set([feature for bundle in self.input_features for feature in bundle])
    for ip_f in input_features:
            orig_features.remove(ip_f)
            orig_features_list = set([feature for bundle in orig_features for feature in bundle])
            result =  E.verif_query(G, orig_features_list, ip_f)
            if result[0] == 'SAT':
                    # print("Singleton , ip",result[1])
                    # print("LENGTH OF SINGLETON: ", len(ip_f))
                    # print("Singleton in image: ", ip_f)
                    new_img = []
                    for (_,_), value in result[1]:
                        new_img.append(value)
                    model = loadModel()
                    pred_out = model.predict([new_img])[0]
                    # print("----------------IN_SING------------------------")
                    new_pred = np.argmax(pred_out)
                    print("New Prediction: ", new_pred)
                    print(pred_out)
                    pred_dict[np.argmax(pred_out)] += 1
                    # print("----------------IN_SING------------------------")
                    # singletons.add(tuple(ip_f))
                    # result_singletons.append(result[1])
                    # LB = LB+1
                    return result[1], new_pred, len(ip_f)
            else:
                # print("Not Singleton , ip",result[1],ip_f)
                pass
            orig_features.append(ip_f)
    return 0, -1, -1

def find_singleton_bundle2(G, input_features, input_lower_bound, input_upper_bound, output_values):
    # Find the single input features that are important 
    # important: removal causes a mis-classification
    E = minExp.XAI()
    E.lower_conf = False
    E.output_values = output_values
    E.input_lb = input_lower_bound
    E.input_ub = input_upper_bound


    orig_features = copy.deepcopy(input_features)
    # orig_features = set([feature for bundle in self.input_features for feature in bundle])
    for ip_f in input_features:
            orig_features.remove(ip_f)
            orig_features_list = set([feature for bundle in orig_features for feature in bundle])
            result =  E.verif_query(G, orig_features_list, ip_f)
            if result[0] == 'SAT':
                    # print("Singleton , ip",result[1])
                    # print("LENGTH OF SINGLETON: ", len(ip_f))
                    # print("Singleton in image: ", ip_f)
                    new_img = []
                    for (_,_), value in result[1]:
                        new_img.append(value)
                    model = loadModel()
                    pred_out = model.predict([new_img])[0]
                    # print("----------------IN_SING------------------------")
                    new_pred = np.argmax(pred_out)
                    print("New Prediction: ", new_pred)
                    # print(pred_out)
                    pred_dict[np.argmax(pred_out)] += 1
                    # print("----------------IN_SING------------------------")
                    # singletons.add(tuple(ip_f))
                    # result_singletons.append(result[1])
                    # LB = LB+1
                    return result[1], new_pred, len(ip_f)
            else:
                # print("Not Singleton , ip",result[1],ip_f)
                pass
            orig_features.append(ip_f)
    return 0, -1, -1


def second_largest_index(arr):
    largest_index = np.argmax(arr)
    largest_value = arr[largest_index]
    arr[largest_index] = -np.inf
    second_largest_index = np.argmax(arr)
    arr[largest_index] = largest_value
    
    return second_largest_index

def getImportantNeurons(sat_in, seg=50, comp=10, channel_axis=None):
    input_features_bundle = image_to_bundles(sat_in, 20, comp, channel_axis)

    # Use the model to find the predicted output/class of the image
    model = tf.keras.models.load_model(MODEL_PATH)
    pred_output = model.predict(np.array([sat_in]))[0]
    pred_class = np.argmax(pred_output)

    G, _ = model_to_graph(MODEL_PATH)

    # get linear equation property
    lin_eqn = generate_linear_eqn(10, pred_class)

    verif_property.add_property(G, True, lin_eqn)
    important_neurons = sort_bundles(G, sat_in, input_features_bundle)

    return important_neurons

def lowerConfidence(sat_in):
    adv_flag = False
    # Segment the input image and sort the bundles according to importance 
    input_features_bundle = image_to_bundles(sat_in, num_segments=50)

    # Use XAI technique to find singleton for minimal modification
    # Read the model and convert it to a graph for the verification purpose

    # Use the model to find the predicted output/class of the image
    model = tf.keras.models.load_model(MODEL_PATH)
    pred_output = model.predict(np.array([sat_in]))[0]
    # print("-------------------START---------------------")
    pred_class = np.argmax(pred_output)
    second_largest = second_largest_index(pred_output)

    pred_class_value = pred_output[pred_class]
    conf_score = pred_class_value - pred_output[second_largest]

    second_largest_position = second_largest
    if(pred_class < second_largest):
        second_largest_position = second_largest - 1

    print("Original Prediction: ", pred_class)
    # print("-------------------START---------------------")
    # print(pred_output)
    # print(pred_class)


    G, _ = model_to_graph(MODEL_PATH)
    G2, _ = model_to_graph(MODEL_PATH)
    # get linear equation property
    lin_eqn = generate_linear_eqn(10, pred_class)

    # Add the linear equation property to the model graph
    verif_property.add_property(G, True, lin_eqn)
    verif_property.add_property(G2, True, lin_eqn)
    important_neurons = sort_bundles(G2, sat_in, input_features_bundle)

    input_lower_bound = [0]*784
    input_upper_bound = [1]*784

    find_singleton_bundle(G, important_neurons, input_lower_bound, input_upper_bound, [second_largest_position, conf_score], [pred_class, pred_class_value])

    # Iteratively check which singleton reduces confidence,
    # either lowers the true class confidence or misclassifies in which case
    # we return adv_flag as true

def generateAdversarial(sat_in):
    try:
        extractedModel, neuron_values_1, epsilon = updateModel(sat_in)
    except Exception as e:
        print(f"Exception: {e}")
        print("UNSAT. Could not find a minimal modification by divide and conquer.")
        return 0, [], [], -1, -1, -1

   
    tempModel = predict(epsilon, 0, sat_in)
    
    num_layers = int(len(tempModel.get_weights())/2)
    phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
    neuron_values_1 = phases[0]
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

def generateAdversarial_XAI(sat_in):
    try:
        extractedModel, neuron_values_1, epsilon = updateModel_XAI(sat_in)
    except Exception as e:
        print(e)
        print("UNSAT. Could not find a minimal modification by divide and conquer.")
        return 0, [], [], -1, -1, -1

   

    flag = False
    """
    tempModel below is needed when we are changing the weights in the original attack based on minimal modification.

    Now we want to get the neurons value of the first hidden layer, so that we can skip below and get directly the
    input+delta such that we get the adversarial attack.

    """

    print("hereflag")
    print(neuron_values_1)
    if flag and neuron_values_1:
        originalModel = loadModel()
        true_output = originalModel.predict([sat_in])
        num_layers = int(len(originalModel.get_weights())/2)
        true_label = np.argmax(true_output)
        print("flag")

        success, adv_inp, predicted_label, k = MarabouAttack(extractedModel, sat_in, neuron_values_1, 2)
        print("Success", success)

        if success:
            print("Attack was successful. Label changed from ", true_label," to ", predicted_label)
            return 1, sat_in, adv_inp, true_label, predicted_label,  k
        else:
            return 0, [], [], -1, -1, -1
    elif not flag and neuron_values_1:

        temp = neuron_values_1
        neuron_values_1 = list(neuron_values_1.values())


        # tempModel = predict(epsilon, 0, sat_in)


        # 
        # num_layers = int(len(tempModel.get_weights())/2)
        # # print("num_layers: ", num_layers)
        # phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        # print("NV: ", neuron_values_1)
        # neuron_values_1 = phases[0]
        # print("NV: ", neuron_values_1)

        
        """
        Now, we have all the epsilons which are to be added to layer 0. 
        Left over task: Find delta such that input+delta can give the same effect as update model
        We want the outputs of hidden layer 1 to be equal to the values stored in neuron_values_1
        """
        originalModel = loadModel()
        true_output = originalModel.predict([sat_in])
        num_layers = int(len(originalModel.get_weights())/2)
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
    # print(getImportantNeurons(inputs[0]))
    # for i in range(10):
    #     lowerConfidence(inputs[i])
    # print(pred_dict)
    # exit(1)
    failed_adv = []
    for i in range(count):
        print("###########################################################################################")
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        print()
        t1 = time()
        success, original, adversarial, true_label, predicted_label, k = generateAdversarial_XAI(sat_in)

        if(success==0):
            failed_adv.append(i)

        if success==1 and counter_inputs[true_label]<30:
            counter_inputs[true_label] = counter_inputs[true_label] + 1
            counter_outputs[predicted_label] = counter_outputs[predicted_label] + 1
        if success==1:
            adversarial_count = adversarial_count + 1
            ks.append(k)
        t2 = time()
        print("Time taken in this iteration:", (t2-t1), "seconds.")
        print("###########################################################################################")    
    
    print("Failed Adv input indices: ", failed_adv)
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
