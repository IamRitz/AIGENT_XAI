import sys
from math import ceil
import copy
from PielouMesaure import PielouMeaure
from extractNetwork import extractNetwork

sys.path.append( "/home/ritesh/Desktop/MTP2/Marabou/" )
sys.path.append( "./XAI/" )

import verif_property
from draw import *
import minExp
import helper
from importance import get_importance 
from importanceLime import limeExplanation
import slic
import networkx as nx

from csv import reader
from time import time
from extractNetwork import extractNetwork
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
from PielouMesaure import PielouMeaure
from label import labelling
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

"""
What this file does?
Find modification in intermediate layers and converts that modification into an adversarial input.
This file implements our algorithm as described in the paper.
"""

counter=0
#
MODEL_PATH = "../Models/CIFAR5/cifar_64.h5"
INP_PATH = "../data/CIFAR5/inputs_64.csv"
OUT_PATH = "../data/CIFAR5/outputs_64.csv"

# MODEL_PATH = "../Models/CIFAR10/cifar.h5"
# INP_PATH = "../data/CIFAR10/inputs10.csv"
# OUT_PATH = "../data/CIFAR10/outputs.csv"

# MODEL_PATH = "./cifar_ext.h5"
# INP_PATH = "./inputs_new.csv"
# OUT_PATH = "./outputs_new.csv"

def loadModel():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def getData():
    inputs = []
    outputs = []
    f1 = open(INP_PATH, 'r')
    f1_reader = reader(f1)
    stopAt = 500
    f2 = open(OUT_PATH, 'r')
    f2_reader = reader(f2)
    i=0
    # adv = [16, 73, 290, 473, 530, 821]
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

def get_layer_value(loaded_model, input, num_layers):
        neurons = []
        l = 1
        for layer in loaded_model.layers:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            result = np.matmul(input,w)+b
            neurons = [max(0, r) for r in result]
            if l == num_layers:
                break
            l = l + 1
        return neurons

def getEpsilons(layer_to_change, inp,labels):
    model = loadModel()
    num_inputs = len(inp)
    sample_output = model.predict(np.array([inp]))[0]
    true_label = np.argmax(sample_output)
    num_outputs = len(sample_output)
    expected_label = sample_output.argsort()[-2]
    print("Attack label is:", expected_label)
    all_epsilons = find(10, model, inp, true_label, num_inputs, num_outputs, 1, layer_to_change, labels)
    
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

def updateModel(sat_in):
    model = loadModel()
    num_layers = int(len(model.get_weights())/2)
    weights = model.get_weights()[0]
    weights = model.get_weights()[1]
    layer_to_change = int(num_layers/2)
    originalModel = model
    sample_output = model.predict(np.array([sat_in]))[0]
    true_output = np.argmax(sample_output)
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
    
    while layer_to_change>0:
        extractedNetwork = o1.extractModel(originalModel, layer_to_change+1)
        layer_to_change = int(layer_to_change/2)
        epsilon = find2(100, extractedNetwork, inp, neuron_values_1, 1, layer_to_change, 0, phases, labels)

        tempModel = predict(epsilon, layer_to_change, sat_in)
        phases = get_neuron_values_actual(tempModel, sat_in, num_layers)
        neuron_values_1 = phases[layer_to_change]
    return extractedNetwork, neuron_values_1,  epsilon 

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
    layer_to_change = int(num_layers/2)
    originalModel = model
    sample_output = model.predict(np.array([sat_in]))[0]
    true_output = np.argmax(sample_output)

    print(f"True Label: {true_output}")
    
    labels = labelling(originalModel, true_output, 0.05)

    # Convert the original model to a networkx graph
    G, layer_sizes = model_to_graph(MODEL_PATH)
    layer_sizes.insert(0, len(sat_in)) 
    
    # G_prime = copy.deepcopy(G)
    G_prime, _ = model_to_graph(MODEL_PATH)
    
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

    # G1 = copy.deepcopy(G)
    remove_till_layer(G, layer_to_change+1)

    phases = get_neuron_values_actual(originalModel, sat_in, num_layers)

    input_layer_values = phases[layer_to_change]
    input_with_indices = [(val, idx) for idx, val in enumerate(input_layer_values)]
    sorted_indices = sorted(input_with_indices, reverse=True)
    sorted_indices = [idx for val, idx in sorted_indices]
    input_features = []

    # pos = []
    # neg = []
    # for i in range(layer_sizes[layer_to_change+1]):
    #     if (input_layer_values[sorted_indices[i]] > 0):
    #         pos.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))
    #     else:
    #         neg.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))

    # input_features.append(pos)
    # input_features.append(neg)

    # divide the sorted features into 3 parts
    # top, medium and bottom
    # each containing 33% of the features

    curr_layer_size = layer_sizes[layer_to_change+1]
    top = int(curr_layer_size/4)
    medium = int(curr_layer_size/4)
    bottom = int(curr_layer_size/4)
    lower_bottom = curr_layer_size - top - medium

    sorted_features = [((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]) for i in range(curr_layer_size)]

    input_features.append(sorted_features[:top])
    input_features.append(sorted_features[top:top+medium])
    input_features.append(sorted_features[top+medium:top+medium+lower_bottom])
    input_features.append(sorted_features[top+medium+lower_bottom:])

    # for i in range(layer_sizes[layer_to_change+1]):
    #     input_features.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))

    # input_lower_bound, input_upper_bound = getInputBounds(phases, labels, layer_sizes, layer_to_change, 20)
    input_lower_bound, input_upper_bound = [0]*layer_sizes[layer_to_change+1], [20]*layer_sizes[layer_to_change+1]


    output_values = None
    E = minExp.XAI()
    ub_exp, lb_exp, pairs = E.explanation(G, input_features, input_lower_bound, input_upper_bound, output_values)

    neuron_values_new = {}
    if(E.result_ub):
        print("UB Found")
        for node, value in E.result_ub[0]:
            neuron_values_new[node] = round(value, 6)
    elif(E.result_singletons):
        print("Singleton Found")
        for node, value in E.result_singletons[0]:
            neuron_values_new[node] = round(value, 6)
    elif(E.result_pairs):
        print("Pairs Found")
        for node, value in E.result_pairs[0]:
            neuron_values_new[node] = round(value, 6)
    else:
        print("No result found.")
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
        # G_new = copy.deepcopy(G_prime)
        G_new, _ = model_to_graph(extractedNetwork, True)
        remove_after_layer(G_new, layer_to_change+1)
        layer_to_change = int(layer_to_change/2)
        # G_prime = copy.deepcopy(G_new)
        remove_till_layer(G_new, layer_to_change+1)

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

        curr_layer_size = layer_sizes[layer_to_change+1]
        top = int(curr_layer_size/4)
        medium = int(curr_layer_size/4)
        bottom = int(curr_layer_size/4)
        lower_bottom = curr_layer_size - top - medium

        sorted_features = [((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]) for i in range(curr_layer_size)]

        input_features.append(sorted_features[:top])
        input_features.append(sorted_features[top:top+medium])
        input_features.append(sorted_features[top+medium:top+medium+lower_bottom])
        input_features.append(sorted_features[top+medium+lower_bottom:])


        # for i in range(layer_sizes[layer_to_change+1]):
        #     input_features.append(((layer_to_change+1, sorted_indices[i]), input_layer_values[sorted_indices[i]]))

        # input_lower_bound, input_upper_bound = getInputBounds(phases, labels, layer_sizes, layer_to_change, 20)
        input_lower_bound, input_upper_bound = [0]*layer_sizes[layer_to_change+1], [20]*layer_sizes[layer_to_change+1]

        output_values = neuron_values_new

        # output_values = {node : 0 for node in neuron_values_new.keys()}
        # print("Output vals: ", output_values)
        E = minExp.XAI()
        try:
            ub_exp, lb_exp, pairs = E.explanation(G_new, input_features, input_lower_bound, input_upper_bound, output_values)
        except Exception as e:
            print("Error Occurred")
            return -1, None, None

        neuron_values_final = {}
        if(E.result_ub):
            print("UB Found")
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            for node, value in E.result_ub[0]:
                neuron_values_final[node] = round(value, 6)
        elif(E.result_singletons):
            print("Singleton Found")
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            for (node, value) in E.result_singletons[0]:
                neuron_values_final[node] = round(value, 6) 
        elif(E.result_pairs):
            print("Pairs Found")
            neuron_values_new = [0] * layer_sizes[layer_to_change+1]
            for (node, value) in E.result_pairs[0]:
                neuron_values_final[node] = round(value, 6) 
        else:
            print("None found")
            break

        if(neuron_values_final):
            neuron_values_new = neuron_values_final
        else:
            return -1, None, None

    epsilon = None
    return extractedNetwork, neuron_values_final,  epsilon 


def image_to_bundles(image_data, num_segments=50, comp=10, channel_axis=None):
    # Generate Segments 
    B = slic.Bundle()
    return B.generate_segments2(np.array(image_data), 32, 32, num_segments, comp, channel_axis)


def MarabouAttack(model, inputs, neuron_values, true_label, k=2):
    # G, _ = model_to_graph(MODEL_PATH)
    # remove_after_layer(G, 1)
    G, _ = model_to_graph(model, True)
    remove_after_layer(G, 1)

    output_values = {}
    for index, value in neuron_values.items():
        output_values[index] = value

    # imp_neurons = getImportantNeurons(inputs, seg)
    imp_neurons = limeExplanation(None, inputs, MODEL_PATH, True)
    result, new_prediction, k = find_singleton_bundle2(G, imp_neurons, [0]*len(inputs), [1]*len(inputs), output_values, true_label)
    if result:
        adv_inp = []
        for (node, val) in result:
            adv_inp.append(val)
        return 1, adv_inp, new_prediction, k

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


def find_singleton_bundle(model, G, input_features, input_lower_bound, input_upper_bound, true_label):
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
                    pred_out = model.predict([new_img])[0]
                    new_pred = np.argmax(pred_out)
                    print("New Prediction: ", new_pred)

                    if true_label != new_pred:
                        return 1, result[1], new_pred, len(ip_f)
            else:
                print("Not Singleton: ", ip_f)

            orig_features.append(ip_f)
    return 0, [], -1, -1


def find_singleton_bundle2(G, input_features, input_lower_bound, input_upper_bound, output_values, true_label):
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
                    new_img = []
                    for (_,_), value in result[1]:
                        new_img.append(value)
                    model = loadModel()
                    pred_out = model.predict([new_img])[0]

                    new_pred = np.argmax(pred_out)
                    print("New Prediction: ", new_pred)

                    if true_label != new_pred:
                        return result[1], new_pred, len(ip_f)
            else:
                print("Not Singleton")
                pass
            orig_features.append(ip_f)
    return 0, -1, -1

def getImportantNeurons(sat_in, seg=50, comp=10, channel_axis=None): # TODO fix it according to img type
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
    input_features_bundle = image_to_bundles(sat_in, num_segments=50, channel_axis=-1)

    # Use XAI technique to find singleton for minimal modification
    # Read the model and convert it to a graph for the verification purpose

    # Use the model to find the predicted output/class of the image
    model = tf.keras.models.load_model(MODEL_PATH)
    pred_output = model.predict(np.array([sat_in]))[0]
    # print("-------------------START---------------------")
    pred_class = np.argmax(pred_output)
    second_largest = np.argsort(pred_output)[-2]

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
    # important_neurons = sort_bundles(G2, sat_in, input_features_bundle)
    important_neurons = limeExplanation(model, sat_in)

    input_lower_bound = [0]*len(sat_in)
    input_upper_bound = [1]*len(sat_in)

    success, adv_inp, new_pred, k = find_singleton_bundle(model, G, important_neurons, input_lower_bound, input_upper_bound, pred_class)

    if success and new_pred != pred_class:
        adv_flag = True
        print("Attack was successful. Label changed from ",pred_class," to ",new_pred)
        print("This was:", k, "pixel attack.")
        adv_inp = [val for (node, val) in adv_inp]
        return 1, sat_in, adv_inp, pred_class, new_pred, k
    else:
        print("Attack was unsuccessful.")
        return 0, [], [], -1, -1, -1

def FindCutoff(inputs, k):
    w = []
    w = inputs.copy()
    w.sort()
    mark = 0.1
    index = ceil(mark*len(w))
    # index = 500
    index = index if index>k else k
    heuristic = w[len(w)-index]
    return heuristic, index


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

    if flag and neuron_values_1:
        originalModel = loadModel()
        true_output = originalModel.predict([sat_in])
        num_layers = int(len(originalModel.get_weights())/2)
        true_label = np.argmax(true_output)

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

        """
        Now, we have all the epsilons which are to be added to layer 0. 
        Left over task: Find delta such that input+delta can give the same effect as update model
        We want the outputs of hidden layer 1 to be equal to the values stored in neuron_values_1
        """
        originalModel = loadModel()
        true_output = originalModel.predict([sat_in])
        num_layers = int(len(originalModel.get_weights())/2)
        true_label = np.argmax(true_output)

        k = 80
        change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)

        if len(change)>0:
            for j in range(6):
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

                if predicted_label!=true_label:
                    print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
                    print("This was:", k, "pixel attack.")
                    # print("Original Input:")
                    return 1, sat_in, ad_inp2, true_label, predicted_label, k
                else:
                    k = k*2
                    print("Changing k to:", k)
                    change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
        return 0, [], [], -1, -1, -1

def GurobiAttack(inputs, model, outputs, k):
    print("Launching attack with Gurobi.")
    tolerance = 10
    """
    Change the tolerance to increase/decrease the L-inf norm.
    """
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    x = []
    cutOff, limit = FindCutoff(inputs, k)
    input_vars = []
    changes = []
    v=0

    for i in range(len(inputs)):
        if inputs[i] >= cutOff and v <= limit:
            v += 1
            changes.append(m.addVar(lb=-tolerance, ub=tolerance, vtype=GRB.CONTINUOUS))
            x.append(m.addVar(vtype=GRB.BINARY))
            m.addConstr(changes[i]-tolerance*x[i]<=0)
            m.addConstr(changes[i]+tolerance*x[i]>=0)
        else:
            changes.append(m.addVar(lb=0, ub=0, vtype=GRB.CONTINUOUS))
            x.append(m.addVar(lb=0, ub=0, vtype=GRB.BINARY))
            m.addConstr(changes[i]==0)

    for i in range(len(inputs)):
        input_vars.append(m.addVar(lb=-tolerance, ub=tolerance, vtype=GRB.CONTINUOUS))
        m.addConstr(input_vars[i]-changes[i]==inputs[i])
    
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    result = np.matmul(input_vars,w)+b
    # print("Number of Changes: ", v)
    tr = 5
    """
    Increasing/decreasing the parameter tr can relax/restrict the solver more and more/less adversarial images can be generated.
    """
    for i in range(len(result)):
        if outputs[i]<=0:
            m.addConstr(result[i]<=tr)
        else:
            m.addConstr(result[i]-outputs[i]<=tr)
    
    sumX = gp.quicksum(x)
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
    modifications = []
    for i in range(len(changes)):
        modifications.append(float(changes[i].X))
    return modifications

def generateAdversarial(sat_in):
    try:
        extractedModel, neuron_values_1, epsilon = updateModel(sat_in)
    except:
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

    k = 12
    change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
    
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
            
            if predicted_label!=true_label:
                print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
                print("This was:", k, "pixel attack.")
                # print("Original Input:")
                return 1, sat_in, ad_inp2, true_label, predicted_label, k
            else:
                k = k*2
                print("Changing k to:", k)
                change = GurobiAttack(sat_in, extractedModel, neuron_values_1, k)
    return 0, [], [], -1, -1, -1

def attack():
    inputs, outputs, count = getData()
    print("Number of inputs in consideration: ",len(inputs))
    i=0
    counter_inputs = [0]*10
    counter_outputs = [0]*10
    adversarial_count = 0
    model = loadModel()
    ks = []

    failed_adv = []

    for i in range(count):
        print("###########################################################################################")
        print("Launching attack on input:", i)
        sat_in = inputs[i]
        t= np.argmax(model.predict([sat_in]))
        print("True label is:", t)
        print()
        t1 = time()
        success, original, adversarial, true_label, predicted_label, k = generateAdversarial_XAI(sat_in)
        # success, original, adversarial, true_label, predicted_label, k = generateAdversarial(sat_in)
        # success, original, adversarial, true_label, predicted_label, k = lowerConfidence(sat_in)

        if(success==0):
            failed_adv.append(i)

        if success==1 and counter_inputs[true_label]<30:
            counter_inputs[true_label] = counter_inputs[true_label] + 1
            counter_outputs[predicted_label] = counter_outputs[predicted_label] + 1
        if success==1:
            adversarial_count = adversarial_count + 1
            ks.append(k)
        t2 = time()
        print(failed_adv)
        print("Time taken in this iteration:", (t2-t1), "seconds.")
        print("###########################################################################################")
    
    print("Failed Adversarial Indices: ", failed_adv)
    print("Attack was successful on:", adversarial_count," images.")
    print(counter_inputs)
    print(counter_outputs)
    print("Mean k value:",np.mean(ks))
    print("Median k value:",np.median(ks))
    print("Mode k value:",stats.mode(ks))
    pm = PielouMeaure(counter_outputs, len(counter_outputs))
    print("Pielou Measure is:", pm)

    return count