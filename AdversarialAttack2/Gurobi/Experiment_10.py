from cProfile import label
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

    expected_label = 4
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
        add(m, input_vars[i]!=0)
        add(m, z3.And(delta[i]>=0, delta[i]<=delta_max))
    
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    out = w.T @ input_vars + b
    # print(out)
    layer_output = ReLU(out)
    tolerance = 1

    for i in range(len(outputs)):
        if outputs[i]>0:
            add(m, out[i]<=outputs[i]+tolerance)
            add(m, out[i]>=outputs[i]-tolerance)
        else:
            add(m, out[i]<=tolerance)
        # add(m, layer_output[i]==outputs[i])
   
    solution = m.check()
    print(solution)
    print(inputs)
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
            print(x[0], val)
            if "input" in str(x[0]):
                ad_inp.append(val)

    else:
        print(m.unsat_core())
    
    print(ad_inp)
    return ad_inp

def GurobiAttack(inputs, model, outputs):
    print("Launching attack with Gurobi.")
    tolerance = 10
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    changes = []
    input_vars = []
    max_change = m.addVar(lb = 0, ub = tolerance, vtype=GRB.CONTINUOUS, name="max_change")

    changes = []

    for i in range(len(inputs)):
        changes.append(m.addVar(vtype=GRB.CONTINUOUS))
        m.addConstr(changes[i]<=tolerance)
        m.addConstr(changes[i]>=-tolerance)

    for i in range(len(inputs)):
        input_vars.append(m.addVar(lb=-10, ub=10, vtype=GRB.CONTINUOUS))
        # m.addConstr(input_vars[i]+changes[i]>=inputs[i])
        # m.addConstr(input_vars[i]-changes[i]<=inputs[i])
        m.addConstr(input_vars[i]-changes[i]==inputs[i])
    
    weights = model.get_weights()
    w = weights[0]
    b = weights[1]
    result = np.matmul(input_vars,w)+b

    thresholds = []
    # threshold = m.addVar(lb = 0, ub = 1, vtype=GRB.CONTINUOUS, name="threshold")
    for i in range(len(result)):
        thresholds.append(m.addVar(lb=0, ub=2, vtype=GRB.CONTINUOUS))
        # m.setObjectiveN(gp.abs_(result[i]-outputs[i]-thresholds[i]), index = i, priority = 1)
        if outputs[i]>0:
            m.addConstr(result[i]-thresholds[i]<=outputs[i])
            m.addConstr(result[i]+thresholds[i]>=outputs[i])
        else:
            m.addConstr(result[i]-thresholds[i]<=0)

    expr = gp.quicksum(changes)
    epsilon_max_2 = m.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.addConstr(expr>=0)
    m.addConstr(expr-epsilon_max_2<=0)
    m.update()
    sum_thresholds = gp.quicksum(thresholds)
    epsilon_max_3 = m.addVar(lb=0,ub=10,vtype=GRB.CONTINUOUS, name="epsilon_max_3")
    m.addConstr(sum_thresholds>=0)
    m.addConstr(sum_thresholds-epsilon_max_3<=0)
    # m.setObjective(epsilon_max_3, GRB.MINIMIZE)
    # m.setObjectiveN(threshold, index = 2, priority = 1)
    # m.setObjectiveN(epsilon_max_3, index = 4, priority = 10)
    m.setObjectiveN(epsilon_max_3, index = 0, priority = 1)
    # m.setObjectiveN(epsilon_max_2, index = 1, priority = 1)
    # m.setObjectiveN(max_change, index = 3, priority = 1)
    # m.setObjectiveN(epsilon_max_2, GRB.MINIMIZE, 1)

    m.optimize()
    print(m.getVarByName("epsilon_max_2"))
    print(m.getVarByName("epsilon_max_3"))
    print(m.getVarByName("sum_thresholds"))
    print(m.getVarByName("max_change"))
    # print(thresholds)
    m.write("abc.lp")

    modifications = []
    for i in range(len(changes)):
        print(inputs[i], float(changes[i].X))
        modifications.append(float(changes[i].X)+inputs[i])
    
    # for i in range(len(result)):
    #     print(result[i], outputs[i])
    print(modifications)
    return modifications

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
    ad_inp = Z3Attack(sat_in, extractedModel, neuron_values_1)
    originalModel = loadModel()
    true_output = originalModel.predict([sat_in])
    true_label = np.argmax(true_output)
    if len(ad_inp)>0:
        ad_output = originalModel.predict([ad_inp])
        print(ad_output)
        predicted_label = np.argmax(ad_output)
        print(predicted_label)
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
        if predicted_label!=true_label:
            print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
    # print()
    # print()
    # ad_inp2 = GurobiAttack(sat_in, extractedModel, neuron_values_1)
    # if len(ad_inp2)>0:
    #     ad_output = originalModel.predict([ad_inp2])
    #     print(ad_output)
    #     predicted_label = np.argmax(ad_output)
    #     print(predicted_label)
    #     vals = get_neuron_values_actual(originalModel, ad_inp2, num_layers)
    #     ch = 0
    #     max_shift = 0
    #     for i in range(len(vals[0])):
    #         ch = ch + abs(vals[0][i]-neuron_values_1[i])
    #         if abs(vals[0][i]-neuron_values_1[i])>max_shift:
    #             max_shift = abs(vals[0][i]-neuron_values_1[i])
    #         # print(vals[0][i], neuron_values_1[i])
    #     print("Overall shift : ", ch)
    #     print("Maximum shift: ", max_shift)
    #     if predicted_label!=true_label:
    #         print("Attack was successful. Label changed from ",true_label," to ",predicted_label)
        
generateAdversarial()