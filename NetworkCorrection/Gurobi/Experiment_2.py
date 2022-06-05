import argparse
from time import time
import gurobipy as gp
import gurobipy as gp
from gurobipy import GRB
from numpy import genfromtxt
import keras
import numpy as np
import sys
# sys.path.append('../')
import tensorflow as tf
import numpy as np
import gurobipy as grb
import os
from relumip import AnnModel
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow

"""
Finds minimal modification in any 1 layer for the ACAS-Xu Networks given by Madhukar Sir so that Output 0 is highest.
"""

def loadModel():
    json_file = open('../Models/ACASXU_2_9.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../Models/ACASXU_2_9.h5")
    print(type(loaded_model))
    return loaded_model

# def getInputs():
#     inputs = genfromtxt('../data/inputs.csv', delimiter=',')
#     return inputs[0]

# def getOutputs():
#     outputs = genfromtxt('../data/outputs.csv', delimiter=',')
#     return [outputs[0]]


# def loadModel():
#     obj = ConvertNNETtoTensorFlow()
#     file = '../Models/ACASXU_run2a_3_8_batch_2000.nnet'
#     model = obj.constructModel(fileName=file)
#     print(type(model))
#     return model

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
        l = 0
        for layer in loaded_model.layers:
            if l==0:
                l = l + 1
                continue
            # # print(l)
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # print(w)
            result = np.matmul(input,w)+b
            
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        # print(len(neurons))
        # print(neurons[len(neurons)-1])
        return neurons

def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max, mode):
        neurons = []
        l = 0
        epsilons = []
        last_layer = num_layers-1
        first_layer = 0
        print("Last layer is: ",last_layer)
        layer_to_change = 0
        weights = model.get_weights()
        print("Number of weights: ",len(weights))
        for i in range(0,len(weights),2):
            # print(i,num_layers)
            w = weights[i]
            b = weights[i+1]
            shape0 = w.shape[0]
            shape1 = w.shape[1]
            epsilon = []
            
            # print(np.shape(input), np.shape(values[int(i/2)]), np.shape(w))
            if int(i/2) == layer_to_change:
                print("For first layer, with mode:", mode, i)
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        if mode==1:
                            ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                            gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                            gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                            gurobi_model.update()
                        else:
                            if col==0 :
                                ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                                gurobi_model.addConstr(ep[col]>=0)
                                gurobi_model.update()
                            else:
                                ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]<=0)
                                gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                                gurobi_model.update()
                    epsilon.append(ep)
            
            else:
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        ep.append(0)
                    epsilon.append(ep)
            
            if int(i/2) == layer_to_change:
                result = np.matmul(input, w + epsilon) + b
                epsilons.append(epsilon)
            else:
                result = np.matmul(input, w) + b 


            if int(i/2) == last_layer:
                input = result
                neurons.append(input)
                continue

            input = []
            for r in range(len(result)):
                if values[int(i/2)][r]>0: 
                    input.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                    gurobi_model.addConstr(input[r]-result[r]==0)
                    # input.append(result[r])
                else:
                    input.append(0)
                
            neurons.append(input)
            l = l + 1
        # print(np.shape(neurons))
        # print(neurons[len(neurons)-1])
        return neurons[len(neurons)-1], epsilons

def find(epsilon, model, inp, true_label, num_inputs, num_outputs, mode):
    num_layers = len(model.layers)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    # m = gp.Model("Model")
    m.setParam('NonConvex', 2)
    ep = []
    input_vars = []
    epsilon_max = m.addVar(lb = 0, ub = epsilon, vtype=GRB.CONTINUOUS, name="epsilon_max")

    neurons = get_neuron_values_actual(model, inp, num_layers)

    # for i in range(num_inputs):
    #     input_vars.append(m.addVar(inp[i], inp[i], vtype=grb.GRB.CONTINUOUS))

    t1 = time()
    m.update()
    result, all_epsilons = get_neuron_values(model, inp, num_layers, neurons, m, epsilon_max, mode)
    # print(result)
    m.update()
    t2 = time()

    m.addConstr(result[0]-result[1]>=0.0001)
    m.addConstr(result[0]-result[2]>=0.0001)
    m.addConstr(result[0]-result[3]>=0.0001)
    m.addConstr(result[0]-result[4]>=0.0001)
    
    t3 = time()
    m.update()
    
    e2 = grb.quicksum([grb.quicksum([grb.quicksum(y) for y in all_epsilons[x]]) for x in range(len(all_epsilons))])
    epsilon_max_2 = m.addVar(lb = 0, ub = epsilon, vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.update()
    m.addConstr(e2+epsilon_max_2>=0)
    m.addConstr(e2-epsilon_max_2<=0)
    m.update()
    m.setObjective(epsilon_max+epsilon_max_2, GRB.MINIMIZE)
    # m.setObjectiveN(epsilon_max, index = 0, priority = 0)
    # m.setObjectiveN(epsilon_max_2, index = 1, priority = 1)
    # m.setObjectiveN(epsilon_max_2, GRB.MINIMIZE, 1)
    t4 = time()
    # print("Epsilons are:", all_epsilons)
    t5 = time()
    print("Begin optimization.")
    m.optimize()
    m.write("abc2.lp")
    t6 = time()
    print("Times taken respectively: ",(t2-t1), (t3-t2), (t4-t3), (t5-t4), (t6-t5),)
    summation = 0

    print("Query has: ", m.NumObj, " objectives.")
    print(m.getVarByName("epsilon_max"))
    print(m.getVarByName("epsilon_max_2"))
    print(len(all_epsilons))
    c = 0
    for i in range(len(all_epsilons)):
        print(np.shape(all_epsilons[i]))
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                if all_epsilons[i][j][k].X>0:
                    summation = summation + all_epsilons[i][j][k].X
                    print(i,j,k)
                    c = c + 1
                # print(all_epsilons[i][j][k].VarName, all_epsilons[i][j][k].X)
                # print(m.getVarByName(all_epsilons[i][j][k]))
    
    print("Effective change was: ", summation)
    print("The number of weights changed were: ",c)
    # m.reset(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='1', help='The mode in which the file should execute. If mode is 1,\
                                                     the implementation corresponds to section 1.2.1 of Report_v1,\
                                                    otherwise the implementation corresponds to section 1.2.2 of Report_v1.')

    args = parser.parse_args()
    mode = int(args.mode)
    model = loadModel()
    inp = getInputs()

    num_inputs = len(inp)
    sample_output = getOutputs()
    op = model.predict(np.array([inp]))
    print(op)
    true_label = (np.argmax(sample_output))
    num_outputs = len(sample_output)
    
    print(true_label)

    find(100, model, inp, true_label, num_inputs, num_outputs, mode)