from time import time
import gurobipy as gp
import gurobipy as gp
from gurobipy import GRB
from numpy import genfromtxt
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
import keras
import numpy as np
import sys
# sys.path.append('../')
import tensorflow as tf
import numpy as np
import gurobipy as grb
import os
from relumip import AnnModel

"""
Finds minimal modification across all layers for the ACAS-Xu Networks given by Madhukar Sir so that Output 0 is highest.
"""

def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/ACASXU_run2a_3_8_batch_2000.nnet'
    model = obj.constructModel(fileName=file)
    print(type(model))
    return model

# def loadModel():
#     json_file = open('../Models/ACASXU_2_9.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = keras.models.model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights("../Models/ACASXU_2_9.h5")
#     print(type(loaded_model))
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
        l = 0
        for layer in loaded_model.layers:
            # if l==0:
            #     l = l + 1
            #     continue
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

def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max):
        neurons = []
        l = 0
        epsilons = []
        last_layer = num_layers-1
        weights = loaded_model.get_weights()
        print(np.shape(values[6]))
        for i in range(0,len(weights),2):
            # print(i,num_layers)
            w = weights[i]
            b = weights[i+1]
            shape0 = w.shape[0]
            shape1 = w.shape[1]
            epsilon = []
            
            # print(np.shape(input), np.shape(values[int(i/2)]), np.shape(w))
            if int(i/2) == last_layer:
                print("For last layer:")
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        # ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                        # gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                        # gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                        # gurobi_model.update()
                        if col!=0:
                            ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                            gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                            gurobi_model.addConstr(ep[col]<=0)
                            gurobi_model.update()
                        else:
                            ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                            gurobi_model.addConstr(ep[col]>=0)
                            gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                            gurobi_model.update()

                            # ep.append(gurobi_model.addVar(lb = 0, ub = 0, vtype=grb.GRB.CONTINUOUS))
                            # # ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                            # # gurobi_model.addConstr(ep[col]<=0)
                            # # gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                            # gurobi_model.update()
                    epsilon.append(ep)
            
            else:
                for row in range(shape0):
                    ep = []
                    # print(row)
                    for col in range(shape1):
                        # ep.append(0)
                        if values[int(i/2)][row]>0:
                            ep.append(gurobi_model.addVar(lb = 0, vtype=grb.GRB.CONTINUOUS))
                            # gurobi_model.addConstr(ep[col]>=0)
                            gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                        else:
                            ep.append(gurobi_model.addVar(lb = 0, ub = 0, vtype=grb.GRB.CONTINUOUS))
                            # gurobi_model.addConstr(ep[col]==0)
                    epsilon.append(ep)
            
            result = np.matmul(input, w + epsilon) + b
            epsilons.append(epsilon)

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
                    input.append(gurobi_model.addVar(lb = 0, ub = 0, vtype=grb.GRB.CONTINUOUS))
                
            neurons.append(input)
            l = l + 1
        # print(np.shape(neurons))
        # print(neurons[len(neurons)-1])
        print(len(epsilons))
        return neurons[len(neurons)-1], epsilons

def find(epsilon, model, inp, true_label, num_inputs, num_outputs):
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
    result, all_epsilons = get_neuron_values(model, inp, num_layers, neurons, m, epsilon_max)
    # print(result)
    m.update()
    t2 = time()

    m.addConstr(result[0]-result[1]>=0.0001)
    m.addConstr(result[0]-result[2]>=0.0001)
    m.addConstr(result[0]-result[3]>=0.0001)
    m.addConstr(result[0]-result[4]>=0.0001)
    # m.addConstr(result[0]-result[1]>=0.001)
    # m.addConstr(result[1]-result[2]>=0.0001)
    # m.addConstr(result[1]-result[3]>=0.0001)
    # m.addConstr(result[1]-result[4]>=0.0001)
    # m.addConstr(result[1]-result[0]>=0.0001)
    
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
    print(type(all_epsilons))
    c = 0
    for i in range(len(all_epsilons)):
        print(np.shape(all_epsilons[i]))
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                if all_epsilons[i][j][k].X>0:
                    summation = summation + all_epsilons[i][j][k].X
                    print(i,j,k, all_epsilons[i][j][k].X)
                    c = c + 1
                # print(all_epsilons[i][j][k].VarName, all_epsilons[i][j][k].X)
                # print(m.getVarByName(all_epsilons[i][j][k]))
    
    print("Effective change was: ", summation)
    print("The number of weights changed were: ",c)
    # np.save('../data/mine.vals', all_epsilons)
    # print("Wrote epsilons to file.")
    eps = []
    for i in range(len(all_epsilons)):
        eps_1 = np.zeros_like(all_epsilons[i])
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                eps_1[j][k] = float(all_epsilons[i][j][k].X)
        eps.append(eps_1)
    return eps
    # m.reset(0)

if __name__ == '__main__':
    # model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
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

    find(10, model, inp, true_label, num_inputs, num_outputs)