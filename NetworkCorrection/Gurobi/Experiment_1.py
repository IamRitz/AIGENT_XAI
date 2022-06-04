import os
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
import argparse
from time import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import numpy as np
import gurobipy as grb
from ConvertNNETtoTensor import ConvertNNETtoTensorFlow
"""
Finds minimal modification only in Layer 0 for the toy example given by Madhukar Sir so that Output1 is greater than Output 0.
"""
def loadModel():
    obj = ConvertNNETtoTensorFlow()
    file = '../Models/testdp1_2_2op.nnet'
    model = obj.constructModel(fileName=file)
    print(type(model))
    print(model.summary())
    return model

def getInputs():
    inp = [-1, -1, -1, -1]
    return inp

def getOutputs():
    out = [1, -1]
    return out

def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 0
        for layer in loaded_model.layers:
            # if l==0:
            #     l = l + 1
            #     continue
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
        print("Phases:",neurons)
        return neurons

def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max, mode):
        neurons = []
        u=0
        l = 0
        epsilons = []
        last_layer = num_layers-1
        first_layer = 0
        # print("Last layer is: ",last_layer)
        layer_to_change = 0
        weights = loaded_model.get_weights()
        # print("Number of weights: ",len(weights))
        for i in range(0,len(weights),2):
            # print(i,num_layers)
            w = weights[i]
            b = weights[i+1]
            shape0 = w.shape[0]
            shape1 = w.shape[1]
            epsilon = []
            
            # print(np.shape(input), np.shape(values[int(i/2)]), np.shape(w))
            if int(i/2) == layer_to_change:
                # print("For first layer, with mode:", mode, i)
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        if mode==1:
                            ep.append(gurobi_model.addVar(lb = -100, vtype=grb.GRB.CONTINUOUS))
                            gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                            gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                            gurobi_model.update()
                            # print(ep[col])
                            u = u + 1
                        
                    epsilon.append(ep)
                # temp = []
                # for row in range(shape0):
                #     t = []
                #     for col in range(shape1):
                #         t.append(gurobi_model.addVar(lb = -100, vtype=grb.GRB.CONTINUOUS))
                #         gurobi_model.addConstr(t[col]-epsilon[row][col]<=w[row][col])
                #         gurobi_model.addConstr(t[col]+epsilon[row][col]>=w[row][col])
                #         gurobi_model.update()
                #     temp.append(t)
                # w=temp
            else:
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        ep.append(0)
                    epsilon.append(ep)
            # print(np.shape(input), np.shape(w))
            if int(i/2) == layer_to_change:
                # print(".........................................")
                # print(input, w+epsilon)
                # print(".........................................")
                result = np.matmul(input, w+epsilon ) + b
                epsilons.append(epsilon)
            else:
                result = np.matmul(input, w) + b 

            # print(result)
            if int(i/2) == last_layer:
                input = result
                neurons.append(input)
                continue
            
            input = []
            for r in range(len(result)):
                if r==0 or r==1 or r==2 or r==3: 
                    input.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                    gurobi_model.addConstr(input[r]-result[r]==0)
                    # input.append(result[r])
                    # print("For r:",r)
                    gurobi_model.update()
                else:
                    input.append(0)
                
            neurons.append(input)
            l = l + 1
        # print(np.shape(neurons))
        # print(neurons[len(neurons)-1])
        # print("Number of epsilons added:", u)
        # print(neurons[len(neurons)-1])
        return neurons[len(neurons)-1], epsilons

def find(epsilon, model, inp, true_label, num_inputs, num_outputs, mode):
    num_layers = len(model.layers)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    # m = gp.Model("Model")
    # m.setParam('NonConvex', 2)
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
    adder = m.addVar(lb = 1, vtype=GRB.CONTINUOUS, name="adder")
    m.addConstr(result[1] - result[0] - adder>=0)

    t3 = time()
    m.update()
    # cum = grb.abs_(all_epsilons[0][0][0])
    # for i in range(len(all_epsilons)):
    #     for j in range(len(all_epsilons[i])):
    #         for k in range(len(all_epsilons[i][j])):
    #             print(grb.abs_(all_epsilons[i][j][k]))
    #             cum=cum+grb.abs_((all_epsilons[i][j][k]))
    e2 = grb.quicksum([grb.quicksum([grb.quicksum([x for x in all_epsilons[i][j] ]) for j in range(len(all_epsilons[i]))]) for i in range(len(all_epsilons))])
    epsilon_max_2 = m.addVar(lb = 0, ub = epsilon, vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.update()
    m.addConstr(e2+epsilon_max_2>=0)
    m.addConstr(e2-epsilon_max_2<=0)
    m.update()
    m.setObjective(epsilon_max+epsilon_max_2+adder, GRB.MINIMIZE)
    
    t4 = time()
    # print("Epsilons are:", all_epsilons)
    t5 = time()
    print("Begin optimization.")
    # m.write("model.ilp")
    m.optimize()
    t6 = time()
    # m.computeIIS()
    # m.write("model.ilp")
    # m.write("model.lp")

    print("Times taken respectively: ",(t2-t1), (t3-t2), (t4-t3), (t5-t4), (t6-t5),)
    summation = 0

    print("\nQuery has: ", m.NumObj, " objectives.")
    print(m.getVarByName("epsilon_max"))
    print(m.getVarByName("epsilon_max_2"))
    print(m.getVarByName("adder"))
    print()
    # print(len(all_epsilons))
    c = 0
    for i in range(len(all_epsilons)):
        # print(len(all_epsilons[i]))
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                if all_epsilons[i][j][k].X!=0:
                    summation = summation + abs(all_epsilons[i][j][k].X)
                    # print(i,j,k, all_epsilons[i][j][k].X)
                    c = c + 1
                # print(all_epsilons[i][j][k].VarName, all_epsilons[i][j][k].X)
                # print(m.getVarByName(all_epsilons[i][j][k]))
    
    print("Effective change was: ", summation)
    print("The number of weights changed were: ",c)

    eps = []
    for i in range(len(all_epsilons)):
        eps_1 = np.zeros_like(all_epsilons[i])
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                eps_1[j][k] = float(all_epsilons[i][j][k].X)
        eps.append(eps_1)
    return eps
    # m.reset(0)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', default='1', help='The mode in which the file should execute. If mode is 1,\
#                                                      the implementation corresponds to section 1.2.1 of Report_v1,\
#                                                     otherwise the implementation corresponds to section 1.2.2 of Report_v1.')

#     args = parser.parse_args()
#     mode = int(args.mode)
#     model = loadModel()
#     inp = getInputs()

#     num_inputs = len(inp)
#     sample_output = getOutputs()
#     op = model.predict(np.array([inp]))
#     # print(op)
#     true_label = (np.argmax(sample_output))
#     num_outputs = len(sample_output)
    
#     # print(true_label)

#     find(100, model, inp, true_label, num_inputs, num_outputs, mode)