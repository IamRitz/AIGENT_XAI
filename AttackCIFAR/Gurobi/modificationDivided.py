from math import ceil
from time import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobipy as grb

"""
Finds minimal modification in any k-th layer for the ACAS-Xu Networks so that Output 0 is highest.
"""


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

def FindCutoff(w):
    positive_vals = []
    negative_vals = []
    for i in range(len(w)):
        for j in range(len(w[i])):
            if w[i][j]>0:
                positive_vals.append(w[i][j])
            else:
                negative_vals.append(w[i][j])
    positive_vals.sort()
    negative_vals.sort()
    # print(len(positive_vals))
    # print()
    # print(len(negative_vals))
    # mark = 4000/(len(positive_vals)+len(negative_vals))
    # print("Marks is:", mark)
    mark = 0.25
    # positive_vals.sort()
    positive_index = ceil(mark*len(positive_vals))
    positive_heuristic = positive_vals[len(positive_vals)-positive_index]
    # print(positive_index)
    # negative_vals.sort()
    negative_index = ceil(mark*len(negative_vals))
    # print(negative_index)
    negative_heuristic = negative_vals[negative_index]
    return positive_heuristic, negative_heuristic
    
def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max, mode, layer_to_change):
        neurons = []
        val_max = 50
        l = 0
        epsilons = []
        last_layer = num_layers-1
        first_layer = 0
        # print("Number of layers: ",num_layers)
        # layer_to_change = 1
        weights = loaded_model.get_weights()
        # print("Number of weights: ",len(weights))
        for i in range(0,len(weights),2):
            # print(i,num_layers)
            w = weights[i]
            b = weights[i+1]
            shape0 = w.shape[0]
            shape1 = w.shape[1]
            epsilon = []
            v = 0
            # print(np.shape(input), np.shape(values[int(i/2)]), np.shape(w))
            if int(i/2) == layer_to_change:
                # print("YES")
                cutOffP, cutOffN = FindCutoff(w)
                # print(cutOffP, cutOffN)
                # print("Adding modifications to layer:", layer_to_change, np.shape(w))
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        if w[row][col]>=cutOffP or w[row][col]<=cutOffN:
                            v= v+1
                            ep.append(gurobi_model.addVar(lb = -val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                            gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                            gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                           # gurobi_model.update()
                        else:
                            ep.append(0)
                    epsilon.append(ep)
                    # print("Epsilon appended:",row)
            
            else:
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        ep.append(0)
                    epsilon.append(ep)
            gurobi_model.update()
            if int(i/2) == layer_to_change:
                # print("Number of chnages : ", v)
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
                    input.append(gurobi_model.addVar(lb = -val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                    gurobi_model.addConstr(input[r]-result[r]==0)
                    # input.append(result[r])
                else:
                    input.append(0)
                
            neurons.append(input)
            l = l + 1
        return neurons[len(neurons)-1], epsilons

def find(epsilon, model, inp, expected_outputs, mode, layer_to_change, phaseGiven, phases):
    # print("Entered.")
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
    if phaseGiven==1:
        neurons = phases

    # for i in range(num_inputs):
    #     input_vars.append(m.addVar(inp[i], inp[i], vtype=grb.GRB.CONTINUOUS))

    t1 = time()
    m.update()
    result, all_epsilons = get_neuron_values(model, inp, num_layers, neurons, m, epsilon_max, mode, layer_to_change)
    # print(result)
    m.update()
    t2 = time()
    
    z, p = 0, 0
    tr = 2
    for i in range(len(result)):
        if expected_outputs[i]<=0:
            m.addConstr(result[i]<=tr)
            z = z+1
        else:
            m.addConstr(result[i]-expected_outputs[i]<=tr)
            p=p+1
    m.update()
    
    e2 = grb.quicksum([grb.quicksum([grb.quicksum(y) for y in all_epsilons[x]]) for x in range(len(all_epsilons))])
    epsilon_max_2 = m.addVar(lb = 0, ub = epsilon, vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.update()
    m.addConstr(e2+epsilon_max_2>=0)
    m.addConstr(e2-epsilon_max_2<=0)
    m.update()
    m.setObjectiveN(epsilon_max_2, index = 2, priority = 10)
    m.optimize()
    
    summation = 0

    c = 0
    neg = 0
    for i in range(len(all_epsilons)):
        # print(np.shape(all_epsilons[i]))
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                # print(all_epsilons[i][j][k])
                if type(all_epsilons[i][j][k])==int:
                    continue
                if all_epsilons[i][j][k].X!=0:
                    summation = summation + abs(all_epsilons[i][j][k].X)
                    # print(i,j,k, all_epsilons[i][j][k].X)
                    c = c + 1
                if all_epsilons[i][j][k].X<0:
                    neg = neg + 1
                
    eps = []
    for i in range(len(all_epsilons)):
        eps_1 = np.zeros_like(all_epsilons[i])
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                if type(all_epsilons[i][j][k])==int:
                    eps_1[j][k] = all_epsilons[i][j][k]
                    continue
                eps_1[j][k] = float(all_epsilons[i][j][k].X)
        eps.append(eps_1)
    # print(len(eps), np.shape(eps[0]))
    return eps
