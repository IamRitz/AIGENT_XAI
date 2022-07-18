from math import ceil
from time import time
from gurobipy import GRB
import numpy as np
import gurobipy as grb

"""
Finds minimal modification in the k-th layer of the divided network.
What is the difference?
In findModuficationsLayerK, the objective function was concerned only 
with maximizing the value of attack label and minimizing the value of true label.
But, in modificationDivided, the objective function is to retain the values found in previos modifications.
"""

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

def FindCutoff(w, num_layers):
    # print(num_layers)
    positive_vals = []
    negative_vals = []
    for i in range(len(w)):
        for j in range(len(w[i])):
            if w[i][j]>=0:
                positive_vals.append(w[i][j])
            else:
                negative_vals.append(abs(w[i][j]))
    positive_vals.sort()
    negative_vals.sort()
    mark = 1
    if num_layers==0:
        mark = 0.001
    positive_index = ceil(mark*len(positive_vals))
    positive_heuristic = positive_vals[len(positive_vals)-positive_index]
    # print(positive_index, len(positive_vals))
    negative_index = ceil(mark*len(negative_vals))
    # print(negative_index, len(negative_vals))
    negative_heuristic = negative_vals[len(negative_vals)-negative_index]

    return positive_heuristic, negative_heuristic
    
def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max, mode, layer_to_change):
        # print(layer_to_change)
        neurons = []
        val_max = 100
        l = 0
        epsilons = []
        last_layer = num_layers-1
        first_layer = 0
        weights = loaded_model.get_weights()
        for i in range(0,len(weights),2):
            w = weights[i]
            b = weights[i+1]
            shape0 = w.shape[0]
            shape1 = w.shape[1]
            epsilon = []
            v = 0
            if int(i/2) == layer_to_change:
                cutOffP, cutOffN = FindCutoff(w, layer_to_change)
                # print(cutOffP, cutOffN)
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        if w[row][col]>=cutOffP or (abs(w[row][col])>=cutOffN and w[row][col]<0):
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
                print("Number of changes : ", v)
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
    env = grb.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = grb.Model("Model", env=env)
    # m = grb.Model("Model")
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
    tr = 5
    for i in range(len(result)):
        if expected_outputs[i]<=0:
            m.addConstr(result[i]<=1)
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
