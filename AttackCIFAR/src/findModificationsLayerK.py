from math import ceil
from time import time
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import gurobipy as grb

"""
Finds minimal modification in any k-th layer of the given Network such that the true label gets minimum value after modification is applied.
"""

def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 0
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
    mark = 0.50
    positive_index = ceil(mark*len(positive_vals))
    positive_heuristic = positive_vals[len(positive_vals)-positive_index]
    negative_index = ceil(mark*len(negative_vals))
    negative_heuristic = negative_vals[negative_index]
    return positive_heuristic, negative_heuristic

def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max, mode, layer_to_change, labels):
        neurons = []
        val_max = 1000
        l = 0
        epsilons = []
        last_layer = num_layers-1
        weights = loaded_model.get_weights()
        for i in range(0,len(weights),2):
            w = weights[i]
            b = weights[i+1]
            shape0 = w.shape[0]
            shape1 = w.shape[1]
            epsilon = []
            v=0
            if int(i/2) == layer_to_change:
                cutOffP, cutOffN = FindCutoff(w)
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        if w[row][col]>=cutOffP or w[row][col]<=cutOffN:
                            if labels[int(i/2)][col]==1:
                                v= v+1
                                ep.append(gurobi_model.addVar(lb=-val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                                gurobi_model.addConstr(ep[col]>=0)
                            elif labels[int(i/2)][col]==-1:
                                v= v+1
                                ep.append(gurobi_model.addVar(lb=-val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]<=0)
                                gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                            else:
                                v= v+1
                                ep.append(gurobi_model.addVar(lb=-val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                                gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                            
                        else:
                            ep.append(0)
                    epsilon.append(ep)
            
            else:
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        ep.append(0)
                    epsilon.append(ep)
            gurobi_model.update()
            
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
                    input.append(gurobi_model.addVar(lb=-val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                    gurobi_model.addConstr(input[r]-result[r]==0)
                else:
                    input.append(0)
                
            neurons.append(input)
            l = l + 1
        return neurons[len(neurons)-1], epsilons

def find(epsilon, model, inp, expected_label, num_inputs, num_outputs, mode, layer_to_change, labels):
    epsilon = 100
    num_layers = len(model.layers)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    # m = gp.Model("Model")
    m.setParam('NonConvex', 2)
    epsilon_max = m.addVar(lb = 0, ub = epsilon, vtype=GRB.CONTINUOUS, name="epsilon_max")

    neurons = get_neuron_values_actual(model, inp, num_layers)

    m.update()
    result, all_epsilons = get_neuron_values(model, inp, num_layers, neurons, m, epsilon_max, mode, layer_to_change, labels)
    m.update()
    resultVar = m.addVar(lb = -100, ub = 100, vtype=GRB.CONTINUOUS, name="resultVar")
    m.addConstr(resultVar-result[expected_label]==0)
    for i in range(len(result)):
        if i==expected_label:
            continue
    m.update()
    
    e2 = grb.quicksum([grb.quicksum([grb.quicksum(y) for y in all_epsilons[x]]) for x in range(len(all_epsilons))])
    epsilon_max_2 = m.addVar(lb = 0, ub = epsilon, vtype=GRB.CONTINUOUS, name="epsilon_max_2")
    m.update()
    m.addConstr(e2+epsilon_max_2>=0)
    m.addConstr(e2-epsilon_max_2<=0)
    m.update()
    m.setObjectiveN(epsilon_max, index = 2, priority = 1)
    m.setObjectiveN(epsilon_max_2, index = 1, priority = 1)
    m.setObjectiveN(resultVar, index = 0, priority = 2)
    m.optimize()
    
    summation = 0

    c = 0
    neg = 0
    for i in range(len(all_epsilons)):
        for j in range(len(all_epsilons[i])):
            for k in range(len(all_epsilons[i][j])):
                if type(all_epsilons[i][j][k])==int:
                    continue
                if all_epsilons[i][j][k].X!=0:
                    summation = summation + abs(all_epsilons[i][j][k].X)
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
    return eps