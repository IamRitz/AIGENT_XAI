from time import time
from gurobipy import GRB
import numpy as np
import gurobipy as grb

"""
Finds minimal modification in the k-th layer of the divided network.
What is the difference?
In findModuficationsLayerK, the objective function was concerned only 
with maximizing the value of attack label and minimizing the value of true label.
But, in modificationDivided, the objective function is to retain the values found in previous modifications.
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

def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model, epsilon_max, mode, layer_to_change, labels):
        neurons = []
        val_max = 10
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
            
            if int(i/2) == layer_to_change:
                for row in range(shape0):
                    ep = []
                    for col in range(shape1):
                        if mode==1:
                            ep.append(gurobi_model.addVar(lb = -val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                            gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                            gurobi_model.addConstr(ep[col]+epsilon_max>=0)
                        else:
                            if col==0 :
                                ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]-epsilon_max<=0)
                                gurobi_model.addConstr(ep[col]>=0)
                            else:
                                ep.append(gurobi_model.addVar(vtype=grb.GRB.CONTINUOUS))
                                gurobi_model.addConstr(ep[col]<=0)
                                gurobi_model.addConstr(ep[col]+epsilon_max>=0)
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
                    input.append(gurobi_model.addVar(lb = -val_max, ub = val_max, vtype=grb.GRB.CONTINUOUS))
                    gurobi_model.addConstr(input[r]-result[r]==0)
                    # input.append(result[r])
                else:
                    input.append(0)
                
            neurons.append(input)
            l = l + 1
        return neurons[len(neurons)-1], epsilons

def find(epsilon, model, inp, expected_outputs, mode, layer_to_change, phaseGiven, phases, labels):
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

    m.update()
    result, all_epsilons = get_neuron_values(model, inp, num_layers, neurons, m, epsilon_max, mode, layer_to_change, labels)
    m.update()
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
                eps_1[j][k] = float(all_epsilons[i][j][k].X)
        eps.append(eps_1)
    return eps
