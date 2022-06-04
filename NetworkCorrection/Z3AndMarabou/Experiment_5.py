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
import z3
import sys
from time import time
sys.path.append('../')
from Gurobi.ConvertNNETtoTensor import ConvertNNETtoTensorFlow
import numpy as np

"""
Finds minimal modification using Z3 but only in Layer 0.
This file is specific to the toy example sent by Madhukar Sir..
"""

def ReLU(input):
    return np.vectorize(lambda y: z3.If(y>=0, y, z3.RealVal(0)))(input)

def absZ3(x):
    return z3.If(x>=0, x, -x)

def result(input, m, model, epsilon_max):
    weights = model.get_weights()
    layer_output = 0
    all_epsilons = []
    layer_to_modify = 0
    for i in range(0,len(weights)-1,2):
        w = weights[i]
        b = weights[i+1]
        # print(w,"\n",w.T,"\n",b)
        # print(type(w))
        # print((w.T).shape)
        shape0 = w.shape[0]
        shape1 = w.shape[1]
        epsilon = []
        if layer_to_modify==i:
            print("Adding modifications to Layer:",i)
            for row in range(shape0):
                ep = []
                for col in range(shape1):
                    ep.append(z3.Real('e'+str(i)+str(row)+str(col)))
                    m.add(ep[col]>=-epsilon_max)
                    m.add(ep[col]<=epsilon_max)
                epsilon.append(ep)
            # print(epsilon)
            all_epsilons.append(epsilon)
        """
        Create an epsilon array at every stage which should be added to the weight array and put constraints on it.
        """
        # A = Array('A', IntSort(), ArraySort(IntSort(), IntSort()))
        out = w.T @ input + b
        if layer_to_modify==i:
            out = (w+epsilon).T @ input + b
            # print(w.T, epsilon)
        layer_output = ReLU(out)
        input = layer_output
    # print(all_epsilons)
    return layer_output, all_epsilons

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
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        print(neurons)
        return neurons

def callZ3(input, model, epsilon_max_2, epsilon_max):
    input_vars = z3.RealVector('input_vars',4)
    m = z3.Solver()
    ###Adding constraints for inputs###
    for i in range(len(input)):
        # print("Input:",i)
        m.add(input_vars[i]==input[i])
    ###Constraints for inputs added.###
    r, all_epsilons = result(input_vars, m, model, epsilon_max)
    sum = z3.Real('sum')
    sum = z3.Sum([z3.Sum([z3.Sum([absZ3(x) for x in all_epsilons[i][j]]) for j in range(len(all_epsilons[i]))]) for i in range(len(all_epsilons))])
    # print(sum)
    m.add(sum<=epsilon_max_2)
    m.add(sum>=0)
    # print(len(r))
    m.add(r[1]>r[0])

    t1 = time()
    solution = m.check()
    t2 = time()
    print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", solution)
    # print("The model is: ", m.model())
    
    epsilons_to_add = []
    curr_eps = []
    if solution==z3.sat:
        solution = m.model()
        dictionary = sorted ([(d, solution[d]) for d in solution], key = lambda x: str(x[0]))
        # print(type(dictionary))
        i = 1
        sum = 0
        max_change = 0
        for x in dictionary:
            r = x[1]
            # print(x, r, type(r))
            val = 0
            if z3.is_algebraic_value(r):
                r = r.approx(20)
            val = float(r.numerator_as_long())/float(r.denominator_as_long())
            """
            This code will only work when modifications are required in layer 0. 
            Change the code for a general case.
            """
            curr_eps.append(val)
            sum = sum +abs(val)
            if max_change<abs(val):
                max_change = abs(val)
            # print(x[0], val)
            if i%4==0:
                epsilons_to_add.append(curr_eps)
                curr_eps = []
            if i==16:
                break
            i = i + 1
        print("Total change was: ", sum)
        print("Maximum change was: ", max_change)
            
    # print(epsilons_to_add)
    return epsilons_to_add[0:4]

def getASolution():
    file = '../Models/testdp1_2_2op.nnet'
    obj = ConvertNNETtoTensorFlow()
    inp, out, model = obj.convert(file)
    weights = model.get_weights()
    epsilon_max_2 = 2.5
    epsilon_max = 0.5
    epsilons_to_add = callZ3(inp, model, epsilon_max_2, epsilon_max)
    print("\nEpsilons are = ", epsilons_to_add)
    weights[0] = weights[0] + np.array(epsilons_to_add)
    print("\nThe modified weights are:")
    print(np.array(weights[0]).T)
    print()
    model.set_weights(weights)
    model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    predicted = obj.predict(model, inp, out)
    print(".................\nAfter Modification, the state of nodes is:")
    get_neuron_values_actual(model, inp, 2)
    print()
    print()

getASolution()