from itertools import permutations
import itertools
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
# from time import time
sys.path.append('../')
from Gurobi.ConvertNNETtoTensor import ConvertNNETtoTensorFlow
import numpy as np
import multiprocessing
import time

"""
Finds minimal modification using Z3 by phase fixing.
Implementation for toy example.
Modification across all layers.
"""

# def ReLU(input):
#     return np.vectorize(lambda y: z3.If(y>=0, y, z3.RealVal(0)))(input)
counter=0

def ReLU(input, phase):
    x=[]
    for i in range(len(input)):
        if phase[i]>0:
            x.append(input[i])
        else:
            x.append(0)
        # x.append(z3.If(phase[i]>=0, input[i], 0))
    return x

def absZ3(x):
    return z3.If(x>=0, x, -x)

def add(m, expr):
    global counter
    m.assert_and_track(expr, "Constraint_: "+str(counter))
    counter = counter + 1

def result(input, m, model, epsilon_max, all_neuron_values, phase, counter):
    weights = model.get_weights()
    layer_output = 0
    all_epsilons = []
    layer_to_modify = 0
    last_layer = 1
    for i in range(0,len(weights)-1,2):
        # print(i, all_neuron_values[int(i/2)])
        w = weights[i]
        b = weights[i+1]
        # print(w,"\n",w.T,"\n",b)
        # print(type(w))
        # print((w.T).shape)
        shape0 = w.shape[0]
        shape1 = w.shape[1]
        epsilon = []
        # print("Adding modifications to Layer:",i)
        for row in range(shape0):
                ep = []
                for col in range(shape1):
                    if int(i/2)==last_layer:
                        # ep.append(z3.Real('e'+str(i)+str(row)+str(col)))
                        # add( m,ep[col]==0)
                        if col==0:
                            ep.append(z3.Real('e'+str(i)+str(row)+str(col)))
                            add(m, ep[col]>=0)
                            add(m, ep[col]<=epsilon_max)
                        else:
                            ep.append(z3.Real('e'+str(i)+str(row)+str(col)))
                            add(m, ep[col]>=-epsilon_max)
                            add(m, ep[col]<=0)
                    else:
                        if phase[col]==-1:
                            ep.append(z3.Real('e'+str(i)+str(row)+str(col)))
                            add(m, ep[col]>=0)
                            add(m, ep[col]<=epsilon_max)
                        else:
                            ep.append(z3.Real('e'+str(i)+str(row)+str(col)))
                            add(m, ep[col]>=-epsilon_max)
                            add(m, ep[col]<=0)
                epsilon.append(ep)
        # print(epsilon)
        all_epsilons.append(epsilon)
        """
        Create an epsilon array at every stage which should be added to the weight array and put constraints on it.
        """
        # A = Array('A', IntSort(), ArraySort(IntSort(), IntSort()))
        out = (w+epsilon).T @ input + b
        if int(i/2)==last_layer:
            input = out
            layer_output = out
            break
        layer_output = ReLU(out, phase)
        input = layer_output
    return layer_output, all_epsilons, counter

def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 1
        # print(len(loaded_model.layers))
        for layer in loaded_model.layers:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # print(w)
            result = np.matmul(input,w)+b
            
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            # print(input)
            
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        # print(neurons)
        return neurons

def callZ3(input, model, epsilon_max_2, epsilon_max, num_layers, phase):
    input_vars = z3.RealVector('input_vars',4)
    m = z3.Solver()
    # m.set("timeout", 5000)
    m.set(unsat_core=True)
    counter = 1
    ###Adding constraints for inputs###
    for i in range(len(input)):
        # print("Input:",i)
        add(m, input_vars[i]==input[i])
    ###Constraints for inputs added.###
    all_neuron_values = get_neuron_values_actual(model, input, num_layers)
    # print(len(all_neuron_values), np.shape(all_neuron_values))
    r, all_epsilons, counter = result(input_vars, m, model, epsilon_max, all_neuron_values, phase, counter)
    # print(result)
    sum = z3.Real('sum')
    sum = z3.Sum([z3.Sum([z3.Sum([absZ3(x) for x in all_epsilons[i][j]]) for j in range(len(all_epsilons[i]))]) for i in range(len(all_epsilons))])
    # print(sum)
    add(m, sum<=epsilon_max_2)
    add(m, sum>=0)
    # print(len(r))
    add(m, r[1]>r[0])

    t1 = time.time()
    solution = m.check()
    t2 = time.time()
    print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", solution)
    # print("The model is: ", m.model())
    temp=[]
    epsilons_to_add = []
    curr_eps = []
    if solution==z3.sat:
        solution = m.model()
        dictionary = sorted ([(d, solution[d]) for d in solution], key = lambda x: str(x[0]))
        # print(type(dictionary))
        i = 1
        sum = 0
        max_change = 0
        # print(dictionary)
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
            """
            This code will only work when modifications are required in layer 0. 
            Change the code for a general case.
            """
            curr_eps.append(val)
            sum = sum +abs(val)
            if max_change<abs(val):
                max_change = abs(val)
            # print(x[0], val)
            
            if i>=16 and i%2==0:
                epsilons_to_add.append(curr_eps)
                curr_eps = []
            elif i<16 and i%4==0:
                epsilons_to_add.append(curr_eps)
                curr_eps = []
            if i==16 or i==24:
                temp.append(epsilons_to_add)
                epsilons_to_add = []
            if i==24:
                break
            i = i + 1
        print("Total change was: ", sum)
        print("Maximum change was: ", max_change)
            
    # print(temp)
    else:
        print(m.unsat_core())
        return temp, z3.unsat
    return temp, z3.sat

def getASolution(phase):
    print(".....................................................................")
    print("Running for phase:,",phase)
    file = '../Models/testdp1_2_2op.nnet'
    obj = ConvertNNETtoTensorFlow()
    inp, out, model = obj.convert(file)
    weights = model.get_weights()
    epsilon_max_2 = 10
    epsilon_max = 5
    num_layers = 2
    epsilons_to_add, sol = callZ3(inp, model, epsilon_max_2, epsilon_max, num_layers, phase)
    if sol==z3.unsat:
        print(z3.unsat)
        print(".....................................................................")
        print()
        print()
        return
    # print("\nEpsilons are = ", epsilons_to_add)
    weights[0] = weights[0] + np.array(epsilons_to_add[0])
    weights[2] = weights[2] + np.array(epsilons_to_add[1])
    # print("\nThe modified weights are:")
    # print(np.array(weights[0]).T)
    # print(np.array(weights[2]).T)
    print()
    model.set_weights(weights)
    model.compile(optimizer=tf.optimizers.Adam(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    predicted = obj.predict(model, inp, out)
    if predicted[0][1]>predicted[0][0]:   
        print("Achieved for phase:", phase)
        print("For Input: ", inp)
        print("Actual output is: ", out)
        print("Predicted Output is: ", predicted)
        print("After Modification, the state of nodes is:")
        print(get_neuron_values_actual(model, inp, 2))
        
    else:
        print("Predicted Output is: ", predicted)
    print(".....................................................................")
    print()
    print()

if __name__ == '__main__':
    W = itertools.product([-1,1], repeat=4)
    for w in W:
        phase = list(w)
        p = multiprocessing.Process(target=getASolution, name="getASolution", args=(phase,))
        p.start()
        time.sleep(10)
        p.terminate()
        p.join()
