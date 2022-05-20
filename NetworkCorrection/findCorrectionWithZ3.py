import argparse
from math import ceil
from wsgiref.validate import InputWrapper
import z3
import sys
from time import time
import keras
from numpy import genfromtxt, safe_eval, var
sys.path.append('../')
import numpy as np
from maraboupy import Marabou
import numpy as np
from MarabouNetworkCustom import read_tf_weights_as_var
from maraboupy import MarabouCore
from maraboupy.Marabou import createOptions
from scipy import stats
z3.set_param('parallel.enable', True)
z3.set_param('get-info','reason-unknown')
"""
This file implemenets code to find correction using z3.
What has been completed till now?
    1. Find epsilon only for the last layer such that the property can be correted.

What is still pending?
    1. Implement finding epsilons for all layers or atleast till two or three layers within permissible time.

"""

class findCorrectionWithZ3:
    def loadModel(self):
        json_file = open('./Models/ACASXU_2_9.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./Models/ACASXU_2_9.h5")
        return loaded_model

    def ReLU(self, input):
        return np.vectorize(lambda y: z3.If(y>=0, y, z3.RealVal(0)))(input)

    def absZ3(self, x):
        return z3.If(x>=0, x, -x)

    def get_neuron_values(self, loaded_model, input, num_layers):
        neurons = []
        l = 0
        for layer in loaded_model.layers:
            if l==0:
                l = l + 1
                continue
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # print(w)
            result = np.matmul(input,w)+b
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            input = [max(r,0) for r in result]
            neurons.append(input)
            l = l + 1
        return neurons

    def fixEpsilon(self):

        return 0
    
    def neuronHeuristics(self, neurons, max_layers):
        all_positive_neurons = []
        for i in range(len(neurons)):
            if i == max_layers-1:
                break
            n = neurons[i]
            for val in n:
                if val>0:
                    all_positive_neurons.append(val)

        # print("Printing all neuron values: ",all_positive_neurons)
        # print("Length: ", len(all_positive_neurons))

        mean = np.mean(all_positive_neurons)
        median = np.median(all_positive_neurons)
        mode = stats.mode(all_positive_neurons)

        # print("Mean: ", mean)
        # print("Median: ", median)
        # print("Mode: ", mode)
        return mean

    def weightHeuristics(self, model, max_layers):
        all_positive_weights = []
        all_negative_weights = []
        l = 0
        for layer in model.layers:
            if l==0:
                l = l + 1
                continue
            w = layer.get_weights()[0]
            # print(np.shape(w))
            shape = np.shape(w)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    if w[i][j]>0:
                        all_positive_weights.append(w[i][j])
                    elif w[i][j]<0:
                        all_negative_weights.append(w[i][j])

        # print("Number of positive weights: ", len(all_positive_weights))

        meanP = np.mean(all_positive_weights)
        medianP = np.median(all_positive_weights)
        modeP = stats.mode(all_positive_weights)

        # print("Mean: ", meanP)
        # print("Median: ", medianP)
        # print("Mode: ", modeP)

        # print("Number of negative weights: ", len(all_negative_weights))

        meanN = np.mean(all_negative_weights)
        medianN = np.median(all_negative_weights)
        modeN = stats.mode(all_negative_weights)

        # print("Mean: ", meanN)
        # print("Median: ", medianN)
        # print("Mode: ", modeN)

        mark = 0.25
        all_positive_weights.sort()
        positive_index = ceil(mark*len(all_positive_weights))
        positive_heuristic = all_positive_weights[len(all_positive_weights)-positive_index]

        all_negative_weights.sort()
        negative_index = ceil(mark*len(all_negative_weights))
        negative_heuristic = all_negative_weights[negative_index]
        # print("My heuristic:", positive_heuristic)
        # print("My heuristic:", negative_heuristic)
        return positive_heuristic, negative_heuristic
    
    def find(self, neurons, mean, m, epsilon_max, input, max_layers, inp, meanP, meanN):
        weight_threshold = 0.1
        # summation = z3.Real('summation')
        sum = z3.Real('sum')
        # summation = 0
        sumify = z3.Real('sumify')
        m.add(z3.And(sumify<=epsilon_max, sumify>=0))
        model = self.loadModel()
        weights = model.get_weights()
        layer_output = 0
        k = 0
        neurons.reverse()
        neurons.append(inp)
        neurons.reverse()
        
        counted_ones = []
        for i in range(0, len(weights)-1,2):
            w = weights[i]
            b = weights[i+1]
            shape0 = (w.T).shape[0]
            shape1 = (w.T).shape[1]
            neuron_current_layer = neurons[int(i/2)]
            # if i==0:
            #     neuron_current_layer = inp
            print(shape0,"  ", shape1, "  ", len(neuron_current_layer))
            epsilon = []
            for row in range(shape0):
                ep = z3.RealVector('e'+str(int(i/2))+'_'+str(row), shape1)
                
                for col in range(shape1):
                    if i==0 or i==2 or i==4 or i==6 or i==10:
                        m.add(ep[col]==0)
                    elif neuron_current_layer[col]>mean:
                        # print(row,",",col)
                        if abs(w[col][row])>meanP or abs(w[col][row])<meanN:
                            # sumify = sumify + z3.Sum([self.absZ3(ep[col]))
                            counted_ones.append(self.absZ3(ep[col]))
                            # counted_ones.append(ep[col])
                            m.add(ep[col]>=-epsilon_max)
                            m.add(ep[col]<=epsilon_max)
                            # m.add(ep[col]==0)
                            k = k + 1
                        else:
                            # print("For col:", ep[col])
                            m.add(ep[col]==0)
                    else:
                        m.add(ep[col]==0)
                epsilon.append(ep)
                # sumify = sumify + z3.Sum([self.absZ3(x) for x in ep])
            """
            Create an epsilon array at every stage which should be added to the weight array and put constraints on it.
            """
            out = (w.T+epsilon) @ input + b
            # out = w.T @ input + b
            # if i==12:
            #     out = (w.T+epsilon) @ input + b
            layer_output = self.ReLU(out)
            input = layer_output
        
        print("Added sum constraints.")
        print("k was: ",k)
        len1 = len(counted_ones)+1
        summing = z3.RealVector("summing", len1)
        m.add(summing[0]==0)
        m.add(summing[len1-1]<=epsilon_max)
        for i in range(len1-1):
            m.add(summing[i+1]==counted_ones[i]+summing[i])
        # print(len(layer_output))
        return layer_output

    def result(self, input, m, epsilon_max, model, neurons, max_layers):
        summation = z3.Real('summation')
        sum = z3.Real('sum')
        summation = 0
        # model = self.loadModel()
        weights = model.get_weights()
        layer_output = 0
        for i in range(0, len(weights)-1,2):
            w = weights[i]
            b = weights[i+1]
            shape0 = (w.T).shape[0]
            shape1 = (w.T).shape[1]
            epsilon = []
            for row in range(shape0):
                if i==12:
                    ep = z3.RealVector('e'+str(row), shape1)
                    for col in ep:
                        summation = summation + self.absZ3(col)
                        m.add(col>=-epsilon_max)
                        m.add(col<=epsilon_max)
                    epsilon.append(ep)
            """
            Create an epsilon array at every stage which should be added to the weight array and put constraints on it.
            """
            # out = (w.T+epsilon) @ input + b
            out = w.T @ input + b
            if i==12:
                out = (w.T+epsilon) @ input + b
            layer_output = self.ReLU(out)
            input = layer_output
        m.add(z3.And(summation<=epsilon_max, summation>=0))
        m.add(summation==sum)
        return layer_output

    def callZ3(self, m, epsilon_max, model, input, neurons, max_layers, mean, meanP, meanN):
        inputs = genfromtxt('./data/inputs.csv', delimiter=',')
        input =inputs[0]
        classes = 5
        input_vars = z3.RealVector('input_vars',len(input))
        output_vars = z3.RealVector('output_vars', classes)
        
        ###Adding constraints for inputs###
        for i in range(len(input)):
            # print("Input:",i)
            m.add(input_vars[i]==input[i])
        ###Constraints for inputs added.###
        r = self.find(neurons, mean, m, epsilon_max, input_vars, max_layers, input, meanP, meanN)
        print(len(r))
        
        ###Adding constraints for outputs###
        m.add(r[0]>r[2])
        m.add(r[0]>r[3])
        m.add(r[0]>r[4])
        m.add(r[1]>r[2])
        m.add(r[1]>r[3])
        m.add(r[1]>r[4])
        for i in range(len(output_vars)):
            m.add(output_vars[i]==r[i])
            m.add(output_vars[i]!=0)
        ###Constraints for outputs added.###

    def calc(self, epsilon_max, tolerance, model, input, neurons, max_layers):
        epsilon_max = 5
        tolerance = 0.0001
        ep = epsilon_max
        sat_epsilon = epsilon_max
        unsat_epsilon = 0
        mean = self.neuronHeuristics(neurons, num_layers)
        meanP, meanN = self.weightHeuristics(model, num_layers)
        
        while abs(sat_epsilon-unsat_epsilon)>tolerance:
            m = z3.Solver()       
            self.callZ3(m, ep, model, input, neurons, max_layers, mean, meanP, meanN)
            t1 = time()
            f = open("demofile2.txt", "a")
            # print(m.assertions())
            for x in m.assertions():
                f.write(str(x))
                f.write("\n")
            f.close()
            print("Assertions written to file.")
            solution = m.check()
            t2 = time()
            print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", solution)
            if solution==z3.sat:
                s = 0
                s2 = 0
                # print(m.model,"\n\n")
                for line in m.model():
                    x2 = m.model()[line].as_fraction()
                    val = float(x2.numerator / x2.denominator)
                    print(line, "---> ", val)
                    if "input" in str(line) or "output" in str(line) or "sum" in str(line):
                        
                        continue
                    s = s + abs(val)
                    # print("This was counted: ",line, "---> ", val)
                    s2 = s2 + val
                sat_epsilon = s
                print("Threshold is: ", ep)
                print("Sat epsilon: ", sat_epsilon)
                print("Total change is: ", s)
                print("Effective change is: ", s2)
            else:
                unsat_epsilon = ep
                print("Unsat time: ", unsat_epsilon)
            ep = (sat_epsilon+unsat_epsilon)/2
            break
        return ep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tolerance', default=0.0001, help='This is the difference between epsilon used when a\
                                                    SAT solution is given and the epsilon when an UNSAT solution is given.')
    
    parser.add_argument('--epsilon_max', default=5, help='This is the maximum amount of change that can be introduced\
                                                    in the network to correct the properties.')

    args = parser.parse_args()
    tolerance = float(args.tolerance)
    epsilon_max = float(args.epsilon_max)

    classObject = findCorrectionWithZ3()
    num_layers = 7
    inputs = genfromtxt('./data/inputs.csv', delimiter=',')
    input = inputs[0]
    model = classObject.loadModel()
    neurons = classObject.get_neuron_values(model, input, num_layers)
    t3 = time()
    epsilon = classObject.calc(epsilon_max, tolerance, model, input, neurons, num_layers)
    t4 = time()
    print("The total time taken was: ",(t4-t3)," seconds.\n The minimum change done was: ",epsilon)
    