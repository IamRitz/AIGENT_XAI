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
# z3.set_param('get-info','reason-unknown')
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
    
    def result(self, input, model):
        weights = model.get_weights()
        layer_output = 0
        for i in range(0, len(weights)-1,2):
            w = weights[i]
            b = weights[i+1]
            out = w.T @ input + b
            layer_output = self.ReLU(out)
            input = layer_output
        return layer_output

    def callZ3(self, m, epsilon_max, model, input):
        inputs = genfromtxt('./data/inputs.csv', delimiter=',')
        input =inputs[0]
        classes = 5
        input_vars = z3.RealVector('input_vars',len(input))
        epsilons = z3.RealVector('epsilons',len(input))
        output_vars = z3.RealVector('output_vars', classes)
        
        ###Adding constraints for inputs###
        for i in range(len(input)):
            m.add(z3.And(input_vars[i]>=input[i]-epsilons[i], input_vars[i]<=input[i]+epsilons[i]))
            m.add(z3.And(epsilons[i]>=-epsilon_max, epsilons[i]<=epsilon_max))
        ###Constraints for inputs added.###
        r = self.result(input_vars, model)
        print(len(r))

        m.add(z3.Sum([self.absZ3(x) for x in epsilons])<=epsilon_max, z3.Sum([self.absZ3(x) for x in epsilons])>=0 )
        
        ###Adding constraints for outputs###
        # m.add(r[0]>r[2])
        # m.add(r[0]>r[3])
        # m.add(r[0]>r[4])
        # m.add(r[1]>r[2])
        # m.add(r[1]>r[3])
        # m.add(r[1]>r[4])
        # m.add(z3.Or(z3.And(r[0]>r[2], r[0]>r[3], r[0]>r[4]),z3.And(r[1]>r[2], r[1]>r[3], r[1]>r[4])))

        for i in range(len(output_vars)):
            m.add(output_vars[i]==r[i])
            m.add(output_vars[i]!=0)
        ###Constraints for outputs added.###

    def calc(self, epsilon_max, tolerance, model, input):
        epsilon_max = 1
        tolerance = 0.0001
        ep = epsilon_max
        sat_epsilon = epsilon_max
        unsat_epsilon = 0
        
        while abs(sat_epsilon-unsat_epsilon)>tolerance:
            m = z3.Solver()       
            self.callZ3(m, ep, model, input)
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
    t3 = time()
    epsilon = classObject.calc(epsilon_max, tolerance, model, input)
    t4 = time()
    print("The total time taken was: ",(t4-t3)," seconds.\n The minimum change done was: ",epsilon)
    