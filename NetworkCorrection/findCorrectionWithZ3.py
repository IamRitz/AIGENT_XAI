import argparse
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

    def result(self, input, m, epsilon_max):
        summation = z3.Real('summation')
        sum = z3.Real('sum')
        summation = 0
        model = self.loadModel()
        weights = model.get_weights()
        layer_output = 0
        for i in range(0, len(weights)-1,2):
            w = weights[i]
            b = weights[i+1]
            shape0 = (w.T).shape[0]
            shape1 = (w.T).shape[1]
            epsilon = []
            for row in range(shape0):
                if i==12 or i==10:
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
            if i==12 or i==10:
                out = (w.T+epsilon) @ input + b
            layer_output = self.ReLU(out)
            input = layer_output
        m.add(z3.And(summation<=epsilon_max, summation>=0))
        m.add(summation==sum)
        return layer_output

    def callZ3(self, m, epsilon_max):
        inputs = genfromtxt('./data/inputs.csv', delimiter=',')
        input =inputs[0]
        input_vars = z3.RealVector('input_vars',5)
        output_vars = z3.RealVector('output_vars',5)
        
        ###Adding constraints for inputs###
        for i in range(len(input)):
            # print("Input:",i)
            m.add(input_vars[i]==input[i])
        ###Constraints for inputs added.###
        
        r = self.result(input_vars, m, epsilon_max)
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
            m.add(output_vars[i]>0)
        ###Constraints for outputs added.###

    def calc(self, epsilon_max, tolerance):
        epsilon_max = 5
        tolerance = 0.0001
        ep = epsilon_max
        sat_epsilon = epsilon_max
        unsat_epsilon = 0
        
        while abs(sat_epsilon-unsat_epsilon)>tolerance:
            m = z3.Solver()       
            self.callZ3(m, ep)
            t1 = time()
            solution = m.check()
            t2 = time()
            print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", solution)
            if solution==z3.sat:
                
                # print(len(m.model()))
                s = 0
                for line in m.model():
                    x2 = m.model()[line].as_fraction()
                    val = float(x2.numerator / x2.denominator)
                    if "input" in str(line) or "output" in str(line) or "sum" in str(line):
                        # print(line, "---> ", val)
                        continue
                    s = s+abs(val)
                sat_epsilon = s
                print("Threshold is: ", ep)
                print("Sat time: ", sat_epsilon)
                print("Total change is: ", s)
            else:
                unsat_epsilon = ep
                print("Unsat time: ", unsat_epsilon)
            ep = (sat_epsilon+unsat_epsilon)/2
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
    t3 = time()
    epsilon = classObject.calc(epsilon_max, tolerance)
    t4 = time()
    print("The total time taken was: ",(t4-t3)," seconds.\n The minimum change done was: ",epsilon)