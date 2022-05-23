import sys
from time import time

from numpy import safe_eval
sys.path.append('../')
import numpy as np
import argparse
from maraboupy import MarabouUtils
from maraboupy import MarabouCore
from maraboupy import Marabou, MarabouNetwork
from gurobipy import GRB
from gurobipy import *
from gurobipy import Model
from WatermarkRemoval.MarabouNetworkWeightsVars import read_tf_weights_as_var
from functools import reduce
# from gurobipy import *
from copy import deepcopy
from pprint import pprint
from maraboupy.Marabou import createOptions

sat = 'SAT'
unsat = 'UNSAT'
class findCorrection:

    def __init__(self, epsilon_max, epsilon_interval, correct_diff, lp):
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval
        self.correct_diff = correct_diff
        self.lp = lp

    def getNetworkSolution(self, network):
        equations = network.equList
        numOfVar = network.numVars
        networkEpsilons = network.epsilons
        epsilonsShape = networkEpsilons.shape 
        model = Model("my model")
        modelVars = model.addVars(numOfVar, lb=-GRB.INFINITY, ub=GRB.INFINITY)
        epsilon = model.addVar(name="epsilon")
        model.setObjective(epsilon, GRB.MINIMIZE)
        for i in range(epsilonsShape[0]):
            for j in range(epsilonsShape[1]):
                model.addConstr(modelVars[networkEpsilons[i][j]], GRB.LESS_EQUAL, epsilon)
                model.addConstr(modelVars[networkEpsilons[i][j]], GRB.GREATER_EQUAL, -1*epsilon)

        for eq in equations:
            addends = map(lambda addend: modelVars[addend[1]] * addend[0], eq.addendList)
            eq_left = reduce(lambda x,y: x+y, addends)
            if eq.EquationType == MarabouCore.Equation.EQ:
                model.addConstr(eq_left, GRB.EQUAL, eq.scalar)
            if eq.EquationType == MarabouCore.Equation.LE:
                model.addConstr(eq_left, GRB.LESS_EQUAL, eq.scalar)
            if eq.EquationType == MarabouCore.Equation.GE:
                model.addConstr(eq_left, GRB.GREATER_EQUAL, eq.scalar)
                
        model.optimize()
        # epsilons_vals = np.array([[modelVars[networkEpsilons[i][j]].x for j in range(epsilonsShape[1])] for i in range(epsilonsShape[0])])
        all_vals = np.array([modelVars[i].x for i in range(numOfVar)])
        return epsilon.x, epsilon.x, all_vals 

    def findEpsilon(self, network):
        outputVars = network.outputVars
        
        for i in range(outputVars.shape[0]):
            MarabouNetwork.MarabouNetwork.addInequality(network, [outputVars[i][0], outputVars[i][2]], [1, -1], self.correct_diff)
            MarabouNetwork.MarabouNetwork.addInequality(network, [outputVars[i][0], outputVars[i][3]], [1, -1], self.correct_diff)
            MarabouNetwork.MarabouNetwork.addInequality(network, [outputVars[i][0], outputVars[i][4]], [1, -1], self.correct_diff)
            MarabouNetwork.MarabouNetwork.addInequality(network, [outputVars[i][1], outputVars[i][2]], [1, -1], self.correct_diff)
            MarabouNetwork.MarabouNetwork.addInequality(network, [outputVars[i][1], outputVars[i][3]], [1, -1], self.correct_diff)
            MarabouNetwork.MarabouNetwork.addInequality(network, [outputVars[i][1], outputVars[i][4]], [1, -1], self.correct_diff)
                
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][2]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][3]], [1, -1], self.correct_diff)
            # MarabouUtils.addInequality(network, [outputVars[i][outputNum], outputVars[i][4]], [1, -1], self.correct_diff)
        return self.getNetworkSolution(network)

    def epsilonABS(self, network, epsilon_var):
        epsilon2 = network.getNewVariable()
        MarabouNetwork.MarabouNetwork.addEquality(network, [epsilon2, epsilon_var], [1, -2], 0)
        
        relu_epsilon2 = network.getNewVariable()
        network.addRelu(epsilon2, relu_epsilon2)
        
        abs_epsilon = network.getNewVariable()
        MarabouNetwork.MarabouNetwork.addEquality(network, [abs_epsilon, relu_epsilon2, epsilon_var], [1, -1, 1], 0)
        return abs_epsilon

    def evaluateEpsilon(self, epsilon):
        inputQuery = MarabouCore.InputQuery()
        inputQuery.setNumberOfVariables(8)

        inputQuery.setLowerBound(4, 0)
        inputQuery.setUpperBound(4, epsilon)

        inputQuery.setLowerBound(5, -epsilon)
        inputQuery.setUpperBound(5, 0)

        inputQuery.setLowerBound(6, 0)
        inputQuery.setUpperBound(6, epsilon)

        inputQuery.setLowerBound(7, -epsilon)
        inputQuery.setUpperBound(7, 0)


        eq3 = MarabouCore.Equation()#For v31
        eq3.addAddend(-1, 2)
        eq3.addAddend(0, 4)
        eq3.addAddend(2, 6)
        eq3.setScalar(2)
        inputQuery.addEquation(eq3)

        eq3 = MarabouCore.Equation()#For v32
        eq3.addAddend(-1, 3)
        eq3.addAddend(0, 5)
        eq3.addAddend(2, 7)
        eq3.setScalar(-2)
        inputQuery.addEquation(eq3)
                
        e = MarabouCore.Equation(MarabouCore.Equation.EquationType.LE)
        e.addAddend(-1, 2)
        e.addAddend(1, 3)
        e.setScalar(0.0001)
        inputQuery.addEquation(e)


        # Run Marabou to solve the query
        # This should return "sat"
        options = createOptions()
        t1=time()
        vars, stats,_ = MarabouCore.solve(inputQuery, options, "")
        t2=time()
        if len(vars) > 0:
            print("SAT")
            print(vars, stats)
            print("Time taken:", t2-t1)
            return vars, stats, t2-t1
        else:
            print("UNSAT")
            return vars, stats, t2-t1

    def findEpsilonInterval(self):
        sat_epsilon = self.epsilon_max
        unsat_epsilon = 0.0
        sat_vals = None
        epsilon = sat_epsilon
        # print(lastLayer)
        satTimes=[]
        unsatTimes=[]
        while abs(sat_epsilon - unsat_epsilon) > self.epsilon_interval:
            status, vals, t = self.evaluateEpsilon(epsilon)
            # print(status)
            # print(vals)
            if status == "sat":
                # print("Changing sat vals")
                sat_epsilon = epsilon
                sat_vals = vals
                satTimes.append(t)
            else:
                unsat_epsilon = epsilon
                unsatTimes.append(t)
            epsilon = (sat_epsilon + unsat_epsilon)/2
            # print("Epsilon is: ", epsilon)
        # print(sat_vals)
        print("Times are:")
        print(satTimes)
        print(unsatTimes)
        print()
        return unsat_epsilon, sat_epsilon , sat_vals

    def run(self, num):
        unsat_epsilon, sat_epsilon, sat_vals = self.findEpsilon() if self.lp else self.findEpsilonInterval()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', default='ACASXU_2_9_0_corrected', help='the name of the model')
    parser.add_argument('--input_num', default=1, help='the name of the model')
    parser.add_argument('--correct_diff', default=0.001, help='the input to correct')
    parser.add_argument('--epsilon_max', default=5, help='max epsilon value')
    parser.add_argument('--epsilon_interval', default=0.0001, help='epsilon smallest change')
    parser.add_argument('--lp', action='store_true', help='solve lp')
    
    args = parser.parse_args()
    epsilon_max = float(args.epsilon_max)
    epsilon_interval = float(args.epsilon_interval)  
    correct_diff = - float(args.correct_diff)  
    input_num = int(args.input_num)  
    
    # model_name = args.model
    # MODELS_PATH = './Models'
    problem = findCorrection(epsilon_max, epsilon_interval, correct_diff, args.lp)
    problem.run(input_num)