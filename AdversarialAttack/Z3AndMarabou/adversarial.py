import sys
from time import time
import keras
from numpy import genfromtxt, safe_eval, var
sys.path.append('../')
import numpy as np
from maraboupy import Marabou
import numpy as np
from MarabouNetworkCustom import read_tf_weights_as_var
from maraboupy import MarabouCore, MarabouUtils
from maraboupy.Marabou import createOptions
import maraboupy

def loadModel():
    json_file = open('./Models/ACASXU_2_9.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Models/ACASXU_2_9.h5")
    return loaded_model

def getInputs():
    inp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921569, 0.011764706, 0.16470589, 0.4627451, 0.75686276, 0.4627451, 0.4627451, 0.23921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05490196, 0.7019608, 0.9607843, 0.9254902, 0.9490196, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.9607843, 0.92156863, 0.32941177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5921569, 0.99607843, 0.99607843, 0.99607843, 0.8352941, 0.7529412, 0.69803923, 0.69803923, 0.7058824, 0.99607843, 0.99607843, 0.94509804, 0.18039216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16862746, 0.92156863, 0.99607843, 0.8862745, 0.2509804, 0.10980392, 0.047058824, 0.0, 0.0, 0.007843138, 0.5019608, 0.9882353, 1.0, 0.6784314, 0.06666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21960784, 0.99607843, 0.99215686, 0.41960785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5254902, 0.98039216, 0.99607843, 0.29411766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24705882, 0.99607843, 0.61960787, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8666667, 0.99607843, 0.6156863, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7607843, 0.99607843, 0.40392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5882353, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13333334, 0.8627451, 0.9372549, 0.22745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411765, 0.99607843, 0.67058825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.9372549, 0.23529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043137256, 0.85882354, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38431373, 0.99607843, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.63529414, 0.99607843, 0.81960785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38431373, 0.99607843, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.93333334, 0.99607843, 0.29411766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38431373, 0.99607843, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.64705884, 0.99607843, 0.7647059, 0.015686275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25882354, 0.94509804, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.654902, 0.99607843, 0.8901961, 0.21568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.8352941, 0.078431375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18039216, 0.59607846, 0.7921569, 0.99607843, 0.99607843, 0.24705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.99607843, 0.8, 0.7058824, 0.7058824, 0.7058824, 0.7058824, 0.7058824, 0.92156863, 0.99607843, 0.99607843, 0.91764706, 0.6117647, 0.039215688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31764707, 0.8039216, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.9882353, 0.91764706, 0.47058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.101960786, 0.8235294, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.6, 0.40784314, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return inp

def callMarabou(epsilon_max):
    filename = "./frozen_graph.pb"
    network = Marabou.read_tf(filename)

    
    correct_diff = 0.001

    inputVars = network.inputVars[0][0]
    outputVars = network.outputVars

    inputs = genfromtxt('./data/inputs.csv', delimiter=',')
    inp = getInputs()

    epsilons = []
    for i in range(len(inp)):
        epsilons.append(i+1)

    e = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
    e2 = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.GE)
    for i in range(len(inp)):
        network.addInequality([inputVars[i], epsilons[i]], [-1, -1], -inp[i])
        network.addInequality([inputVars[i], epsilons[i]], [1, -1], inp[i])
        network.setLowerBound(epsilons[i], -epsilon_max)
        network.setUpperBound(epsilons[i], epsilon_max)
        e.addAddend(1, epsilons[i])
        e2.addAddend(1, epsilons[i])
        # network.setLowerBound(inputVars[i], inp[i])
        # network.setUpperBound(inputVars[i], inp[i])
    e.setScalar(epsilon_max)
    e2.setScalar(-epsilon_max)
    # network.addEquation(e)
    network.addEquation(e2)

    # print("Type:",e.EquationType)

    true_label = 0

    for i in range(outputVars.shape[0]):
        # network.addInequality([outputVars[i][0], outputVars[i][2]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][3]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][4]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][1], outputVars[i][2]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][1], outputVars[i][3]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][1], outputVars[i][4]], [-1, 1], -correct_diff)
        for j in range(10):
            if j==true_label:
                continue
            e3 = MarabouUtils.Equation(EquationType=MarabouUtils.MarabouCore.Equation.LE)
            e3.addAddend(-1, outputVars[i][j])
            e3.addAddend(1, outputVars[i][true_label])
            e3.setScalar(0.001)
            network.addEquation(e3)

        # network.addInequality([outputVars[i][0], outputVars[i][1]], [1, -1], correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][2]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][3]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][4]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][5]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][6]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][7]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][8]], [-1, 1], -correct_diff)
        # network.addInequality([outputVars[i][0], outputVars[i][9]], [-1, 1], -correct_diff)

    # Call to C++ Marabou solver
    vals = network.solve("abc.txt",verbose=False)
    print("Output was: ",vals[0])
    print("Printing values:")
    # print(vals)
    print(len(vals[1]))
    s = 0
    if len(vals[1])==0:
        return 0, vals[0]
    for i in range(784):
        # print(i,"-->",vals[1][i])
        s = s + abs(vals[1][i]-inp[i])
    print("Sum was:", s)
    return s, vals[0]

epsilon_max = 10
sum, result = callMarabou(epsilon_max)

for i in range(10):
    print("Iteration count:", i, "Calling with epsilon:", epsilon_max)
    curr_sum, curr_result = callMarabou(epsilon_max)
    print(curr_sum,"   ", curr_result)
    if "unsat" not in curr_result:
        if curr_sum<=sum:
            sum = curr_sum
            epsilon_max = epsilon_max/2
        else:
            epsilon_max = epsilon_max*2/3
    else:
        epsilon_max = epsilon_max*5
    print("New epsilon_max:", epsilon_max)

    print()
