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


def example():
    x = z3.Real('x')
    y = z3.Real('y')
    s = z3.Solver()
    s.add(x + y > 5, x > 1, y > 1)
    print(s.check())
    print(s.model())

def loadModel():
    json_file = open('./Models/ACASXU_2_9.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Models/ACASXU_2_9.h5")
    return loaded_model

def ReLU(input):
    return np.vectorize(lambda y: z3.If(y>=0, y, z3.RealVal(0)))(input)

def result(input, m):
    epsilon_max = 0.005
    model = loadModel()
    weights = model.get_weights()
    layer_output = 0
    for i in range(0,len(weights)-1,2):
        w = weights[i]
        b = weights[i+1]
        # print(w,"\n",w.T,"\n",b)
        # print(type(w))
        # print((w.T).shape)
        shape0 = (w.T).shape[0]
        shape1 = (w.T).shape[1]
        epsilon = []
        for row in range(shape0):
            ep = []
            for col in range(shape1):
                ep.append(z3.Real('e'+str(row)+str(col)))
                m.add(ep[col]>=-epsilon_max)
                m.add(ep[col]<=epsilon_max)
            epsilon.append(ep)
        """
        Create an epsilon array at every stage which should be added to the weight array and put constraints on it.
        """
        # A = Array('A', IntSort(), ArraySort(IntSort(), IntSort()))
        out = (w.T+epsilon) @ input + b
        layer_output = ReLU(out)
        input = layer_output
    return layer_output

    # w1, b1, w2, b2, w1, b1, w2, b2
def callZ3():
    inputs = genfromtxt('./data/inputs.csv', delimiter=',')
    input =inputs[0]
    input_vars = z3.RealVector('input_vars',5)
    m = z3.Solver()
    ###Adding constraints for inputs###
    for i in range(len(input)):
        # print("Input:",i)
        m.add(input_vars[i]==input[i])
    ###Constraints for inputs added.###
    r = result(input_vars, m)
    print(len(r))
    m.add(r[0]>=r[2])
    m.add(r[0]>=r[3])
    m.add(r[0]>=r[4])
    m.add(r[1]>=r[2])
    m.add(r[1]>=r[3])
    m.add(r[1]>=r[4])

    t1 = time()
    solution = m.check()
    t2 = time()
    print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", solution)
    print("The model is: ", m.model())

callZ3()

# def create_marabou_equations():
#     epsilon_max = 0.005
#     inputs = genfromtxt('./data/inputs.csv', delimiter=',')
#     input =inputs[0]
#     model = loadModel()
    
#     num_layers = 7
#     l = 0
#     var_counter = 10

#     """
#     Deciding a definition for required variables:
#     How many variables do I need in total?  
#         5 for input + (5*50) for w1 +
#         50 for l2 + (50*50) for w2 +
#         50 for l3 + (50*50) for w3 +
#         50 for l4 + (50*50) for w4 +
#         50 for l5 + (50*50) for w5 +
#         50 for l6 + (50*50) for w6 +
#         5 for l7 + (50*5) for w7.
    
#     How will the variables be numbered?

    
#     """
#     input_vars = z3.RealVector('input_vars',5)
#     output_vars = z3.RealVector('output_vars',5)
#     # print(input_vars)
#     # print(output_vars)
#     neuron_vars = []    ### Each index of this array will contain the forward value of the neurons.###
#     weight_vars = []    ### Variables corresponding to the weights###
#     epsilon_vars = []    ### Variables corresponding to the epsilons###

#     """
#     weight_vars is designed as:
#     [
#         [Weight variables for w2 i.e 5*50 variables]
#         [Weight variables for w3 i.e 50*50 variables]
#         [Weight variables for w4 i.e 50*50 variables]
#         ...
#         [Weight variables for w7 i.e 50*5 variables]
#     ]
#     """
#     ### Begining to initialize an array to store the variables corresponding to weights. ###
#     for layer in model.layers:
#         if l==0:
#             l = l + 1
#             continue
#         w = layer.get_weights()[0]
#         print(np.array(w).shape)
#         weights = np.zeros_like(w)
#         # for i in range(np.array(w).shape[0]):
#         #     for j in range(np.array(w).shape[1]):
#         #         weights[i][j] = var_counter
#         #         var_counter = var_counter + 1
#         weight_vars.append(weights)
    
#     # print(weight_vars)
#     ### Initializing weights variable array complete. ###

#     ### Begining to initialize an array to store the variables corresponding to epsilons. ###
#     l = 0
#     for layer in model.layers:
#         if l==0:
#             l = l + 1
#             continue
#         w = layer.get_weights()[0]
#         print(np.array(w).shape)
#         epsilon = np.zeros_like(w)
#         # for i in range(np.array(w).shape[0]):
#         #     for j in range(np.array(w).shape[1]):
#         #         epsilon[i][j] = var_counter
#         #         var_counter = var_counter + 1
#         epsilon_vars.append(epsilon)
    
#     # print(epsilon_vars)
#     ### Initializing epsilon variable array complete. ###

#     ### Begining to initialize an array to store the variables corresponding to neurons. ###
#     l = 0
#     for layer in model.layers:
#         if l==0:
#             l = l + 1
#             continue
#         w = layer.get_weights()[0]
#         neuron_count = np.array(layer.get_weights()[0]).shape[0]
#         # print("Number of neurons is: ", neuron_count)
#         n_var = [0]*neuron_count
#         # for n in range(neuron_count):
#         #     n_var.append(0)
#         #     var_counter = var_counter + 1
#         neuron_vars.append(n_var)
    
#     # print(neuron_vars)
#     ### Initializing neuron variable array complete. ###

#     m = z3.Solver()
#     ###Adding constraints for inputs###
#     for i in range(len(input)):
#         # print("Input:",i)
#         m.add(input_vars[i]==input[i])
#     ###Constraints for inputs added.###

#     ###Adding constraints for outputs###
#     m.add(output_vars[0]>=output_vars[2])
#     m.add(output_vars[0]>=output_vars[3])
#     m.add(output_vars[0]>=output_vars[4])
#     m.add(output_vars[1]>=output_vars[2])
#     m.add(output_vars[1]>=output_vars[3])
#     m.add(output_vars[1]>=output_vars[4])
#     ###Constraints for outputs added.###

#     ###Adding constraints for edge weights.###
#     l = -1
#     for layer in model.layers:
#         # print("New layer")
#         l = l + 1
#         if l==0:
#             continue
#         current_weight_vars = weight_vars[l-1]
#         current_epsilon_vars = epsilon_vars[l-1]
#         w = np.array(layer.get_weights()[0])
#         for i in range(w.shape[0]):
#             for j in range(w.shape[1]):
#                 m.add(current_weight_vars[i][j]<=(current_epsilon_vars[i][j]+w[i][j]))

#                 m.add(current_weight_vars[i][j]>=(-current_epsilon_vars[i][j]+w[i][j]))
#     ###Constraints for edge weights added.###

#     ###Adding constraints for all epsilons.###
#     for i in range(num_layers):
#         current_epsilon_vars = epsilon_vars[i]
#         for j in range(current_epsilon_vars.shape[0]):
#             for k in range(current_epsilon_vars.shape[1]):
#                 # e.addAddend(1, epsilon_vars[i][j][k])
#                 m.add(epsilon_vars[i][j][k]>=-epsilon_max)
#                 m.add(epsilon_vars[i][j][k]<=epsilon_max)
#     # e.setScalar(epsilon_max)
#     # inputQuery.addEquation(e)
#     ###Constraints for epsilons added.###

#     ###Finally, calculating the actual output by finding constraints.###
#     neurons = []
#     l = 0
#     # print(len(weight_vars))
#     # print(len(model.layers))
#     for layer in range(num_layers):
#         current_weights = weight_vars[layer]
#         prev_layer_neurons = neuron_vars[layer]
#         next_layer_neurons = output_vars
#         if layer!=6:
#             next_layer_neurons = neuron_vars[layer+1]
#         if layer==0:
#             prev_layer_neurons = input_vars
        
#         current_bias = model.layers[layer+1].get_weights()[1]
#         print(current_weights.shape)
#         print(len(current_bias))
#         # Shape of X is: 1, len(prev_layer_neurons)
#         # Shape of Y is: current_weights.shape[0], current_weights.shape[1]
#         # Shape of result will be: 1, current_weights.shape[1]
#         temp_vars = [0]*current_weights.shape[1]

#         for i in range(len(temp_vars)):
#             e = MarabouCore.Equation(MarabouCore.Equation.EquationType.LE)
#             e.addAddend(-1, temp_vars[i])
            
#             e.setScalar(-1*(current_bias[i]))

#             for j in range(current_weights.shape[1]):
#                 # iterate through rows of Y
#                 for k in range(current_weights.shape[0]):
#                     #Multiply prev_layer_neurons[k] with current_weights[k][j]
#                     s.add(temp_vars[i]==temp_vars+)
#                     if current_weights[k][j]>=count or prev_layer_neurons[k]>=count:
#                         print("Greater.")
#                     e.addAddend(prev_layer_neurons[k], current_weights[k][j])
#                     # result[i][j] += X[i][k] * Y[k][j]
            
            
#             inputQuery.addEquation(e)
#             MarabouCore.addReluConstraint(inputQuery, temp_vars[i], next_layer_neurons[i])

#             # e10 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
#             # e10.addAddend(1, next_layer_neurons[i])
#             # e10.setScalar(0)     
#     t1=time()
#     options = createOptions(verbosity=2)
#     Marabou_result = MarabouCore.solve(inputQuery, options, "abcdef.txt")
#     t2=time()
#     print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", Marabou_result)
#     print("Var_count is: ", var_counter)
#     return Marabou_result
    
# create_marabou_equations()

