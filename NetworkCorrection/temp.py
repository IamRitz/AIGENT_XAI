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

def loadModel():
    json_file = open('./Models/ACASXU_2_9.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Models/ACASXU_2_9.h5")
    return loaded_model


def create_marabou_equations():
    epsilon_max = 0.005
    inputs = genfromtxt('./data/inputs.csv', delimiter=',')
    input =inputs[0]
    model = loadModel()
    
    num_layers = 7
    l = 0
    var_counter = 10

    """
    Deciding a definition for required variables:
    How many variables do I need in total?  
        5 for input + (5*50) for w1 +
        50 for l2 + (50*50) for w2 +
        50 for l3 + (50*50) for w3 +
        50 for l4 + (50*50) for w4 +
        50 for l5 + (50*50) for w5 +
        50 for l6 + (50*50) for w6 +
        5 for l7 + (50*5) for w7.
    
    How will the variables be numbered?

    
    """
    input_vars = [i for i in range(5)]
    output_vars = [i for i in range(5,10)]
    print(input_vars)
    print(output_vars)
    neuron_vars = []    ### Each index of this array will be an array of length 2 which will contain the backward and forward value of the neurons.###
    weight_vars = []    ### Variables corresponding to the weights###
    epsilon_vars = []    ### Variables corresponding to the epsilons###

    """
    weight_vars is designed as:
    [
        [Weight variables for w2 i.e 5*50 variables]
        [Weight variables for w3 i.e 50*50 variables]
        [Weight variables for w4 i.e 50*50 variables]
        ...
        [Weight variables for w7 i.e 50*5 variables]
    ]
    """
    ### Begining to initialize an array to store the variables corresponding to weights. ###
    for layer in model.layers:
        if l==0:
            l = l + 1
            continue
        w = layer.get_weights()[0]
        print(np.array(w).shape)
        weights = np.zeros_like(w)
        for i in range(np.array(w).shape[0]):
            for j in range(np.array(w).shape[1]):
                weights[i][j] = var_counter
                var_counter = var_counter + 1
        weight_vars.append(weights)
    
    # print(weight_vars)
    ### Initializing weights variable array complete. ###

    ### Begining to initialize an array to store the variables corresponding to epsilons. ###
    l = 0
    for layer in model.layers:
        if l==0:
            l = l + 1
            continue
        w = layer.get_weights()[0]
        print(np.array(w).shape)
        epsilon = np.zeros_like(w)
        for i in range(np.array(w).shape[0]):
            for j in range(np.array(w).shape[1]):
                epsilon[i][j] = var_counter
                var_counter = var_counter + 1
        epsilon_vars.append(epsilon)
    
    # print(epsilon_vars)
    ### Initializing epsilon variable array complete. ###

    ### Begining to initialize an array to store the variables corresponding to neurons. ###
    l = 0
    for layer in model.layers:
        if l==0:
            l = l + 1
            continue
        w = layer.get_weights()[0]
        neuron_count = np.array(layer.get_weights()[0]).shape[0]
        print("Number of neurons is: ", neuron_count)
        n_var = []
        for n in range(neuron_count):
            n_var.append(var_counter)
            var_counter = var_counter + 1
        neuron_vars.append(n_var)
    
    # print(neuron_vars)
    ### Initializing neuron variable array complete. ###

    inputQuery = MarabouCore.InputQuery()
    inputQuery.setNumberOfVariables(var_counter+2000)
    ###Adding constraints for inputs###
    for i in range(len(input)):
        eq3 = MarabouCore.Equation(MarabouCore.Equation.EquationType.EQ)        ###The first step is to assign input values for each of the input variable.###
        eq3.addAddend(1, input_vars[i])
        eq3.setScalar(input[i])
        inputQuery.addEquation(eq3)
    ###Constraints for inputs added.###

    ###Adding constraints for outputs###
    e1 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
    e1.addAddend(1, 5)
    e1.addAddend(-1, 7)
    e1.setScalar(0)
    inputQuery.addEquation(e1)

    e2 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
    e2.addAddend(1, 5)
    e2.addAddend(-1, 8)
    e2.setScalar(0)
    inputQuery.addEquation(e2)

    e3 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
    e3.addAddend(1, 5)
    e3.addAddend(-1, 9)
    e3.setScalar(0)
    inputQuery.addEquation(e3)

    e4 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
    e4.addAddend(1, 6)
    e4.addAddend(-1, 7)
    e4.setScalar(0)
    inputQuery.addEquation(e4)

    e5 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
    e5.addAddend(1, 6)
    e5.addAddend(-1, 8)
    e5.setScalar(0)
    inputQuery.addEquation(e5)

    e6 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
    e6.addAddend(1, 6)
    e6.addAddend(-1, 9)
    e6.setScalar(0)
    inputQuery.addEquation(e6)
    ###Constraints for outputs added.###

    ###Adding constraints for edge weights.###
    l = -1
    for layer in model.layers:
        # print("New layer")
        l = l + 1
        if l==0:
            continue
        current_weight_vars = weight_vars[l-1]
        current_epsilon_vars = epsilon_vars[l-1]
        w = np.array(layer.get_weights()[0])
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                e7 = MarabouCore.Equation(MarabouCore.Equation.EquationType.LE)
                e7.addAddend(1, current_weight_vars[i][j])
                e7.addAddend(-1, current_epsilon_vars[i][j])
                e7.setScalar(w[i][j])
                inputQuery.addEquation(e7)

                e8 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
                e8.addAddend(1, current_weight_vars[i][j])
                e8.addAddend(1, current_epsilon_vars[i][j])
                e8.setScalar(w[i][j])
                inputQuery.addEquation(e8)
    ###Constraints for edge weights added.###

    ###Adding constraints for all epsilons.###
    e = MarabouCore.Equation(MarabouCore.Equation.EquationType.LE)
    for i in range(num_layers):
        current_epsilon_vars = epsilon_vars[i]
        for j in range(current_epsilon_vars.shape[0]):
            for k in range(current_epsilon_vars.shape[1]):
                e.addAddend(1, epsilon_vars[i][j][k])
    e.setScalar(epsilon_max)
    inputQuery.addEquation(e)
    ###Constraints for epsilons added.###

    ###Finally, calculating the actual output by finding constraints.###
    neurons = []
    l = 0
    # print(len(weight_vars))
    # print(len(model.layers))
    for layer in range(num_layers-1):
        current_weights = weight_vars[layer]
        prev_layer_neurons = neuron_vars[layer]
        next_layer_neurons = neuron_vars[layer+1]
        if layer==0:
            prev_layer_neurons = input_vars
        
        current_bias = model.layers[layer+1].get_weights()[1]
        print(current_weights.shape)
        print(len(current_bias))
        # Shape of X is: 1, len(prev_layer_neurons)
        # Shape of Y is: current_weights.shape[0], current_weights.shape[1]
        # Shape of result will be: 1, current_weights.shape[1]
        temp_vars = []
        for i in range(current_weights.shape[1]):
            temp_vars[i] = var_counter
            var_counter = var_counter + 1

        for i in range(len(temp_vars)):
            e = MarabouCore.Equation(MarabouCore.Equation.EquationType.EQ)
            e.addAddend(-1, temp_vars[i])
            e.setScalar(-1*(current_bias[i]))

            for j in range(current_weights.shape[1]):
                # iterate through rows of Y
                for k in range(current_weights.shape[0]):
                    #Multiply prev_layer_neurons[k] with current_weights[k][j]
                    e.addAddend(1, current_weights[k][j])
                    # result[i][j] += X[i][k] * Y[k][j]
            
            
            inputQuery.addEquation(e)
            MarabouCore.addReluConstraint(inputQuery, temp_vars[i], next_layer_neurons[i])

            # e10 = MarabouCore.Equation(MarabouCore.Equation.EquationType.GE)
            # e10.addAddend(1, next_layer_neurons[i])
            # e10.setScalar(0)
        # w = layer.get_weights()[0]
        # b = layer.get_weights()[1]
        # ep = np.zeros_like(w)
        # for row in range(0,len(w)):
        #     for col in range(0,len(w[0])):
        #         # print(row," ",col, " ", ep[row][col], " ", w[row][col])
        #         inputQuery.setLowerBound(ep[row][col], w[row][col]-epsilon_max)
        #         inputQuery.setUpperBound(ep[row][col], w[row][col]+epsilon_max)
        # # print(w)
        # result = np.matmul(input,ep)+b
        # l = l + 1
        # if l == num_layers:
        #     input = result
        #     neurons.append(input)
        #     continue
        # input = [r for r in result]
        # for i in range(0, len(result)):
        #     inputQuery.setLowerBound(input[i], 0)
        # neurons.append(input)
        
    
    t1=time()
    options = createOptions()
    Marabou_result = MarabouCore.solve(inputQuery, options, "")
    t2=time()
    print("Time taken in solving the query is: ",(t2-t1)," seconds. \nThe query was: ", Marabou_result[0])
    return Marabou_result
    
create_marabou_equations()

# filename = "./ProtobufNetworks/ACASXU_2_9.pb"
# network = Marabou.read_tf(filename)
# print("NETWORK: ", network)

# inputVars = network.inputVars[0][0]
# outputVars = network.outputVars
# # print(network.reluList)

# inputs = genfromtxt('./data/inputs.csv', delimiter=',')
# outputs = genfromtxt('./data/outputs.csv', delimiter=',')

# correct_diff = 0.001

# # for i in range(5):
# #     network.addEquality([inputVars[i]], [1.0], inputs[0][i])

# # for i in range(outputVars.shape[0]):
# #     network.addEquality([outputVars[i][0]], [1.0], outputs[0][0])
# #     network.addEquality([outputVars[i][1]], [1.0], outputs[0][1])
# #     network.addEquality([outputVars[i][2]], [1.0], outputs[0][2])
# #     network.addEquality([outputVars[i][3]], [1.0], outputs[0][3])
# #     network.addEquality([outputVars[i][4]], [1.0], outputs[0][4])
    
# print(network.evaluateWithoutMarabou(inputValues=[inputs[0]]))

# # for i in range(outputVars.shape[0]):
# #     network.addInequality([outputVars[i][0], outputVars[i][2]], [1, -1], correct_diff)
# #     network.addInequality([outputVars[i][0], outputVars[i][3]], [1, -1], correct_diff)
# #     network.addInequality([outputVars[i][0], outputVars[i][4]], [1, -1], correct_diff)
# #     network.addInequality([outputVars[i][1], outputVars[i][2]], [1, -1], correct_diff)
# #     network.addInequality([outputVars[i][1], outputVars[i][3]], [1, -1], correct_diff)
# #     network.addInequality([outputVars[i][1], outputVars[i][4]], [1, -1], correct_diff)

# # Constraints added for inputs and outputs of the DNN
# # What is left? Adding contraints for the intermediate layers


# varMap = network.varMap      #Corresponds to the neuron values in each layer.\
# # Like network.inputs gives the input values, network.varMap can be used to set constraints for the neurons of any layer.
# # print(type(varMap))      #The varMap is of dictionary type.
# """"
# So, these are in pairs of three i.e the first three rows of type_vars correspond to the first layer i.e input layer and so on.
# But, I have to put constraints on the edge weights and not the neuron values.
# """

# constantMap = network.constantMap     #Corresponds to the edge weights of each layer.\
# # Like network.inputs gives the input values, network.constantMap can be used to set constraints for the edge weights of any layer.

# # print(type(constantMap))
# # print(len(constantMap))

# for x in constantMap.values():
#     # print(type(x[0][0]))
#     # # network.addInequality([x[0][0]], [1], 0.005)
#     # # network.addEquality([x[0][0]], [1.0], 1)
#     # print()
#     break

# # print(constantMap)

# # print(type(outputVars[0][0]))
# # vals = network.solve(verbose=True)
# # print("Done Marabou work.")
# # print(vals)
# inp = np.array([inputs[0]])
# # print(type(inp))
# # print(inp.shape)
# # n1 = read_tf_weights_as_var(filename=filename, inputVals=inp)



# def abc():
#         model_name = 'ACASXU_2_9'
#         orig_model_name = 'ACASXU_2_9'
#         lastlayer_inputs = np.load('./data/{}.lastlayer.input.npy'.format(orig_model_name))
#         filename = './ProtobufNetworks/last.layer.{}.pb'.format(model_name)
#         num = 1
#         lastlayer_inputs = lastlayer_inputs[:num]
#         print(len(lastlayer_inputs))
#         # print(lastlayer_inputs)
#         print("Last layer: ", type(lastlayer_inputs))
#         print(lastlayer_inputs.shape)
#         network = read_tf_weights_as_var(filename=filename, inputVals=lastlayer_inputs)

# # abc()