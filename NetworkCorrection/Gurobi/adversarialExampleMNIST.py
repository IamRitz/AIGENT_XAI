import gurobipy as gp
import gurobipy as gp
from gurobipy import GRB
from numpy import genfromtxt
import keras
import numpy as np
import sys
# sys.path.append('../')
import tensorflow as tf
import numpy as np
import gurobipy as grb
import os
from relumip import AnnModel

from relumip.utils.visualization import plot_results_2d


def loadModel():
    json_file = open('./Models/ACASXU_2_9.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../Models/ACASXU_2_9.h5")
    return loaded_model

def getInputs():
    inputs = genfromtxt('./data/inputs.csv', delimiter=',')
    return inputs[0]

def getmnist():
    inp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.23921569, 0.011764706, 0.16470589, 0.4627451, 0.75686276, 0.4627451, 0.4627451, 0.23921569, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05490196, 0.7019608, 0.9607843, 0.9254902, 0.9490196, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.9607843, 0.92156863, 0.32941177, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5921569, 0.99607843, 0.99607843, 0.99607843, 0.8352941, 0.7529412, 0.69803923, 0.69803923, 0.7058824, 0.99607843, 0.99607843, 0.94509804, 0.18039216, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16862746, 0.92156863, 0.99607843, 0.8862745, 0.2509804, 0.10980392, 0.047058824, 0.0, 0.0, 0.007843138, 0.5019608, 0.9882353, 1.0, 0.6784314, 0.06666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.21960784, 0.99607843, 0.99215686, 0.41960785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5254902, 0.98039216, 0.99607843, 0.29411766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.24705882, 0.99607843, 0.61960787, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8666667, 0.99607843, 0.6156863, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7607843, 0.99607843, 0.40392157, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5882353, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.13333334, 0.8627451, 0.9372549, 0.22745098, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49411765, 0.99607843, 0.67058825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.9372549, 0.23529412, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.043137256, 0.85882354, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32941177, 0.99607843, 0.8352941, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38431373, 0.99607843, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.63529414, 0.99607843, 0.81960785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38431373, 0.99607843, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.93333334, 0.99607843, 0.29411766, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.38431373, 0.99607843, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.64705884, 0.99607843, 0.7647059, 0.015686275, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25882354, 0.94509804, 0.78039217, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.011764706, 0.654902, 0.99607843, 0.8901961, 0.21568628, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.8352941, 0.078431375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18039216, 0.59607846, 0.7921569, 0.99607843, 0.99607843, 0.24705882, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8392157, 0.99607843, 0.8, 0.7058824, 0.7058824, 0.7058824, 0.7058824, 0.7058824, 0.92156863, 0.99607843, 0.99607843, 0.91764706, 0.6117647, 0.039215688, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31764707, 0.8039216, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.9882353, 0.91764706, 0.47058824, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.101960786, 0.8235294, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.99607843, 0.6, 0.40784314, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return inp

def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 0
        for layer in loaded_model.layers:
            # if l==0:
            #     l = l + 1
            #     continue
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # print(w)
            result = np.matmul(input,w)+b
            
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        # print(len(neurons))
        # print(neurons[len(neurons)-1])
        return neurons

def get_neuron_values(loaded_model, input, num_layers, values, gurobi_model):
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
            input = []
            for r in range(len(result)):
                if values[l][r]>0: 
                    """ 
                    For ann keep: values[l-1][r], For find: values[l][r]
                    """
                    input.append(result[r])
                else:
                    input.append(0)
            neurons.append(input)
            l = l + 1
        return neurons[len(neurons)-1]

def find(epsilon1, epsilon2, model, inp, true_label, num_inputs, num_outputs):
    num_layers = len(model.layers)
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model("Model", env=env)
    ep = []
    input_vars = []
    try:
        epsilon_max = m.addVar(lb=0,ub=epsilon1,vtype=GRB.CONTINUOUS, name="epsilon_max")

        for i in range(num_inputs):
            ep.append(m.addVar(vtype=grb.GRB.CONTINUOUS))
            m.addConstr(ep[i]-epsilon_max<=0)
            m.addConstr(-ep[i]<=0)
        # input_vars = [ep[0,i]+inp[i] for i in range(len(inp))]
        
        neurons = get_neuron_values_actual(model, inp, num_layers)

        for i in range(num_inputs):
            input_vars.append(m.addVar(inp[i]-epsilon1, inp[i]+epsilon1, vtype=grb.GRB.CONTINUOUS))
            m.addConstr(input_vars[i]+ep[i]>=inp[i])
            m.addConstr(input_vars[i]-ep[i]<=inp[i])

        result = get_neuron_values(model, input_vars, num_layers, neurons, m)
        m.update()
        # print(result)
        # print(type(result))
        for i in range(num_outputs):
            if i==true_label:
                continue
            m.addConstr(result[i]-result[true_label]>=0.001)
        
        m.update()
        expr = grb.quicksum(ep)
        # print("Expr:", expr)
        epsilon_max_2 = m.addVar(lb=0,ub=epsilon2,vtype=GRB.CONTINUOUS, name="epsilon_max_2")
        m.addConstr(expr+epsilon_max_2>=0)
        m.addConstr(expr-epsilon_max_2<=0)
        m.update()
        m.setObjective(epsilon_max+epsilon_max_2, GRB.MINIMIZE)
        # m.setObjectiveN(epsilon_max_2, GRB.MINIMIZE, 1)

        m.optimize()
        if m.Status == GRB.INFEASIBLE:
            # print("returning")
            return [], 2
        summation = 0
        # print(ep)
        c = 0
        final_change = []
        for i in range(len(ep)):
            summation = summation + ep[i].X
            final_change.append(ep[i].X)
            if ep[i].X>0:
                # print(i, ep[i].X)
                c = c + 1
        # print("Effective change was: ", summation)
        # print("The number of weights changed were: ",c)
        # print("Query has: ", m.NumObj, " objectives.")
        # print(m.getVarByName("epsilon_max"))
        # print(m.getVarByName("epsilon_max_2"))
        return final_change, 1
    except:
        return [], 0

def ann(epsilon, tf_model, inp, true_label, num_inputs, num_outputs):
    """
    This function is used to generate adversarial examples.
    """
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    opt_model = gp.Model("Model", env=env)

    input_vars = []
    epsilons = []
    output_vars = []

    epsilon_max = opt_model.addVar(lb=0,ub=epsilon,vtype=GRB.CONTINUOUS, name="epsilon_max")
    # epsilon_max = epsilon
    for i in range(num_inputs):
        epsilons.append(opt_model.addVar(vtype=grb.GRB.CONTINUOUS))
        opt_model.addConstr(epsilons[i]-epsilon_max<=0)
        opt_model.addConstr(-epsilons[i]<=0)
    
    for i in range(num_inputs):
        input_vars.append(opt_model.addVar(inp[i]-epsilon, inp[i]+epsilon, vtype=grb.GRB.CONTINUOUS))
        opt_model.addConstr(input_vars[i]+epsilons[i]>=inp[i])
        opt_model.addConstr(input_vars[i]-epsilons[i]<=inp[i])

    for i in range(num_outputs):
        output_vars.append(opt_model.addVar(-grb.GRB.INFINITY, grb.GRB.INFINITY, vtype=grb.GRB.CONTINUOUS))
    
    opt_model.update()
    expr = grb.quicksum(epsilons)
    # print("Expr:", expr)
    opt_model.addConstr(expr+epsilon_max>=0)
    opt_model.addConstr(expr-epsilon_max<=0)

    for i in range(num_outputs):
        if i==true_label:
            continue
        opt_model.addConstr(output_vars[i]-output_vars[true_label]>=0.001)
    
    opt_model.update()

    ann_model = AnnModel(tf_model=tf_model, modeling_language='GUROBI')
    ann_model.connect_network_input(opt_model, input_vars)
    ann_model.connect_network_output(opt_model, output_vars)

    ann_model.embed_network_formulation(bound_tightening_strategy='LP')

    opt_model.setObjective(epsilon_max, grb.GRB.MINIMIZE)
    opt_model.optimize()

    # sample_input = inp
    # sample_output = tf_model.predict([sample_input])

    # print(sample_output)
    summation = 0
    for v in opt_model.getVars():
            # print('%s %g' % (v.VarName, v.X))
            if "C" in v.VarName:
                index = int(v.VarName[1:])
                if index>=num_inputs and index<num_inputs*2:
                    # print(v.VarName, v.X, inp[index-num_inputs])
                    summation = summation + float(v.X) - float(inp[index-num_inputs])
                # else:
                #     print(v.VarName, v.X)

    print("Query has: ", opt_model.NumObj, " objectives.")
    print(opt_model.getVarByName("epsilon_max"))
    print("Effective change was: ", summation)

def example():
    try:

        # Create a new model
        m = gp.Model("myModel")
        m.addVar()

        # Create variables
        x = m.addVar(lb=0.1,ub=0.5,vtype=GRB.CONTINUOUS, name="x")
        y = m.addVar(vtype=GRB.CONTINUOUS, name="y")
        z = m.addVar(vtype=GRB.CONTINUOUS, name="z")

        # Set objective
        m.setObjective(0.5* x + 0.1*y + 0.22 * z, GRB.MAXIMIZE)

        # Add constraint: x + 2 y + 3 z <= 4
        m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

        # Add constraint: x + y >= 1
        m.addConstr(x + y >= 1, "c1")

        # Optimize model
        m.optimize()

        for v in m.getVars():
            print('%s %g' % (v.VarName, v.X))

        print('Obj: %g' % m.ObjVal)

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

if __name__ == '__main__':
    model = tf.keras.models.load_model(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +'/Models/mnist.h5')
    inp = getmnist()

    num_inputs = len(inp)
    sample_output = model.predict([inp])
    true_label = (np.argmax(sample_output))
    num_outputs = len(sample_output[0])

    print(true_label)

    ann(5, model, inp, true_label, num_inputs, num_outputs)
    epsilons, status = find(0.5, 10, model, inp, true_label, num_inputs, num_outputs)