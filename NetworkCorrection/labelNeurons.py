import os
import sys
from time import time
import numpy
sys.path.append('../')
import numpy as np
import argparse
import keras
from numpy import genfromtxt


class labelNeurons:
    # def __init__(self, epsilon_max):
    #     self.epsilon_max = epsilon_max

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
            l = l + 1
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            input = [max(r,0) for r in result]
            neurons.append(input)
        return neurons

    def create_Marabou_equation(self, loaded_model, input, num_layers):
        

        neurons = []
        l = 0
        for layer in loaded_model.layers:
            if l==0:
                l = l + 1
                continue
            w = layer.get_weights()[0]
            ep = np.zeros_like(w)
            # print(w)
            print(ep)
        return neurons
    
    def labelNodes(self, model, all_layer_outputs ,last_layer_labels, threshold):
        # print("Label Nodes:")
        # print(len(all_layer_outputs))
        labels = []
        # threshold = 0
        label_previous_layer = []
        for layer in range(7, 1, -1):
            curent_layer_edgeWeights = model.layers[layer].get_weights()[0]
            if layer == 7:
                i, d, s = 0, 0, 0
                label_current_layer = []
                node_num=-1
                for node in curent_layer_edgeWeights:
                    node_num=node_num+1
                    # print(node_num)
                    # print(all_layer_outputs[layer-2][node_num])
                    """
                    If a neuron is evaluating to 0 after a ReLU operation, it means that it would've evaluated to (-inf, 0] before ReLU operation. 
                    In this case, I can fix this neuron as an increment neuron or maybe as a neuron which is insignificant and the edges attached to it,
                    their weights should either be increased or left as it is, so their epsilons should be fixed in the positive range.
                    """
                    if all_layer_outputs[layer-2][node_num]==0:
                        # print(node," is an increment neuron.")
                        label_current_layer.append(1)
                        i = i + 1
                    elif (node[0]!=0 or node[1]!=0) and (abs(node[2])<=threshold and abs(node[3])<=threshold and abs(node[4])<=threshold):
                        # print(node," is an increment neuron.")
                        label_current_layer.append(1)
                        i = i + 1
                    elif (node[2]!=0 or node[3]!=0 or node[4]!=0) and (abs(node[0])<=threshold and abs(node[1])<=threshold):
                        # print(node," is a decrement neuron.")
                        label_current_layer.append(0)
                        d = d + 1
                    else:
                        # print(node," is a split neuron.")
                        # label_current_layer.append(2)
                        """
                        Since this is a split neuron, we will have to create duplicate of the DNN at this point.
                        """
                        # pid = os.fork()
                        # if pid > 0 :
                        #     print("I am parent process:")
                        #     label_current_layer.append(1)
                        #     i = i + 1
                        # else:
                        #     print("I am child process:")
                        #     label_current_layer.append(0)
                        #     d = d + 1
                        s = s + 1
                        label_current_layer.append(2)
                # print(i, " ", d," ", s)
                # print(label_current_layer)
                labels.append(label_current_layer)
                label_previous_layer = label_current_layer
            else:
                # threshold = 0.005
                i, d, s = 0, 0, 0
                label_current_layer = []
                node_num=-1
                for node in curent_layer_edgeWeights:
                    node_num=node_num+1
                    # print(node_num)
                    # print(all_layer_outputs[layer-2][node_num])
                    count_inc = 0
                    count_dec = 0
                    count_split = 0
                    if all_layer_outputs[layer-2][node_num]==0:
                        # print(node," is an increment neuron.")
                        label_current_layer.append(1)
                        i = i + 1
                        continue
                    for n in range(50):
                        if label_previous_layer[n]==1 and abs(node[n])<=threshold:
                            count_inc = count_inc+1
                        if label_previous_layer[n]==0 and abs(node[n])<=threshold:
                            count_dec = count_dec+1
                        elif label_previous_layer[n]==2 and abs(node[n])<=threshold:
                            count_split = count_split+1
                    if count_inc!=0 and count_dec==0 and count_split==0:
                        i = i+1
                        label_current_layer.append(1)

                    elif count_inc==0 and count_dec!=0 and count_split==0:
                        d = d+1
                        label_current_layer.append(0)

                    else:
                        s = s+1
                        label_current_layer.append(2)
                        """
                        Since this is a split neuron, we will have to create duplicate of the DNN at this point.
                        """
                        # pid = os.fork()
                        # if pid > 0 :
                        #     print("I am parent process:")
                        #     label_current_layer.append(1)
                        #     i = i + 1
                        # else:
                        #     print("I am child process:")
                        #     label_current_layer.append(0)
                        #     d = d + 1

                # print(i, " ", d," ", s)
                # print(label_current_layer)
                labels.append(label_current_layer)
                label_previous_layer = label_current_layer
                # break
        labels.reverse()
        # for l in labels:
        #     print(l)
        return labels
    
    def fixEpsilonRanges(self, loaded_model, all_layer_outputs, labels, max_layers):
        epsilons = []
        for layer in range(max_layers-1, 0, -1):
            current_layer = layer-1
            next_layer = layer
            weight = loaded_model.layers[next_layer+1].get_weights()[0]
            # print(weight)
            epsilon = np.zeros_like(weight)
            for label in range(0,len(labels[next_layer])):
                if labels[next_layer][label]==1:
                    for curr_layer_neuron in range(0, len(epsilon)):
                        if weight[curr_layer_neuron][label]!=0:
                            epsilon[curr_layer_neuron][label] = 1
                    #This means the next layer neuron is an increment neuron, i.e the edge weight should only increase.
                elif labels[next_layer][label]==0:
                    for curr_layer_neuron in range(0, len(epsilon)):
                        if weight[curr_layer_neuron][label]!=0:
                            epsilon[curr_layer_neuron][label] = -1
                    #This means the next layer neuron is an decrement neuron, i.e the edge weight should only decrease.
                else:
                    for curr_layer_neuron in range(0, len(epsilon)):
                        if weight[curr_layer_neuron][label]!=0:
                            epsilon[curr_layer_neuron][label] = 2
                    #This indicates a split neuron.
            epsilons.append(epsilon)
            # print(epsilon)
        return epsilons

    def findEpsilon(self, loaded_model, all_layer_outputs, labels, max_layers):
        
        return

    def run(self, threshold, last_layer_labels, max_layers):     
        json_file = open('./Models/ACASXU_2_9.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = keras.models.model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("./Models/ACASXU_2_9.h5")
        print("Loaded model from disk.")

        num_layers = len(loaded_model.layers)
        inputs = genfromtxt('./data/inputs.csv', delimiter=',')

        all_layer_outputs = self.get_neuron_values(loaded_model, inputs[0], num_layers)

        # print("Printing outputs of all layers:")
        # for output in all_layer_outputs:
        #     print(len(output), output)

        t1 = time()
        labels = self.labelNodes(loaded_model, all_layer_outputs ,last_layer_labels, threshold)
        t2 = time()
        print("Time taken in labelling of all nodes was: ", (t2-t1)," seconds.")
        labels.append(last_layer_labels)
        # for l in labels:
        #     print(l)
        epsilon = self.fixEpsilonRanges(loaded_model, all_layer_outputs, labels, max_layers)
        self.create_Marabou_equation(loaded_model, inputs[0], num_layers)
        return loaded_model, all_layer_outputs, labels
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.005, help='This is the threshold for edge weights. \
                        If an edge weight is below this threshold, it will not be considered while labelling neurons \
                        as increment/decrement/split neurons.')

    parser.add_argument('--max_layers', default=7, help='Number of layers in the DNN.')
    
    args = parser.parse_args()
    threshold = float(args.threshold)
    max_layers = int(args.max_layers)
    last_layer_labels = [1, 1, 0, 0, 0]         ### 2 stands for split neurons, 1 stands for increment neuron and 0 stands for decrement neuron.###
        
    MODELS_PATH = './Models'
    object = labelNeurons()
    loaded_model, all_layer_outputs, labels = object.run(threshold, last_layer_labels, max_layers)