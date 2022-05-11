import sys
from time import time
import numpy
sys.path.append('../')
import numpy as np
import argparse
import keras
from numpy import genfromtxt


sat = 'SAT'
unsat = 'UNSAT'
class findCorrection:
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
                        label_current_layer.append(2)
                        s = s + 1
                # print(i, " ", d," ", s)
                print(label_current_layer)
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

                # print(i, " ", d," ", s)
                print(label_current_layer)
                labels.append(label_current_layer)
                label_previous_layer = label_current_layer
                # break

        return labels

    def run(self, threshold):     
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

        last_layer_labels = [1, 1, 0, 0, 0]         ### 2 stands for split neurons, 1 stands for increment neuron and 0 stands for decrement neuron.###
        t1 = time()
        labels = self.labelNodes(loaded_model, all_layer_outputs ,last_layer_labels, threshold)
        t2 = time()
        print("Time taken in labelling of all nodes was: ", (t2-t1)," seconds.")
        for l in labels:
            print(l)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.005, help='This is the threshold for edge weights. \
                        If an edge weight is below this threshold, it will not be considered while labelling neurons \
                        as increment/decrement/split neurons.')
    
    args = parser.parse_args()
    threshold = float(args.threshold)

    MODELS_PATH = './Models'
    problem = findCorrection()
    problem.run(threshold)