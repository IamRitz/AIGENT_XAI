import sys
import labelNeurons
sys.path.append('../')
import argparse
import keras
import numpy as np

class createSplits:
    def fixEpsilonRanges():

        return 0
        
    def run(self, loaded_model, all_layer_outputs, labels, max_layers):
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
            print(epsilon)
        return epsilons

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.005, help='This is the threshold for edge weights. \
                        If an edge weight is below this threshold, it will not be considered while labelling neurons \
                        as increment/decrement/split neurons.')
    
    args = parser.parse_args()
    threshold = float(args.threshold)
    last_layer_labels = [1, 1, 0, 0, 0]         ### 2 stands for split neurons, 1 stands for increment neuron and 0 stands for decrement neuron.###
        
    MODELS_PATH = './Models'
    object_1 = labelNeurons.labelNeurons()
    loaded_model, all_layer_outputs, labels = object_1.run(threshold, last_layer_labels)

    max_layers = 7
    object_2 = createSplits()
    object_2.run(loaded_model, all_layer_outputs, labels, max_layers)