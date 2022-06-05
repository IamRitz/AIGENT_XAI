import sys
import numpy as np
sys.path.append('../')
from Gurobi.ConvertNNETtoTensor import ConvertNNETtoTensorFlow

"""
Compares results of original and modified networks.
"""
def get_neuron_values_actual(loaded_model, input, num_layers):
        neurons = []
        l = 1
        # print(len(loaded_model.layers))
        for layer in loaded_model.layers:
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            # print(w)
            print(w)
            print(input)
            result = np.matmul(input,w)+b
            # print(l)
            if l == num_layers:
                input = result
                neurons.append(input)
                continue
            print(result)
            input = [max(0, r) for r in result]
            neurons.append(input)
            l = l + 1
        # print(neurons)
        return neurons

obj = ConvertNNETtoTensorFlow()

file = '../Models/testdp1_2_2op.nnet'
inp, out, model = obj.convert(file)

file = '../Models/testdp1_2_2opModifiedZ3.nnet'
inp, out, model = obj.convert(file)

file = '../Models/testdp1_2_2opModifiedGurobi.nnet'
inp, out, model = obj.convert(file)
