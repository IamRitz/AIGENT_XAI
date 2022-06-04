import sys
sys.path.append('../')
from Gurobi.ConvertNNETtoTensor import ConvertNNETtoTensorFlow

"""
Compares results of original and modified networks.
"""

obj = ConvertNNETtoTensorFlow()

file = '../Models/testdp1_2_2op.nnet'
inp, out, model = obj.convert(file)

file = '../Models/testdp1_2_2opModifiedZ3.nnet'
inp, out, model = obj.convert(file)

file = '../Models/testdp1_2_2opModifiedGurobi.nnet'
inp, out, model = obj.convert(file)
