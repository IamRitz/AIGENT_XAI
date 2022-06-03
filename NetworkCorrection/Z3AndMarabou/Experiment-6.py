import sys
sys.path.append('../')
from Gurobi.ConvertNNETtoTensor import ConvertNNETtoTensorFlow

obj = ConvertNNETtoTensorFlow()

file = '../Models/testdp1_2_2op.nnet'
inp, out, model = obj.convert(file)

file = '../Models/testdp1_2_2opModified.nnet'
inp, out, model = obj.convert(file)
