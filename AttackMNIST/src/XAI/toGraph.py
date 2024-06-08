import helper
import onnx
import onnx.numpy_helper
from onnx import numpy_helper
import numpy as np
import networkx as nx
import tensorflow as tf


def network_data_tf(model_filename):
    """
    Given the name of the .h5 model file, load it and convert it into a
    NetworkX graph. Returns this NetworkX.

    Return:
    NetworkX directed graph of the loaded .h5 model file.
    """
    # Load the TensorFlow model
    model = tf.keras.models.load_model(model_filename)

    # Initialize weights and biases dictionary
    weights = {}
    biases = {}

    # Extract weights and biases from each layer
    for layer in model.layers:
        layer_weights_biases = layer.get_weights()
        weights[layer.name] = layer_weights_biases[0]
        biases[layer.name] = layer_weights_biases[1]

    keys = list(weights.keys())

    # Size of input layer
    inp_size = len(weights[keys[0]])

    # Size of layers other than input layer in order
    layer_sizes = [len(biases[key]) for key in keys] 

    # Initialize the networkx graph
    G = nx.DiGraph()

    # Construct the input layer
    for i in range(inp_size):
        G.add_node((0, i))

    # Construct the rest of the layers
    for layer_num in range(0, len(keys)):
        sizes = layer_sizes[layer_num]
        for j in range(sizes):
            G.add_node((layer_num+1, j))

    # Add edges for weights
    for layer_id, layer_name in enumerate(keys):
        layer_weights = weights[layer_name]
        for i in range(len(layer_weights)):
            for j in range(layer_sizes[layer_id]):
                G.add_edges_from([((layer_id, i), (layer_id + 1, j), {"weight": layer_weights[i][j]})])

    for layer_id, layer_name in enumerate(keys):
        layer_biases = biases[layer_name]
        for i in range(len(layer_biases)):
            G.nodes[(layer_id+1, i)]["bias"] = layer_biases[i]

    return G


def network_data_onnx(onnx_filename):
    """
    Given the name of the onnx example file, load it and convert it into a
    NetworkX graph. Returns this NetworkX.

    Return:
    NetworkX directed graph of the loaded onnx file.
    """
    onnx_weights = {}
    onnx_model = onnx.load(onnx_filename)
    model_data  = onnx_model.graph.initializer
    # Get the weights and biases out
    for init in model_data:
        weight_bias = numpy_helper.to_array(init)
        onnx_weights[init.name] = weight_bias
    
    keys = list(onnx_weights.keys())
    print("keys: ", keys)
    inp_size = len(onnx_weights[keys[0]][0])
    print("input size", inp_size)

    #Get the layer sizes
    layer_sizes = []
    for key in keys:
        size = len(onnx_weights[key])
        if(key.split(".")[1]=="bias"):
            layer_sizes.append(size)
    print(layer_sizes)
    G = nx.DiGraph()

    #Construct the input layer
    for i in range(inp_size):
        G.add_node((0,i))
    
    #Construct the rest layers
    
    layer_id=0
    for i in range(len(layer_sizes)):
        sizes=layer_sizes[i]
        layer_id+=1
        for i in range(sizes):
            G.add_node((layer_id,i))

    w1=onnx_weights["first.weight"]
    b1=onnx_weights["first.bias"]
    w2=onnx_weights["second.weight"]
    b2=onnx_weights["second.bias"]

    layer_id = 0

    for i in range(len(w1[0])):
        for j in range(len(w1)):
            G.add_edges_from([((layer_id,i),(layer_id+1,j),{"weight":w1[j][i]})])
    
    layer_id= layer_id+1
    
    for i in range(len(b1)):
        G.nodes[(layer_id,i)]["bias"]=b1[i]
        
    for i in range(len(w2[0])):
        for j in range(len(w2)):
            G.add_edges_from([((layer_id,i),(layer_id+1,j),{"weight":w2[j][i]})])

    layer_id = layer_id+1

    for i in range(len(b2)):
        G.nodes[(layer_id,i)]["bias"]=b2[i]

    return G

if __name__ == "__main__":

    import sys
    test_no = int(sys.argv[1])

    if test_no == 0:
        """
        Test loading onnx
        """
    
        G = network_data( 
                "./networks/path_to_save_model.onnx" )

        print( "Layer sizes: ", list( map( len, helper.getLayers( G ))))
