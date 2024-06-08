import helper
import torch
import numpy as np

def get_importance( G, inp , end_relu):
    """
    Evaluates the network on the given input `inp`. Input can also be a
    stack of vectors.
    
    Returns:
    
    1.  The vector of return values
    2.  A list with the values at each layer. At each layer, if the layer
        has a ReLU, the value is the output of the ReLU, or if the layer is
        an input layer, the value is the input, and otherwise the value is
        the output of the linear layer.
    """
    layers = helper.getLayers(G)
    #print(G.edges[(0,0),(1,0)])
    cval = torch.tensor(inp, requires_grad = True, dtype=torch.float32)
    vals = [cval]

    relu = torch.nn.ReLU()

    weights = []
    biases  = []

    #print(len(layers)-1)
    for i in range(len(layers)-1):
        weight_matrix_lyr = []
        bias_matrix_lyr = []
        for y in range(len(layers[i])):
            weight_neuron = []
            # bias_neuron = []
            node = layers[i][y]  
            adj = G.adj[node]
            #print(node)
            for x in adj:            
                w = G.edges[ node, x]['weight']
                weight_neuron.append(w)            
            if i == 0:
                bias_matrix_lyr.append(0)             
            else:
                bias_matrix_lyr.append(G.nodes[node]['bias']) 
            weight_matrix_lyr.append(weight_neuron)
        weights.append(weight_matrix_lyr)
        biases.append(bias_matrix_lyr)
    
    bias_lst_lyr = []
    
    for y in range(len(layers[len(layers)-1])):
        node = layers[len(layers)-1][y] 
        bias_lst_lyr.append(G.nodes[node]['bias'])
    biases.append(bias_lst_lyr)

    for w, b in zip(weights[:-1], biases[1:-1]):
        cval = relu(cval @ torch.from_numpy(np.array(w, dtype=np.float32)) + torch.from_numpy(np.array(b, dtype=np.float32)))
        vals.append(cval)
                   
    # Evaluate last layer
    cval = cval @ torch.from_numpy(np.array(weights[-1], dtype=np.float32)) + torch.from_numpy(np.array(biases[-1], dtype=np.float32))
    if end_relu:
        cval = relu(cval)
    vals.append(cval)

    vals[-1][0].backward(inputs = vals)

    grads = [ v.grad.numpy() for v in vals ]

    imp_neu = torch.mul(torch.tensor(inp, dtype=torch.float32), torch.from_numpy(grads[0]))

    
    important_neurons = []
    for i,val in enumerate(imp_neu):
        tuple_neu = ((0,i),val.item())
        important_neurons.append(tuple_neu)

    # print(important_neurons)

    important_neurons = sorted(important_neurons, key=lambda x: x[1])
    # print(important_neurons)


    #print('grads: ', (important_neurons))

    return important_neurons

if __name__ == '__main__':
    pass
