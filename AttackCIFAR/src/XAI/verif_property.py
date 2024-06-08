from draw import *
import helper
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def add_property(G, flag, conj_lin_equations):
    layers = helper.getLayers(G)

    last_layer_nodes = layers[-1]
    num_last_layer = last_layer_nodes[0][0]
    second_last_layer_nodes = layers[-2]

    # Nodes to be added as a conjunction of linear equation for the last layer
    nodes_in_conj_eqn_layer = [(num_last_layer, i) for i in range(len(conj_lin_equations))]

    # Weights that will connect to nodes in conj_eqn_layer from last layer
    weights_conj_eqn_layer = np.array([[row[i] for row in conj_lin_equations] for i in range(len(conj_lin_equations[0])-1)], dtype=np.float64)

    # Weights connecting second last layer and last layer
    weights_second_last_to_last_layer = np.zeros(
            ( len(second_last_layer_nodes), len(last_layer_nodes) ),
            dtype = np.float64)

    # Adding the weights from G
    for i, ni in enumerate(second_last_layer_nodes):
        for j, nj in enumerate(last_layer_nodes):
            if G.has_edge(ni,nj):
                weights_second_last_to_last_layer[i][j] = G.edges[ ni, nj ][ 'weight' ]


    bias_last_layer = np.zeros( ( len(last_layer_nodes), ), dtype = np.float64 )
    bias_conj_eqn_layer = np.zeros(
        ( len(nodes_in_conj_eqn_layer), ), dtype = np.float64
    )

    # Add biases
    for i, n in enumerate( last_layer_nodes ):
        bias_last_layer[i] = G.nodes[ n ][ 'bias' ]

    for i in range(len(nodes_in_conj_eqn_layer)):
        bias_conj_eqn_layer[i] = -1 * conj_lin_equations[i][-1]

    new_weights_last_layer = weights_second_last_to_last_layer @ weights_conj_eqn_layer
    new_biases_last_layer = bias_conj_eqn_layer + bias_last_layer @ weights_conj_eqn_layer 

    # Remove the last layer
    for i in last_layer_nodes:
        G.remove_node(i)

    # Added the new layer in place of last layer with conj_lin_equation
    # property for the original last layer
    for i, node in enumerate(nodes_in_conj_eqn_layer):
        G.add_node(node, bias=bias_conj_eqn_layer[i])

    # Add edges for the above
    for i, weights in enumerate(new_weights_last_layer):
        for j, w in enumerate(weights):
            if(w != 0):
                ni, nj = (num_last_layer-1, i), (num_last_layer, j)
                G.add_edge(ni, nj, weight=w)

    # Add a new layer to the last which sums up the conj_lin_equation
    # for the verification query to check
    if flag:
        if(len(conj_lin_equations) > 1):
            G.add_node((num_last_layer+1,0),bias=0)
            node = (num_last_layer+1,0)
            for i in range(len(conj_lin_equations)):
                G.add_edge((num_last_layer,i),node,weight=1)

def add_property2(G, flag, conj_lin_equations):
    layers = helper.getLayers(G)

    last_layer_nodes = layers[-1]
    num_last_layer = last_layer_nodes[0][0]
    second_last_layer_nodes = layers[-2]

    # Nodes to be added as a conjunction of linear equation for the last layer
    nodes_in_conj_eqn_layer = [(num_last_layer+1, i) for i in range(len(conj_lin_equations))]

    # Weights that will connect to nodes in conj_eqn_layer from last layer
    weights_conj_eqn_layer = np.array([[row[i] for row in conj_lin_equations] for i in range(len(conj_lin_equations[0])-1)], dtype=np.float64)

    for i, node in enumerate(nodes_in_conj_eqn_layer):
        G.add_node(node, bias=conj_lin_equations[i][-1])

    # Add edges for the above
    for i, weights in enumerate(weights_conj_eqn_layer):
        for j, w in enumerate(weights):
            if(w != 0):
                ni, nj = (num_last_layer, i), (num_last_layer+1, j)
                G.add_edge(ni, nj, weight=w)

    # Add a new layer to the last which sums up the conj_lin_equation
    # for the verification query to check
    if flag:
        if(len(conj_lin_equations) > 1):
            G.add_node((num_last_layer+1,0),bias=0)
            node = (num_last_layer+1,0)
            for i in range(len(conj_lin_equations)):
                G.add_edge((num_last_layer,i),node,weight=1)

if __name__ == "__main__":
    G1 = nx.DiGraph()
    G1.add_nodes_from([ ((0, 0), {'bias': 0}), ((0, 1), {'bias' : 0})])
    G1.add_nodes_from([
        ( (1, 0), {'bias': 0.} ),
        ( (1, 1), {'bias': 0.} ),
        ( (1, 2), {'bias': -1.} ),
        ( (1, 3), {'bias': -1.} ),
        ( (1, 4), {'bias': 0.} ),
        ( (1, 5), {'bias': 0.} ),
        ( (1, 6), {'bias': -1.} ),
        ( (1, 7), {'bias': -1.} )])
    G1.add_nodes_from([ ( (2, 0), {'bias': 1.} ), ( (2, 1), {'bias': 2.} ),
                        ( (2, 2), {'bias': 3.} ), ( (2, 3), {'bias': 4.} )])

    G1.add_edges_from([
        ( (0, 0), (1, 0), {'weight': 1000.} ),
        ( (0, 0), (1, 1), {'weight': -1000.} ),
        ( (0, 0), (1, 2), {'weight': 1000.} ),
        ( (0, 0), (1, 3), {'weight': -1000.} )  ])
    G1.add_edges_from([
        ( (0, 1), (1, 0), {'weight': -1000.} ),
        ( (0, 1), (1, 1), {'weight': 1000.} ),
        ( (0, 1), (1, 2), {'weight': -1000.} ),
        ( (0, 1), (1, 3), {'weight': 1000.} )  ])
    G1.add_edges_from([
        ( (0, 1), (1, 4), {'weight': 1000.} ),
        ( (0, 1), (1, 5), {'weight': -1000.} ),
        ( (0, 1), (1, 6), {'weight': 1000.} ),
        ( (0, 1), (1, 7), {'weight': -1000.} )  ])
    G1.add_edges_from([
        ( (0, 0), (1, 4), {'weight': -1000.} ),
        ( (0, 0), (1, 5), {'weight': 1000.} ),
        ( (0, 0), (1, 6), {'weight': -1000.} ),
        ( (0, 0), (1, 7), {'weight': 1000.} )  ])

    G1.add_edges_from([ ( (1, 0), (2, 0), {'weight': 1.} ), ( (1, 2), (2, 0), {'weight': -1.} ) ])
    G1.add_edges_from([ ( (1, 1), (2, 1), {'weight': 1.} ), ( (1, 3), (2, 1), {'weight': -1.} ) ])
    G1.add_edges_from([ ( (1, 4), (2, 2), {'weight': 1.} ), ( (1, 6), (2, 2), {'weight': -1.} ) ])
    G1.add_edges_from([ ( (1, 5), (2, 3), {'weight': 1.} ), ( (1, 7), (2, 3), {'weight': -1.} ) ])


    G1_layer_sizes = [ 2, 8, 4 ]
    length  = len(G1_layer_sizes)

    equations = [
        [1,-1,0,0,1],
        [1,0,-1,0,2],
        [1,0,0,-1,3]
    ]
    # print(f' Nodes: {G1.nodes()}')

    # fig, ax = plt.subplots(figsize=(8, 6))
    # draw_neural_network(G1, ax, G1_layer_sizes)
    # plt.axis('off')
    # plt.show()

    add_property2(G1,False,equations)
    print(G1.nodes(data=True))
    print(G1.edges(data=True))

    # if(len(equations) > 1):
    #     G1_layer_sizes[-1] -= 1
    #     G1_layer_sizes.append(1)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # draw_neural_network(G1, ax, G1_layer_sizes)
    # plt.axis('off')
    # plt.show()

    # print(f' Nodes: {G1.nodes()}')
    # print(f'Edges: {G1.edges(data=True)}')

