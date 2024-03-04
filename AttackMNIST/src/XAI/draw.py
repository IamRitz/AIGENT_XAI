import networkx as nx
import matplotlib.pyplot as plt

def draw_neural_network(G, ax, layer_sizes, visualize=None):
    layer_spacing = 1.5
    node_spacing = 0.4

    # Fix position for each layer and in-layer spacing of neurons/nodes
    # pos = {}
    # for i, layer_size in enumerate(layer_sizes):
    #     for j in range(layer_size):
    #         pos[(i, j)] = (i * layer_spacing, j * node_spacing - (layer_size - 1) * node_spacing / 2)


    # Print in Reverse order of nodes in each layer
    pos = {}
    for i in range(len(layer_sizes)):
        layer_nodes = [node for node in G.nodes if node[0] == i]
        layer_nodes.reverse()
        for j, node in enumerate(layer_nodes):
            pos[node] = (i * layer_spacing, j * node_spacing - (len(layer_nodes) - 1) * node_spacing / 2)

    node_colors = ['skyblue' for _ in G.nodes]
    if visualize is not None:
        node_colors = ['orange' if node in visualize else 'skyblue' for node in G.nodes ]

    # Draw nodes and edges
    nx.draw(G, pos, ax=ax, node_size=700, with_labels=False, node_color=node_colors, font_size=8, font_color='black', font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels={(i, j): f'${i}_{j}$' for i, j in G.nodes()}, font_size=8, font_color='black', font_weight='bold')

    # position the edge-weights properly
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6, font_weight='bold', label_pos=0.8)


    # Draw biases on top of nodes
    for i in range(1, len(layer_sizes)):
        for j in range(layer_sizes[i]):
            node_pos = (i, j)
            bias_label = f'{G.nodes[node_pos]["bias"]}'
            ax.text(pos[node_pos][0], pos[node_pos][1] + 0.04, bias_label, fontsize=8, color='black', ha='center', va='bottom')



if __name__ == '__main__':

    G = nx.DiGraph()
    layer_sizes = [3, 3, 2]  # Number of nodes in each layer
    bias = {
                # layer-1
                (0,0) :   0, (0,1) :   0, (0,2) : 0,

                # layer-2
                (1,0) :  -4, (1,1) :  -3, (1,2) : 0,
                
                # layer-3 (o/p layer)
                (2,0) :   0, (2,1) :   0,
           }

    weights = {
                # layer-1 to layer-2
                ((0,0), (1,0)) : {'weight' : 2}, ((0,0), (1,1)) : {'weight' : 2}, ((0,0), (1,2)) : {'weight' : 2},
                ((0,1), (1,0)) : {'weight' : 3}, ((0,1), (1,1)) : {'weight' : 3}, ((0,1), (1,2)) : {'weight' : 3},
                ((0,2), (1,0)) : {'weight' : 5}, ((0,2), (1,1)) : {'weight' : 7}, ((0,2), (1,2)) : {'weight' : 6},
            
                # layer-2 to layer-3 (output-layer)
                ((1,0), (2,0)) : {'weight' : 1}, ((1,0), (2,1)) : {'weight' : -1},
                ((1,1), (2,0)) : {'weight' : 1}, ((1,1), (2,1)) : {'weight' : -1},
                ((1,2), (2,0)) : {'weight' : 0}, ((1,2), (2,1)) : {'weight' : 1},
            }

    # Create nodes along with the bias defined above according to number of nodes in each layer using layer_sizes
    nodes = [ ((i, j), {'bias': bias[(i,j)]}) for i,_ in enumerate(layer_sizes) for j in range(layer_sizes[i])]
    edges = [(i, j, {'weight' : weights[(i,j)]['weight']}) for (i,j), _ in weights.items()]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_neural_network(G, ax, layer_sizes)
    plt.axis('off')
    # nx.draw_spring(G, with_labels=True)
    plt.show()
