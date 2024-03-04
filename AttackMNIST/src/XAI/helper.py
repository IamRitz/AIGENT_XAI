import networkx as nx

def getLayers(G):
    layer_dict = {}     # arrange nodes in G according to their layer
    for node in G.nodes:

        if node[0] not in layer_dict:
            layer_dict[node[0]] = []
        layer_dict[node[0]].append(node)

    layers = []
    for i in sorted( layer_dict.keys() ):
        layers.append( layer_dict[i] )
    for l in layers: l.sort()

    return layers

def getLayerSize(layers):
    G_layer_sizes = []
    for i in layers:
        G_layer_sizes.append(len(i))
    return G_layer_sizes

def findClassForImage(G, input_val):

    simulationDict = {}
    layers = getLayers(G)
    for node, value in input_val:
        simulationDict[node] = value

    print(simulationDict)
    H = nx.reverse_view(G)

    for i in range(len(layers)-1):
        for y in range(len(layers[i+1])):
            node = layers[i+1][y]
            value_node = 0   
            adj = H.adj[node]
            # print(adj)
            for x in adj:            
                w = G.edges[x, node]['weight']
                try:
                    value_node += simulationDict[x]*w
                except Exception as e:
                    print("Error: ", e)

            value_node += G.nodes[node]['bias'] 

            if((i+1)!=(len(layers)-1)):  
                value_node = max( value_node, 0 )  
                   
            simulationDict[node] = value_node 

    last_layer = layers[-1]
    last_layer_simul = {node :simulationDict[node] for node in last_layer}
    max_key = max(last_layer_simul, key=lambda k: int(last_layer_simul[k]))

    return max_key

def computeValForNetwork(G,val):
    simulationDict = {}
    layers = getLayers(G)
    for i,j in enumerate(val):
        simulationDict[(0,i)] = j

    H = nx.reverse_view(G)

    for i in range(len(layers)-1):
        for y in range(len(layers[i+1])):
            node = layers[i+1][y]
            value_node = 0   
            adj = H.adj[node]
            for x in adj:            
                w = G.edges[x, node]['weight']

                value_node += simulationDict[x]*w

            value_node += G.nodes[node]['bias'] 

            if((i+1)!=(len(layers)-1)):  
                value_node = max( value_node, 0 )  
                   
            simulationDict[node] = value_node 


    return simulationDict

if __name__ == "__main__":
    print("Hello")
