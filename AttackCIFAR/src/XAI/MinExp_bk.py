from draw import *
import networkx as nx
import copy
import sys
import threading
import random
import verif_property
import helper
import itertools
sys.path.append( "../Marabou/" )
from maraboupy import Marabou, MarabouCore

class XAI:

    def __init__(self, LB=0, UB=784):
        self.lock = threading.Lock()
        self.LB = LB
        self.UB = UB
        self.input_features = []
        self.input_lb = []
        self.input_ub = []
        self.singletons = set()
        self.pairs = set()
        self.free = set()

    def contrastive_singleton():
        # Find the single input features that are important 
        # important: removal causes a mis-classification
        pass
    def contrastive_pairs():
        # Find the contrastive pair of input features without singletons that are important
        # important: removal causes a mis-classification
        pass
    def lb_thread(self):
        # contrastive_singleton()
        # contrastive_pairs()
        pass

    def verify(self, G, fixed_features, free_list):
        """
        Encode the merged graph into a Marabou query and attempt to verify it using
        Marabou.

        Arguments:
        fixed_features - The features whose values cannot be changed
        free_features  - The who can take any value in their domain
        G           -   The graph for the network
        inp_lb      -   The lower bound on the inputs
        inp_ub      -   The upper bound on the inputs
        Reuturns:
        A counterexample and Sat is there is an assignment, None otherwise
        """
        #print("FREE", free_features)
        # Get the input nodes and output node
        layers = helper.getLayers( G )
        inp_nodes = layers[0]
        assert len( layers[-1] ) == 1
        out_node = layers[-1][0]
        # Create variables for forward and backward
        n2v_post = { n : i for i, n in enumerate( G.nodes() ) }
        n2v_pre = {
            n : i + len(n2v_post)
            for i, n in enumerate( itertools.chain( *layers[1:] ))
        }
        # Reverse view
        rev=nx.reverse_view(G)

        # Set up solver
        solver = MarabouCore.InputQuery()
        solver.setNumberOfVariables( len(n2v_post) + len(n2v_pre) )
        # Encode the network

        for node in G.nodes():
            eq = MarabouCore.Equation()
            flag = False
            for pred in rev.neighbors(node):
                flag = True
                a = G.edges[(pred,node)]['weight']
                eq.addAddend(a, n2v_post[pred])
            if flag:  #and G.neighbors(node)!=[]:
                eq.addAddend(-1, n2v_pre[node])
                eq.setScalar(-1*G.nodes[node]['bias'])
                solver.addEquation(eq)
                if(node!=out_node):
                    MarabouCore.addReluConstraint(solver,
                            n2v_pre[node], n2v_post[node])
                else:
                    eq1 = MarabouCore.Equation()
                    eq1.addAddend(1,n2v_pre[out_node])
                    eq1.addAddend(-1,n2v_post[out_node])
                    eq1.setScalar(0)
                    solver.addEquation(eq1)

        # Encode precondition
        for feature in fixed_features:
            node,val = feature[0],feature[1]
            solver.setLowerBound(n2v_post[node],val)
            solver.setUpperBound(n2v_post[node],val)

        for feature in free_list:
            
            node = feature[0]
            #print("Node", feature[1])
            solver.setLowerBound(n2v_post[node],self.input_lb[node[1]])
            solver.setUpperBound(n2v_post[node],self.input_ub[node[1]])

            
        # Encode postcondition
     
        solver.setUpperBound( n2v_post[ out_node ], 0)

        options=Marabou.createOptions(
            verbosity = 0 )
        ifsat, var, stats = MarabouCore.solve(solver,options,'')
        if(len(var)>0):
            ycex=var[n2v_post[out_node]]
            cex=[]
            for node in inp_nodes:
                cex.append((node, var[n2v_post[node]]))

            #if config.DEBUG:
            #    helper.log("Out (4,0): ", var[ n2v_post[ (4,0) ]])

            return ('SAT',cex)
        else:
            return  ('UNSAT',None)

    def ub_thread(self):
        # With all the features as important, this threads tends to remove input_features that doesn't
        # cause misclassification and thus find all the free input features

        exp = set(self.input_features) 

        remaining_features = self.input_features
        L = 0
        R = len(remaining_features)-1
        while L <= len(remaining_features)-1:
            while L <= R:
                mid = (L+R)//2
                exp = set(remaining_features)-self.free
                to_remove = set(remaining_features[L:mid+1])
                print("Checking for ", to_remove)
                t = list(self.free)
                t.extend(to_remove)
                tuple_cex = self.verify(G, exp, t)
                if tuple_cex[0] == 'UNSAT':
                    self.free = self.free.union(to_remove) 
                    print("free",to_remove)
                    self.UB = self.UB- len(to_remove)
                    L = mid+1
                else:
                    print("Not free",to_remove)
                    R= mid-1

            L  = L + 1
            R  = len(remaining_features)-1

        return exp


    def explanation(self, G, input_features, input_lb, input_ub):
        # Run the lower and upper bound thread in parallel
        self.input_features = input_features
        self.input_lb = input_lb
        self.input_ub = input_ub
        thread1 = threading.Thread(target=self.ub_thread)
        thread2 = threading.Thread(target=self.lb_thread)
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        final_result = set(self.input_features) - self.free
        return final_result

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


    G_copy = copy.deepcopy(G)
    conj_lin_equations = [ [1,-1,0] ]
    verif_property.add_property(G,False,conj_lin_equations)
    inp_features = [ ((0,0),0),((0,1),1),((0,2),1) ]
    inp_lb = [0,0,0]
    inp_ub = [1,1,1]
    E = XAI()
    exp = E.explanation(G, inp_features, inp_lb, inp_ub)
    print("Explanation: ", exp)

    visualize = []
    for node in exp:
        visualize.append(node[0])

    
    nodes = [ ((i, j), {'bias': bias[(i,j)]}) for i,_ in enumerate(layer_sizes) for j in range(layer_sizes[i]) if i==2]
    edges = [(i, j, {'weight' : weights[(i,j)]['weight']}) for (i,j), _ in weights.items()]
    print(G.nodes(data=True))
    print(G.edges(data=True))
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_neural_network(G_copy, ax, layer_sizes, visualize)
    plt.axis('off')
    # nx.draw_spring(G, with_labels=True)
    plt.show()
