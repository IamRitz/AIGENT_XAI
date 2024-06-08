from math import ceil
from draw import *
import networkx as nx
import copy
import sys
import threading
from multiprocessing import Process, Queue
import random
import verif_property
import helper
import itertools
from itertools import chain
sys.path.append( "/home/ritesh/Desktop/Marabou/" )
from maraboupy import Marabou, MarabouCore

class XAI:

    def __init__(self, LB=0, UB=784):
        # self.lock = threading.Lock()
        self.LB = LB
        self.UB = UB
        self.input_features = []
        self.lower_conf = False
        self.conf_score = None
        self.second_largest = None
        self.pred_class = None
        self.pred_value = None
        self.input_lb = []
        self.input_ub = []
        self.singletons = set()
        self.pairs = set()
        self.free = set()
        self.G = None
        self.free_queue = Queue()
        self.contrastive_queue = Queue()
        self.result_pairs = []
        self.result_singletons = []
        self.result_ub = []
        self.output_values = None
        self.smallest = -1

    def contrastive_singleton(self):
        # Find the single input features that are important 
        # important: removal causes a mis-classification
        orig_features = set(self.input_features)
        for ip_f in self.input_features:
                orig_features.remove(ip_f)
                result =  self.verif_query(self.G, orig_features, [ip_f])
                if result[0] == 'SAT':
                        self.result_singletons.append(result[1])
                        self.singletons.add(ip_f)
                        self.LB = self.LB+1
                        return
                else:
                    # print("Not Singleton , ip", result[1], ip_f[0])
                    pass
                orig_features.add(ip_f)


    def contrastive_singleton_bundle(self):
        # Find the single input features that are important 
        # important: removal causes a mis-classification

        orig_features = self.input_features
        # orig_features = set([feature for bundle in self.input_features for feature in bundle])
        for ip_f in self.input_features:
                orig_features.remove(ip_f)
                orig_features_list = set([feature for bundle in orig_features for feature in bundle])
                result =  self.verif_query(self.G, orig_features_list, ip_f)
                if result[0] == 'SAT':
                        # print("Singleton , ip",result[1])
                        # print("LENGTH OF SINGLETON: ", len(ip_f))
                        # print("Singleton in image: ", ip_f)
                        self.singletons.add(tuple(ip_f))
                        self.result_singletons.append(result[1])
                        self.LB = self.LB+1

                else:
                    # print("Not Singleton , ip",result[1],ip_f)
                    pass
                orig_features.append(ip_f)

    def contrastive_pairs(self):
        # Find the contrastive pair of input features without singletons that are important
        # important: removal causes a mis-classification
        ip_features = set(self.input_features) - self.singletons
        contr_pairs = list(itertools.combinations(ip_features, 2))
        orig_features = set(self.input_features)
        for pair in contr_pairs:
                feature_1 =  pair[0]
                feature_2 = pair[1]
                if feature_1 in orig_features:
                    orig_features.remove(feature_1)
                if feature_2 in orig_features:
                    orig_features.remove(feature_2)
                result =  self.verif_query(self.G, orig_features, [feature_1,feature_2])
                if result[0] == 'SAT':
                        # print("Pairs", pair, result[1])
                        self.result_pairs.append(result[1])
                        self.pairs.add(pair)
                        return
                else:
                    # print("Not Pairs",pair)
                    pass
                orig_features.add(feature_1)
                orig_features.add(feature_2)

    def contrastive_pairs_bundle(self):
        # Find the contrastive pair of input features without singletons that are important
        # important: removal causes a mis-classification

        ip_features = set([tuple(feature) for feature in self.input_features if tuple(feature) not in self.singletons])
        # ip_features = list(set(self.input_features) - self.singletons)
        contr_pairs = list(itertools.combinations(ip_features, 2))

        # orig_features = set([feature for bundle in self.input_features for feature in bundle])
        orig_features = self.input_features
        # print(f'fixed features: {orig_features}')
        for pair in contr_pairs:
                feature_1 =  pair[0]
                feature_2 = pair[1]
                if feature_1 in orig_features:
                    orig_features.remove(feature_1)
                if feature_2 in orig_features:
                    orig_features.remove(feature_2)
                free_pairs = []
                free_pairs.extend(feature_1)
                free_pairs.extend(feature_2)

                orig_features_list = set([feature for bundle in orig_features for feature in bundle])
                result =  self.verif_query(self.G, orig_features_list, free_pairs)
                if result[0] == 'SAT':
                        # print("----------------------------------------Pairs----------------------------------------")
                        # pair = [tuple(sublist) if isinstance(sublist, list) else sublist for sublist in pair]
                        self.pairs.add(pair)
                        self.result_pairs.append(result[1])
                else:
                    # print("----------------------------------------Not Pairs----------------------------------------")
                    pass
                
                orig_features.append(feature_1)
                orig_features.append(feature_2)

    def lb_thread(self, q):
        self.contrastive_singleton()
        self.contrastive_pairs()

        q.put(self.singletons)
        q.put(self.result_singletons)
        q.put(self.pairs)
        q.put(self.result_pairs)
        return

    def lb_thread_bundle(self, q):
        self.contrastive_singleton_bundle()
        self.contrastive_pairs_bundle()

        q.put(self.singletons)
        q.put(self.result_singletons)
        q.put(self.pairs)
        q.put(self.result_pairs)

    def verif_query(self, G, orig_features, free_list):
        layers = helper.getLayers( G )
        inp_nodes = layers[0]

        if(self.output_values is None and not self.lower_conf):
            assert len( layers[-1] ) == 1

        out_node = layers[-1]

        if(self.lower_conf):
            out_node2 = layers[-2]

        # Create variables for forward and backward
        n2v_post = { n : i for i, n in enumerate( G.nodes() ) }
        n2v_pre = {
            n : i + len(n2v_post)
            for i, n in enumerate( itertools.chain( *layers[1:] ))
        }
        # print(n2v_post)
        # print(n2v_pre)
        # exit(1)

        # n2v_final = {}
        # if(self.output_values):
        #     n2v_final = {
        #         n : i + len(n2v_post) + len(n2v_pre)
        #         for i, n in enumerate(layers[-1])
        #     }
            # print("n2v_final: ", n2v_final)
        # print("n2v_pre: ", n2v_pre)
        # print("n2v_post: ", n2v_post)
        # Reverse view
        rev=nx.reverse_view(G)
        # print("Verif: ", G.nodes)
        # print("Verif2: ", rev.nodes)
        # exit(1)

        # Set up solver
        solver = MarabouCore.InputQuery()
        solver.setNumberOfVariables( len(n2v_post) + len(n2v_pre))
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
                # if(node!=out_node):
                #     MarabouCore.addReluConstraint(solver,
                #             n2v_pre[node], n2v_post[node])
                if(node in out_node and self.output_values):
                    MarabouCore.addReluConstraint(solver,
                            n2v_pre[node], n2v_post[node])
                elif(node not in out_node and not self.lower_conf):
                    MarabouCore.addReluConstraint(solver,
                            n2v_pre[node], n2v_post[node])
                elif(self.lower_conf and node not in out_node2 and node not in out_node):
                    MarabouCore.addReluConstraint(solver,
                            n2v_pre[node], n2v_post[node])
                else:
                    eq1 = MarabouCore.Equation()
                    eq1.addAddend(1,n2v_pre[node])
                    eq1.addAddend(-1,n2v_post[node])
                    eq1.setScalar(0)
                    solver.addEquation(eq1)

        # if(self.output_values):
        #     for node in layers[-1]:
        #         eq1 = MarabouCore.Equation()
        #         eq1.addAddend(1,n2v_final[node])
        #         eq1.addAddend(-1,n2v_post[node])
        #         eq1.setScalar(0)
        #         solver.addEquation(eq1)

        # Encode precondition
        for feature in orig_features:
            node,val = feature[0],feature[1]
            solver.setLowerBound(n2v_post[node],val)
            solver.setUpperBound(n2v_post[node],val)

        for feature in free_list:
            # print(feature)
            node = feature[0]
            #print("Node", feature[1])
            solver.setLowerBound(n2v_post[node],self.input_lb[node[1]])
            solver.setUpperBound(n2v_post[node],self.input_ub[node[1]])


        # Encode postcondition
        if(not self.output_values and not self.lower_conf):
            # solver.setUpperBound( n2v_post[ out_node[0]], 0)
            # print("Out node: ", n2v_post[out_node[0]])
            solver.setUpperBound( n2v_post[ out_node[0]], 100)
        elif (self.output_values):
            for node in out_node:
                # print(f"Output Constraint: {node} : {self.output_values[node]}")
                # print(n2v_post)
                # print(self.output_values)
                solver.setLowerBound( n2v_post[ node ], self.output_values[node])
                solver.setUpperBound( n2v_post[ node ], self.output_values[node] + 2)
        elif self.lower_conf:
            node = n2v_post[out_node[0+self.second_largest]]
            solver.setUpperBound(node, self.conf_score//2)

        try:
            options=Marabou.createOptions(verbosity = 0)
            options._timeoutInSeconds = 300
            ifsat, var, stats = MarabouCore.solve(solver,options,'')
        except Exception as e:
            val = []
            print(e)
            print("Error in solving")

        if(len(var)>0):
            # ycex=var[n2v_post[out_node]]
            cex=[]
            for node in inp_nodes:
                cex.append((node, var[n2v_post[node]]))
            return ('SAT',cex)
        else:
            return  ('UNSAT',None)


    def verif_query2(self, G, orig_features, free_list):
        layers = helper.getLayers( G )
        inp_nodes = layers[0]
        # print("INPUT NODES: ", inp_nodes)

        # print(self.output_values)
        # print(layers[-1])
        # exit(1)
        if(self.output_values is None):
            assert len( layers[-1] ) == 1

        out_node = layers[-1]

        # Create variables for forward and backward
        n2v_post = { n : i for i, n in enumerate( G.nodes() ) }
        n2v_pre = {
            n : i + len(n2v_post)
            for i, n in enumerate( itertools.chain( *layers[1:] ))
        }

        # n2v_final = {}
        # if(self.output_values):
        #     n2v_final = {
        #         n : i + len(n2v_post) + len(n2v_pre)
        #         for i, n in enumerate(layers[-1])
        #     }
            # print("n2v_final: ", n2v_final)
        # print("n2v_pre: ", n2v_pre)
        # print("n2v_post: ", n2v_post)
        # Reverse view
        rev=nx.reverse_view(G)
        # print("Verif: ", G.nodes)
        # print("Verif2: ", rev.nodes)
        # exit(1)

        # Set up solver
        solver = MarabouCore.InputQuery()
        solver.setNumberOfVariables( len(n2v_post) + len(n2v_pre))
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
                # if(node!=out_node):
                #     MarabouCore.addReluConstraint(solver,
                #             n2v_pre[node], n2v_post[node])

                if(node in out_node and self.output_values):
                    MarabouCore.addReluConstraint(solver,
                            n2v_pre[node], n2v_post[node])
                elif(node not in out_node):
                    MarabouCore.addReluConstraint(solver,
                            n2v_pre[node], n2v_post[node])
                else:
                    eq1 = MarabouCore.Equation()
                    eq1.addAddend(1,n2v_pre[out_node[0]])
                    eq1.addAddend(-1,n2v_post[out_node[0]])
                    eq1.setScalar(0)
                    solver.addEquation(eq1)

        # if(self.output_values):
        #     for node in layers[-1]:
        #         eq1 = MarabouCore.Equation()
        #         eq1.addAddend(1,n2v_final[node])
        #         eq1.addAddend(-1,n2v_post[node])
        #         eq1.setScalar(0)
        #         solver.addEquation(eq1)
        
        # Encode precondition
        # print(orig_features)
        # exit(1)
        for feature in orig_features:
            node,val = feature[0],feature[1]
            # print(f"Original: {node}: {val}")
            solver.setLowerBound(n2v_post[node],val)
            solver.setUpperBound(n2v_post[node],val)

        for feature in free_list:
            # print(feature)
            node = feature[0]
            # print("Node Value: ", feature[1])
            # print(f"Bounds: {self.input_lb[node[1]]}, {self.input_ub[node[1]]}")
            solver.setLowerBound(n2v_post[node],self.input_lb[node[1]])
            solver.setUpperBound(n2v_post[node],self.input_ub[node[1]])

            
        # Encode postcondition

        if(not self.output_values):
            solver.setUpperBound( n2v_post[ out_node[0]], 0)
        else:
            for node in out_node:
                # print(f"Output Constraint: {node} : {self.output_values[node]}")
                # solver.setLowerBound( n2v_post[ node ], self.output_values[node])
                # solver.setUpperBound( n2v_post[ node ], self.output_values[node])
                solver.setLowerBound( n2v_post[ node ], self.output_values[node])
                solver.setUpperBound( n2v_post[ node ], self.output_values[node] + 3)


        options=Marabou.createOptions(
            verbosity = 0 )
        ifsat, var, stats = MarabouCore.solve(solver,options,'')
        if(len(var)>0):
            # ycex=var[n2v_post[out_node]]
            cex=[]
            for node in inp_nodes:
                cex.append((node, var[n2v_post[node]]))
            return ('SAT', cex)
        else:
            return  ('UNSAT',None)

    def ub_thread(self, q):
        # With all the features as important, this threads tends to remove input_features that doesn't
        # cause misclassification and thus find all the free input features
        Explanation = []
        L = 0
        R = len(self.input_features)-1
        count = 0
        change = float('inf')
        mapping = {}
        while L <= len(self.input_features)-1:
            k = 2
            while L <= R:
                print("value of k", k)
                if k > len(self.input_features):
                    print("break")
                    break
                Mid = k
                Explanation = set(self.input_features) - self.free
                # print(f"Exp: {Explanation}")
                potential_free = set(self.input_features[L:Mid+1])
                # print("Potentially Free(Binary Search): ", potential_free)
                free_list = list(self.free)
                free_list.extend(potential_free)
                for node, val in free_list:
                    mapping[node] = val
                # print(f'free_list : {free_list}')
                print("--------------------------------------------------")
                result = self.verif_query(self.G, Explanation-set(potential_free), free_list)
                print("--------------------------------------------------")
                if result[0] == 'UNSAT':
                    self.free = self.free.union(potential_free) 
                    # print("Confirmed to be Free: ", potential_free)
                    self.UB = self.UB- len(potential_free)
                    L = Mid+1
                    k = Mid+1
                else:
                    temp = 0
                    for node, val in result[1]:
                        temp += abs(mapping.get(node, 0) - val)
                    if(temp < change):
                        self.smallest = len(self.result_ub) # this was done to get minimum changed upper bound
                        change = temp
                    if(result[1] not in self.result_ub):
                        self.result_ub.append(result[1])
                    print("Confirmed Not to be free", potential_free)
                    # print("Result_UB: ", result[1])
                    R = Mid-1
                    k = Mid-1
                    break
            print("here")
            break
            print("here2")
            L = L + 1
            R = len(self.input_features)-1
        
        q.put(self.free)
        q.put(self.result_ub)
        q.put(self.smallest)
        return Explanation

    def ub_thread_bundle(self, q):
        # With all the features as important, this threads tends to remove input_features that doesn't
        # cause misclassification and thus find all the free input features

        Explanation = []

        L = 0
        R = len(self.input_features)-1
        count = 0
        inp_f_fl = [feature for bundle in self.input_features for feature in bundle]
        while L <= len(self.input_features)-1:
            break
            while L <= R:
                Mid = (L+R)//2
                # inp_f_fl = set(chain(*self.input_features))
                Explanation = set(inp_f_fl) - self.free
                Explanation = set(inp_f_fl)
                potential_free = set([feature for bundle in self.input_features[L:Mid+1] for feature in bundle])
                # print("Potentially Free(Binary Search): ", potential_free)
                free_list = list(self.free)
                free_list = list()
                # print(free_list)
                free_list.extend(potential_free)
                # print(f'free_list: {free_list}')
                # exit(1)
                # print(f'free list: {free_list}')
                result = self.verif_query(self.G, Explanation - set(free_list), free_list)
                if result[0] == 'UNSAT':
                    self.free = self.free.union(potential_free) 
                    # print("Confirmed to be Free: ")
                    self.UB = self.UB- len(potential_free)
                    L = Mid+1
                else:
                    # print("Confirmed Not to be free", potential_free)
                    # print("Confirmed Not to be free")
                    # print("-------------------Adv Example---------------------")
                    # print(result[1])
                    self.result_ub.append(result[1])
                    break
                    # exit(1)
                    R = Mid-1
            break
            L = L + 1
            R = len(self.input_features)-1

        q.put(self.free)
        q.put(self.result_ub)
        return Explanation


    def explanation(self, G, input_features, input_lb, input_ub, output_values=None):
        # Run the lower and upper bound thread in parallel
        self.input_features = input_features
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.G = G
        self.output_values = output_values

        thread1 = threading.Thread(target=self.lb_thread_bundle, args=(self.contrastive_queue, ))
        thread2 = threading.Thread(target=self.ub_thread_bundle, args=(self.free_queue, ))
        # thread1 = threading.Thread(target=self.lb_thread, args=(self.contrastive_queue, ))
        # thread2 = threading.Thread(target=self.ub_thread, args=(self.free_queue, ))
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()
        

        # # Create processes for lower bound and upper bound calculations
        # p1 = Process(target=self.ub_thread_bundle, args=(self.free_queue, ))
        # p2 = Process(target=self.lb_thread_bundle, args=(self.contrastive_queue, ))
        #
        # # Start the processes
        # p1.start()
        # p2.start()
        #
        # # Wait for the processes to finish
        # p1.join()
        # p2.join()

        # lower_bound_result = set([feature for bundle in self.singletons for feature in bundle])
        # pair_result = set([feature for bundle in self.pairs for feature in bundle])

        self.free = self.free_queue.get()
        self.result_ub = self.free_queue.get()

        # self.smallest = self.free_queue.get()
        
        self.singletons = self.contrastive_queue.get()

        self.result_singletons = self.contrastive_queue.get()
        # print("SINGLETONS: ", self.singletons)

        self.pairs = self.contrastive_queue.get()

        # print("PAIRS: ", self.pairs)
        self.result_pairs = self.contrastive_queue.get()

        # Bundle
        # inp_features_list = [feature for bundle in self.input_features for feature in bundle]
        # upper_bound_result = [feature for feature in inp_features_list if feature not in self.free]
        upper_bound_result = self.result_ub
        # upper_bound_result = []
        # lower_bound_result = []
        # pair_result = []
        lower_bound_result = self.result_singletons
        pair_result = self.result_pairs

        # Non-Bundle
        # upper_bound_result = []
        # upper_bound_result = [feature for feature in self.input_features if feature not in self.free]
        # lower_bound_result = self.singletons
        # pair_result = self.pairs

        # print("Free: ", self.free)
        # print(f"Unsat result Singletons: {len(self.result_singletons)}")
        # print(f"Unsat result Pairs: {len(self.result_pairs)}")
        # print(f"Unsat result UB: {len(self.result_ub)}")

        # print("----------------------------------------")
        # print(upper_bound_result)

        return upper_bound_result, lower_bound_result, pair_result


def remove_till_layer(G, layer_num):
    nodes_to_remove = [node for node in G.nodes if node[0]<layer_num]
    G.remove_nodes_from(nodes_to_remove)
    # Remove isolated nodes if any
    G.remove_nodes_from(list(nx.isolates(G)))

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

    # fig, ax = plt.subplots(figsize=(8, 6))
    # draw_neural_network(G, ax, layer_sizes)
    # plt.axis('off')
    # # nx.draw_spring(G, with_labels=True)
    # plt.show()

    # print(G.nodes)
    # remove_till_layer(G, 1)


    # fig, ax = plt.subplots(figsize=(8, 6))
    # draw_neural_network(G, ax, layer_sizes)
    # plt.axis('off')
    # nx.draw_spring(G, with_labels=True)
    # plt.show()

    G_copy = copy.deepcopy(G)
    conj_lin_equations = [ [1,-1,0] ]

    print(G.nodes)
    verif_property.add_property(G,False,conj_lin_equations)
    print(G.nodes)

    # fig, ax = plt.subplots(figsize=(8, 6))
    # draw_neural_network(G, ax, layer_sizes)
    # plt.axis('off')
    # nx.draw_spring(G, with_labels=True)
    # plt.show()

    # inp_features = [ ((0,0),1),((0,1),1),((0,2),1) ]
    # inp_lb = [0,0,0]
    # inp_ub = [1,1,1]
    # E = XAI()
    # ub_exp, lb_exp, pairs = E.explanation(G, inp_features, inp_lb, inp_ub)
    # print("Upper_bound: ", ub_exp)
    # print("Singletons: ", lb_exp)
    # print("Pairs: ", pairs)


    # visualize = []
    # for node in ub_exp:
    #     visualize.append(node[0])
    #
    # 
    # nodes = [ ((i, j), {'bias': bias[(i,j)]}) for i,_ in enumerate(layer_sizes) for j in range(layer_sizes[i]) if i==2]
    # edges = [(i, j, {'weight' : weights[(i,j)]['weight']}) for (i,j), _ in weights.items()]
    # print(G.nodes(data=True))
    # print(G.edges(data=True))
    # fig, ax = plt.subplots(figsize=(8, 6))
    # draw_neural_network(G_copy, ax, layer_sizes, visualize)
    # plt.axis('off')
    # # nx.draw_spring(G, with_labels=True)
    # plt.show()
