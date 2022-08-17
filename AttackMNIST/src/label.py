import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
"""
What this file does?
This file implements the increment decrement approach which is used as a helper function by methods defined in other files.
"""

def labelling(model,true_output, threshold):
    """
    -1 is a decrement node and +1 is an increment node. 0 stands for neutral.
    """
    labels = []
    current_layer_labels = []
    max_layers = len(model.layers)
    for i in range(10):
        if i==true_output:
            current_layer_labels.append(-1)
        else:
            current_layer_labels.append(1)
    labels.append(current_layer_labels)
    label_previous_layer = current_layer_labels
    current_layer_labels = []
    for layer in range(max_layers-1, 0, -1):
        curent_layer_edgeWeights = model.layers[layer].get_weights()[0]
        i, d, s = 0, 0, 0
        node_num=-1
        for node in curent_layer_edgeWeights:
            
            node_num=node_num+1
            count_inc = 0
            count_dec = 0
            count_split = 0
            for n in range(len(curent_layer_edgeWeights[0])):
                if label_previous_layer[n]==1 and abs(node[n])<=threshold:
                    count_inc = count_inc+1
                if label_previous_layer[n]==0 and abs(node[n])<=threshold:
                    count_dec = count_dec+1
                elif label_previous_layer[n]==2 and abs(node[n])<=threshold:
                        ount_split = count_split+1
            if count_inc!=0 and count_dec==0 and count_split==0:
                i = i+1
                current_layer_labels.append(1)

            elif count_inc==0 and count_dec!=0 and count_split==0:
                d = d+1
                current_layer_labels.append(-1)

            else:
                s = s+1
                current_layer_labels.append(0)
        labels.append(current_layer_labels)
        label_previous_layer = current_layer_labels
        current_layer_labels = []
    rev = labels[::-1]
    return rev
