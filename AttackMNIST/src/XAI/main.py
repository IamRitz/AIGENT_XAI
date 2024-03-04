import torch
import onnx
import onnxruntime
import matplotlib.pyplot as plt
import minExp
import verif_property
import reader
import ast 
import copy
import helper
import verif_property
import numpy as np
from PIL import Image, ImageDraw
import torch
import slic
import tensorflow as tf
from itertools import chain


def compute_gradient( G, inp , end_relu):
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


def exp_image(image_path, explanation, alpha=0.5):
    # Open the original image
    original_image = Image.open(image_path)
    
    # Create a new image with the same size as the original image
    overlay = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Draw on the overlay image with the specified alpha value
    for neuron in explanation:
        nid = neuron[0]
        neu = nid[1]
        row_ind = neu // 28
        col_ind = neu - (neu // 28) * 28
        draw.point((row_ind, col_ind), fill=(0, 255, 0, int(255 * alpha)))
    
    # Blend the original image and the overlay image
    blended_image = Image.blend(original_image.convert("RGBA"), overlay, alpha=alpha)
    
    return blended_image

if __name__=="__main__":
    # model = "./networks/path_to_save_model.onnx"
    model_path = "../AIGENT/AttackMNIST/Models/mnist.h5"
    # image_csv_path = "./images/inputs.csv"
    image_path = "../Images/digits/2_Digits_6.png"

    B = slic.Bundle()
    seg = 50
    comp = 10
    c_axis = None
    input_features_bundle = B.generate_segments(image_path, seg, comp, c_axis)

    G = reader.network_data_tf(model_path)
    # G1 = copy.deepcopy(G)
    f = open("./images/2img6", "r")
    image = ast.literal_eval(f.read())
    model = tf.keras.models.load_model(model_path)
    pred_output = model.predict(np.array([image]))[0]
    pred_class = np.argmax(pred_output)

    # lin_eqn = [
    #     [1,-1,0,0,0,0,0,0,0,0,0],
    #     [1,0,-1,0,0,0,0,0,0,0,0],
    #     [1,0,0,-1,0,0,0,0,0,0,0],
    #     [1,0,0,0,-1,0,0,0,0,0,0],
    #     [1,0,0,0,0,-1,0,0,0,0,0],
    #     [1,0,0,0,0,0,-1,0,0,0,0],
    #     [1,0,0,0,0,0,0,-1,0,0,0],
    #     [1,0,0,0,0,0,0,0,-1,0,0],
    #     [1,0,0,0,0,0,0,0,0,-1,0]
    # ]


    l = [0] * 11
    l[pred_class] = 1
    
    lin_eqn = [[i for i in l] for _ in range(9)]

    count = 0
    for l in lin_eqn:
        if(count == pred_class):
            count += 1
        l[count] = -1
        count += 1

    # print(lin_eqn)

    verif_property.add_property(G,False,lin_eqn)
    #print(G.nodes())
    imp_neus = compute_gradient(G,image,False)
    img_dict = {}
    img_dict.update({(0, i): val for i, val in enumerate(image)})

    imp_neu_dict = {}
    imp_neu_dict.update({neu[1] : val for neu, val in imp_neus})
    bundle_imp = []
    for bundle in input_features_bundle:
        sum = 0
        for neuron in bundle:
            sum += imp_neu_dict[neuron[0][1]]
        bundle_imp.append(sum)

    bundle_imp = enumerate(bundle_imp)
    bundle_imp = sorted(bundle_imp, key = lambda x: x[1])
    # print(bundle_imp)
    inp_f_bundle = []
    for ind, val in bundle_imp:
        inp_f_bundle.append(input_features_bundle[ind])

    important_neurons = []
    [important_neurons.append((i[0],img_dict[i[0]])) for i in imp_neus]

    inp_lb = [0]*784
    inp_ub = [1]*784
    
    #inputs = [node for node in G.nodes() if node[0] == 0]
    # sorted(inputs)
    # inp_features = [(node,image[i]) for i,node in enumerate(inputs)]
    # inp_features = sorted(inp_features, key=lambda x: x[1])
    # print("inp_features", inp_features)
    E = minExp.XAI() 
    # print(f"Important Neurons: {important_neurons}")
    # print(f"Bundle: {input_features_bundle}")
    # exit(1)

    important_neurons = inp_f_bundle
    explanation,lb_exp,pairs = E.explanation(G, important_neurons, inp_lb, inp_ub)

    # AdvImage(model, image_path, explanation, pred_class)
    
    print("explanation",explanation)	
    image = exp_image(image_path, explanation)
    image.save("2output_ub.png")
    	
    print("Lower bound explanation: ", lb_exp)	
    lb_exp = chain.from_iterable(lb_exp)
    image = exp_image(image_path, lb_exp)
    image.save("2output_lb.png")

    # # exp_image(image_path, pairs)
    print("Pair explanation: ", pairs)
    pairs_exp = set()
    for p in pairs:
        for item in p:
            pairs_exp = pairs_exp.union(item)
    image = exp_image(image_path, pairs_exp)
    image.save("2output_pairs.png")
