import sys
sys.path.append('../')
import keras
from keras import models
from keras import layers
from WatermarkRemoval import utils

def get_output_of_layer(layer, layer_outputs, new_input):
        # if we have already applied this layer on its input(s) tensors,
        # just return its already computed output
        if layer.name in layer_outputs:
            return layer_outputs[layer.name]

        # if this is the starting layer, then apply it on the input tensor
        if layer.name == "input":
            out = layer(new_input)
            layer_outputs[layer.name] = out
            return out

        # find all the connected layers which this layer
        # consumes their output
        prev_layers = []
        for node in layer._inbound_nodes:
            print("HI")
            prev_layers.append(node.inbound_layers)

        # get the output of connected layers
        pl_outs = []
        for pl in prev_layers:
            pl_outs.extend([get_output_of_layer(pl, layer_outputs, new_input)])

        # apply this layer on the collected outputs
        out = layer(pl_outs[0] if len(pl_outs) == 1 else pl_outs)
        layer_outputs[layer.name] = out
        return out

def perform():
        # note that we start from the last layer of our desired sub-model.
        # this layer could be any layer of the original model as long as it is
        # reachable from the starting layer
        load_model_name = 'ACASXU_2_9'
        model = utils.load_model('./Models/{}.json'.format(load_model_name), './Models/{}.h5'.format(load_model_name))
        submodel = keras.Model(inputs=model.inputs, outputs=model.layers[-3].output, name='submodel-'+str(1))
        print(submodel.summary())
        in_shape = (submodel.output.shape[1])
        print(in_shape)
        last_layer_in = keras.layers.Input(shape=in_shape, name='ll_input-'+str(2))
        print(type(last_layer_in), "\n", last_layer_in)
        last_layer_model = keras.Model(inputs=last_layer_in, outputs=model.layers[-1](last_layer_in), name='lastlayer-'+str(1))
        print('\n------------------Last Layer Summary------------------\n')
        print(last_layer_model.summary())
        print('\n------------------------------------------------------\n')
        # layer_outputs = {}
        # new_input = layers.Input(batch_shape=model.get_layer("input").get_input_shape_at(0))
        # print(model.layers[-2])
        # new_output = get_output_of_layer(model.layers[-1], layer_outputs, new_input)

        # # create the sub-model
        # model = models.Model(new_input, new_output)
        # print(model.summary())
        # submodel = keras.Model(inputs=model.layers[-3], outputs=model.layers[-1].output, name='submodel-'+str(1))
        # print(submodel.summary())

print(perform())