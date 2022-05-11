import os
import uuid
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import model_from_json


def splitModel(net_model):
    print('\n------------------Model Summary------------------\n')
    print(net_model.summary())

    submodel = keras.Model(inputs=net_model.inputs, outputs=net_model.layers[-2].output, name='submodel-'+str(uuid.uuid4())[:5])
 #   print('\n------------------SubModel Summary------------------\n')
 #   print(submodel.summary())

    in_shape = (submodel.output.shape[1],)
    last_layer_in = keras.layers.Input(shape=in_shape, name='ll_input-'+str(uuid.uuid4())[:5])
    last_layer_model = keras.Model(inputs=last_layer_in, outputs=net_model.layers[-1](last_layer_in), name='lastlayer-'+str(uuid.uuid4())[:5])
 #   print('\n------------------Last Layer Summary------------------\n')
 #   print(last_layer_model.summary())
    print('\n------------------------------------------------------\n')

    return submodel, last_layer_model

def save_model(json_path, model_path, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(json_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_path)
    print("Saved model to disk")

def load_model(json_path, model_path):
    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path)
    print("Loaded model from disk")
    return loaded_model

load_model_name = "ACASXU_2_9"
model = load_model('./Models/{}.json'.format(load_model_name), './Models/{}.h5'.format(load_model_name))
splitModel(model)
