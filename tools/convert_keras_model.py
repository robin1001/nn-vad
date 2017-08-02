from __future__ import print_function

import sys
import struct
import numpy as np

import keras
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout

class LayerType:
    fully_connect = 0x00
    relu = 0x01
    sigmoid = 0x02
    tanh = 0x03
    softmax = 0x04

def error_msg(msg):
    print(msg)
    sys.exit(-1)

def write_layer_head(fid, layer_type, in_dim, out_dim):
    fid.write(struct.pack('<b2i', layer_type, in_dim, out_dim))

def write_ndarray(fid, arr):
    if arr.dtype.name != 'float32':
       error_msg('params must in float32 format, but got %s'. arr.dtype.name) 
    if arr.ndim == 1: #vector
        fid.write(struct.pack('<i', arr.shape[0]))
    elif arr.ndim == 2: #matrix
        fid.write(struct.pack('<2i', arr.shape[0], arr.shape[1]))
    else:
        error_msg("unsopported ndarry dim %d" % arr.ndim)

    for e in arr.flat:
        fid.write(struct.pack('<f', e))

def convert_activation(fid, act, in_dim, out_dim):
    if act == 'relu': 
        print(act, in_dim, out_dim)
        write_layer_head(fid, LayerType.relu, in_dim, out_dim)
    elif act == 'sigmoid':  
        print(act, in_dim, out_dim)
        write_layer_head(fid, LayerType.sigmoid, in_dim, out_dim)
    elif act == 'tanh':  
        print(act, in_dim, out_dim)
        write_layer_head(fid, LayerType.tanh, in_dim, out_dim)
    elif act == 'softmax':  
        print(act, in_dim, out_dim)
        write_layer_head(fid, LayerType.softmax, in_dim, out_dim)
    elif act == 'linear': 
        pass
    else:
        error_msg('activation %s is not supported' % act)

def convert_keras_model_to_net(model, out_filename):
    fid = open(out_filename, "wb")
    layers = model.layers
    for layer in layers:
        layer_name = layer.name
        class_name = layer.__class__.__name__
        in_dim, out_dim = layer.input_shape[1], layer.output_shape[1]
        if class_name == 'Dense':
            if not layer.use_bias:
                error_msg('Dense layer must use bias %s' % layer_name)
            print(class_name, in_dim, out_dim)
            write_layer_head(fid, LayerType.fully_connect, in_dim, out_dim)
            write_ndarray(fid, layer.kernel.get_value().T)
            write_ndarray(fid, layer.bias.get_value())
            if layer.activation != None:
                act = layer.activation.__name__
                convert_activation(fid, act, out_dim, out_dim)
        elif class_name == 'Activation':
            act = layer.activation.__name__
            convert_activation(fid, act, in_dim, out_dim)
        else:
            error_msg('error, layer %s %s is supported' % (layer_name, class_name))
    fid.close()

if __name__ == '__main__':
    usage = '''Usage: convert keras sequential model to self net model
               eg: convert_keras_model.py keras_model_file out_net_file'''
    if len(sys.argv) != 3:
        error_msg(usage)

    model = load_model(sys.argv[1]) 
    if not isinstance(model, Sequential):
        error_msg('model %s is not Sequential model' % sys.argv[1])
    convert_keras_model_to_net(model, sys.argv[2])


