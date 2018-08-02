from __future__ import print_function

import sys
import struct
import re
import numpy as np
import argparse

FLAGS = None

class LayerType:
    fully_connect = 0x00
    relu = 0x01
    sigmoid = 0x02
    tanh = 0x03
    softmax = 0x04

def error_msg(msg):
    print('error: '+ msg)
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

def read_matrix(arr, offset, rows, cols):
    mat = np.zeros((rows, cols), dtype=np.float32)
    assert(arr[offset] == '[')
    offset += 1 
    for i in range(0, rows):
        for j in range(0, cols):
            mat[i, j] = float(arr[offset])
            offset += 1
    assert(arr[offset] == ']')
    offset += 1
    return mat, offset

def read_vector(arr, offset, dim):
    vec = np.zeros((dim), dtype=np.float32)
    assert(arr[offset] == '[')
    offset += 1 
    for i in range(0, dim):
        vec[i] = float(arr[offset])
        offset += 1
    assert(arr[offset] == ']')
    offset += 1
    return vec, offset

class Component:
    # read from arr, return new offset
    def read(self, arr, offset):
        self.out_dim = int(arr[offset])
        offset += 1
        self.in_dim = int(arr[offset])
        offset += 1
        offset = self.read_data(arr, offset)
        if arr[offset] == '<!EndOfComponent>':
            offset += 1
        return offset
    # default do nothing for activation function sigmoid/tanh/softmax
    def read_data(self, arr, offset):
        return offset

    def write(self, fid):
        error_msg('not implement')

class Affine(Component):
    def __init__(self, bias = True):
        self.have_bias = bias

    def read_data(self, arr, offset):
        while re.match('<[^ >]+>', arr[offset]):
            #print(arr[offset])
            offset += 2
        self.w, offset = read_matrix(arr, offset, self.out_dim, self.in_dim)
        if self.have_bias:
            self.b, offset = read_vector(arr, offset, self.out_dim)
        else:
            self.b = np.zeros((self.out_dim), dtype=np.float32)
        return offset
    
    def write(self, fid):
        write_layer_head(fid, LayerType.fully_connect, self.in_dim, self.out_dim)
        write_ndarray(fid, self.w)
        write_ndarray(fid, self.b)

class Sigmoid(Component):
    def write(self, fid):
        write_layer_head(fid, LayerType.sigmoid, self.in_dim, self.out_dim)

class ReLU(Component):
    def write(self, fid):
        write_layer_head(fid, LayerType.relu, self.in_dim, self.out_dim)

class Tanh(Component):
    def write(self, fid):
        write_layer_head(fid, LayerType.tanh, self.in_dim, self.out_dim)

class Softmax(Component):
    def write(self, fid):
        write_layer_head(fid, LayerType.softmax, self.in_dim, self.out_dim)

def read_nnet1(filename):
    net = []
    with open(filename, 'r') as fid:
        # kaldi binary nnet file start with '\0B'
        if fid.read(2) == '\0B':
            error_msg('''kaldi nnet1 binary file is not supported, convert it to txt file use nnet-copy''')
        fid.seek(0)
        arr = fid.read().split()
        offset = 0
        while offset < len(arr):
            token = arr[offset]
            if token == '<Nnet>' or token == '</Nnet>': 
                offset += 1
            elif token == '<AffineTransform>': 
                layer = Affine()
                offset = layer.read(arr, offset + 1)
                net.append(layer)
            elif token == '<LinearTransform>': 
                layer = Affine(False)
                offset = layer.read(arr, offset + 1)
                net.append(layer)
            elif token == '<Sigmoid>': 
                layer = Sigmoid()
                offset = layer.read(arr, offset + 1)
                net.append(layer)
            elif token == '<Tanh>': 
                layer = Tanh()
                offset = layer.read(arr, offset + 1)
                net.append(layer)
            elif token == '<ReLU>': 
                layer = ReLU()
                offset = layer.read(arr, offset + 1)
                net.append(layer)
            elif token == '<Softmax>': 
                layer = Softmax()
                offset = layer.read(arr, offset + 1)
                net.append(layer)
            else:
                error_msg('unsupported token %s' % token)
    return net

def convert_nnet1_model_to_net(model, out_filename):
    fid = open(out_filename, "wb")
    for i, layer in enumerate(model[:-1]):
        print(layer.__class__.__name__, layer.in_dim, layer.out_dim)
        layer.write(fid)
    layer = model[-1]
    if FLAGS.remove_last_softmax and layer.__class__.__name__ == 'Softmax':
        print('Remove last softmax layer')
    else:
        print(layer.__class__.__name__, layer.in_dim, layer.out_dim)
        layer.write(fid)
    fid.close()

if __name__ == '__main__':
    usage = '''Usage: convert kaldi nnet1 model to net model
               eg: convert_kaldi_nnet1_model.py keras_model_file out_net_file'''
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('--remove-last-softmax', action='store_true',
                        help='whether to remove last softmax or not')
    parser.add_argument('kaldi_nnet1_model',
                        help='standard kaldi nnet1 text model file')
    parser.add_argument('out_net_model',
                        help='net format out file')
    FLAGS = parser.parse_args()
    net = read_nnet1(FLAGS.kaldi_nnet1_model)
    convert_nnet1_model_to_net(net, FLAGS.out_net_model)

