#!/usr/bin/python

# Created on 2017-08-02
# Author: Binbin Zhang

import sys
import math
import struct

def error_msg(msg):
    print('error: '+ msg)
 
def convert_cmvn(kaldi_cmvn_file, out_file):
    means = []
    variance = []
    with open(kaldi_cmvn_file, 'r') as fid:
        # kaldi binary file start with '\0B'
        if fid.read(2) == '\0B':
            error_msg('''kaldi cmvn binary file is not supported, convert it to txt file use copy-matrix''')
        fid.seek(0)
        arr = fid.read().split()
        assert(arr[0] == '[')
        assert(arr[-2] == '0')
        assert(arr[-1] == ']')
        feat_dim = (len(arr) - 2 - 2) / 2
        for i in range(1, feat_dim+1):
            means.append(float(arr[i]))
        count = float(arr[feat_dim+1])
        for i in range(feat_dim+2, 2*feat_dim+2):
            variance.append(float(arr[i]))

    for i in range(len(means)): 
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20: variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])

    print feat_dim
    print count
    print means
    print variance
    with open(out_file, 'wb') as fid:
        fid.write(struct.pack('<2i', 2, feat_dim))
        for e in means:
            fid.write(struct.pack('<f', e))
        for e in variance:
            fid.write(struct.pack('<f', e))

if __name__ == '__main__':
    usage = '''Usage: convert kaldi cmvn model to net matrix 
               eg: convert_kaldi_cmvn.py kaldi_cmvn_file out_net_file'''
    if len(sys.argv) != 3:
        error_msg(usage)

    convert_cmvn(sys.argv[1], sys.argv[2])

