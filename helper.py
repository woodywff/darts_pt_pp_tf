import os
import gzip
import numpy as np


def load_fmnist(path, is_train=True):
    '''
    Rerutn: images.shape = (num of imgs,28*28)
    ATTENTION: the returned ndarray is not writable.
    '''
    kind = 'train' if is_train else 't10k'
    labels_path = os.path.join(path, '{}-labels-idx1-ubyte.gz'.format(kind))
    images_path = os.path.join(path, '{}-images-idx3-ubyte.gz'.format(kind))
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

def calc_param_size(model):
    '''
    Show the memory cost of model.parameters, in MB. 
    '''
    return np.sum(np.prod(v.size()) for v in model.parameters())*4e-6

def print_red(something):
    print("\033[1;31m{}\033[0m".format(something))