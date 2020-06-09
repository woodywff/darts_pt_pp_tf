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
    return

import torch
def accuracy(output, target, topk=(1,)):
    '''
    output: probability output from the network.
    target: class labels.
    topk: for example, topk=(1,2,5) would return top-1, top-2, and top-5 accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.sum(correct[:k].view(-1).float())
        res.append(round((correct_k/batch_size).item(),3))
    return res   

def drop_path(x, drop_rate):
    '''
    dropout on the channel dimension.
    drop_rate: probability of an element to be zero-ed.
    This is a backup for torch.nn.Dropout2d.
    '''
    if drop_rate > 0.:
        keep_prob = 1.-drop_rate
        mask = torch.zeros(1,x.size(1)).bernoulli_(keep_prob)
        x /= keep_prob
        x *= mask
    return x