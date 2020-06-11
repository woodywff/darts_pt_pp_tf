from collections import namedtuple
import numpy as np
import pdb

'''
Genotype saves the searched normal cell and reduction cell.
'''
Genotype = namedtuple('Genotype', ['normal','reduce'])

# This is a subset of darts_xx.ops.OPS
PRIMITIVES = ['max_pool_3x3',
              'avg_pool_3x3',
              'skip_connect',
              'sep_conv_3x3',
              'sep_conv_5x5',
              'dil_conv_3x3',
              'dil_conv_5x5']

class GenoParser:
    '''
    This is the class for genotype operations.
    n_nodes: How many nodes in a cell.
    '''
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        
    def parse(self, softmax_alphas):
        '''
        Each MixedOp would keep the Op with the highest alpha value.
        For each node, two edges with the highest alpha values are kept as the inputs.
        softmax_alphas: output from F.softmax(alphas, dim=-1)
        '''
        i = 0
        res = []
        for n_edges in range(2, 2 + self.n_nodes):
            gene = []
            for edge in range(n_edges):
                argmax = np.argmax(softmax_alphas[i])
                gene.append((softmax_alphas[i][argmax], PRIMITIVES[argmax], edge))
                i += 1
            gene.sort()
            res += [(op[1], op[2]) for op in gene[-2:]]
        return res