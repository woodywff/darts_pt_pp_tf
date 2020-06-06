from collections import namedtuple
import numpy as np
from ops import PRIMITIVES
import pdb

Genotype = namedtuple('Genotype', ['normal','reduce'])
'''
Genotype saves the searched downward cell and upward cell
'''

class GenoParser:
    def __init__(self, n_nodes):
        '''
        This is the class for genotype operations.
        n_nodes: How many nodes in a cell.
        '''
        self.n_nodes = n_nodes
        
    def parse(self, alphas):
        '''
        Each MixedOp would keep the Op with the highest alpha value.
        For each node, two edges with the highest alpha values are kept as the inputs.
        '''
        alphas = F.softmax(self.alphas, dim=-1).detach().cpu().numpy()
        i = 0
        res = []
        for n_edges in range(2, 2 + self.n_nodes):
            gene = []
            for edge in range(n_edges):
                argmax = np.argmax(alphas[i])
                gene.append((alphas[i][argmax], PRIMITIVES[argmax], edge))
                i += 1
            gene.sort()
            res += [(op[1], op[2]) for op in gene[-2:]]
        return res