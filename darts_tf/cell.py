from .ops import OPS, FactorizedReduce, ReLUConvBN, BatchNorm
from genotype import PRIMITIVES
import pdb
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential


class MixedOp(Layer):
    def __init__(self, channels, stride):
        '''
        channels: in_channels == out_channels for MixedOp
        '''
        super().__init__()
        self._ops = []
        for prim_op in PRIMITIVES:
            op = OPS[prim_op](channels, stride, False)
            if 'pool' in prim_op:
                op = Sequential([op, BatchNorm(affine=False)])
            self._ops.append(op)

    def call(self, x, alphas):
        '''
        alphas: one row of alpha_reduce or alpha_normal
        '''
        return sum(alphas[i] * op(x) for i, op in enumerate(self._ops))


class Cell(Layer):
    def __init__(self, n_nodes, node_c, reduction, reduction_prev):
        '''
        n_nodes: How many nodes in a cell.
        node_c: node out channels.
        reduction: if True, this is a reduction layer, otherwise a normal layer.
        reduction_prev: if True, the former layer is a reduction layer.
        '''
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.node_c = node_c
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(node_c, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(node_c, 1, 1, affine=False)
        self.preprocess1 = ReLUConvBN(node_c, 1, 1, affine=False)
        
        self._ops = []
        for i in range(self.n_nodes):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(node_c, stride)
                self._ops.append(op)
        return

    def call(self, x0, x1, alphas):
        '''
        x0, x1: Inputs to a cell
        alphas: alpha_reduce or alpha_normal
        '''
        x0 = self.preprocess0(x0)
        x1 = self.preprocess1(x1)
        xs = [x0, x1]
        i = 0
        for node in range(self.n_nodes):
            outputs = []
            for x in xs:
                outputs.append(self._ops[i](x, alphas[i]))
                i += 1
            xs.append(sum(outputs)) 
        return tf.concat(xs[-self.n_nodes:], axis=-1) 
            
       