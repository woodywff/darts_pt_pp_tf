import torch
import torch.nn as nn
from .ops import OPS, FactorizedReduce, ReLUConvBN
from genotype import PRIMITIVES
import pdb


class MixedOp(nn.Module):
    def __init__(self, channels, stride):
        '''
        channels: in_channels == out_channels for MixedOp
        '''
        super().__init__()
        self._ops = nn.ModuleList()
        for prim_op in PRIMITIVES:
            op = OPS[prim_op](channels, stride, False)
            if 'pool' in prim_op:
                op = nn.Sequential(op, nn.BatchNorm2d(channels, affine=False))
            self._ops.append(op)

    def forward(self, x, alphas):
        '''
        alphas: alpha_reduce or alpha_normal
        '''
        return sum(alpha * op(x) for alpha, op in zip(alphas, self._ops))


class Cell(nn.Module):
    def __init__(self, n_nodes, c0, c1, node_c, reduction, reduction_prev):
        '''
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        node_c: node out channels.
        reduction: if True, this is a reduction layer, otherwise a normal layer.
        reduction_prev: if True, the former layer is a reduction layer.
        '''
        super().__init__()
        self.reduction = reduction
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c0, node_c, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(c0, node_c, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(c1, node_c, 1, 1, 0, affine=False)
        
        self.n_nodes = n_nodes
        self.node_c = node_c
        
        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()
        for i in range(self.n_nodes):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(node_c, stride)
                self._ops.append(op)
        return

    @property
    def out_channels(self):
        return self.n_nodes * self.node_c
    
    def forward(self, x0, x1, alphas):
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
            xs.append(sum(outputs)) # debug: dim_assert
        return torch.cat(xs[-self.n_nodes:], dim=1) # debug: dim_assert
            
       