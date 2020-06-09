import torch
import torch.nn as nn
from ops import OPS, FactorizedReduce, ReLUConvBN
import pdb
# from genotype import Genotype

FLAG_DEBUG = False

class SearchedCell(nn.Module):
    def __init__(self, gene, n_nodes, c0, c1, node_c, reduction, reduction_prev, drop_rate=0):
        '''
        gene: Genotype, searched architecture of a cell
        n_nodes: How many nodes in a cell.
        c0, c1: in_channels for two inputs.
        node_c: node out channels.
        reduction: if True, this is a reduction layer, otherwise a normal layer.
        reduction_prev: if True, the former layer is a reduction layer.
        drop_rate: dropout rate.
        '''
        super().__init__()
        self.n_nodes = n_nodes
        self.node_c = node_c
        self.drop_rate = drop_rate
        self.genolist = gene.reduce if reduction else gene.normal
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(c0, node_c)
        else:
            self.preprocess0 = ReLUConvBN(c0, node_c, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(c1, node_c, 1, 1, 0)
        
        self._ops = nn.ModuleList([OPS[i[0]](C=node_c, 
                                             stride=2 if reduction and i[1] < 2 else 1, 
                                             affine=True) for i in self.genolist])
        
        return

    @property
    def out_channels(self):
        return self.n_nodes * self.node_c

    def forward(self, x0, x1):
        '''
        x0, x1: Inputs to a cell
        '''
        x0 = self.preprocess0(x0)
        x1 = self.preprocess1(x1)
        xs = [x0, x1]
        i = 0
        for node in range(self.n_nodes):
            outputs = []
            for _ in range(2):
                temp = self._ops[i](xs[self.genolist[i][1]])
                temp = nn.Dropout2d(self.drop_rate)(temp)
                outputs.append(temp)
                i += 1
            xs.append(sum(outputs)) 
        return torch.cat(xs[-self.n_nodes:], dim=1) 
            

class SearchedNet(nn.Module):
    def __init__(self, gene, in_channels, init_node_c, out_channels, depth, n_nodes, drop_rate=0):
        '''
        gene: Genotype, searched architecture of a cell
        in_channels: RGB channel.
        init_node_c: Initial number of filters or output channels for the node.
        out_channels: How many classes are there in the target.
        depth: Number of cells.
        n_nodes: Number of nodes in each cell.
        drop_rate: dropout rate.
        '''
        super().__init__()
        stem_c = min(in_channels, n_nodes) * init_node_c # stem out_channels
        self.stem = nn.Sequential(
          nn.Conv2d(in_channels, stem_c, 3, padding=1, bias=False),
          nn.BatchNorm2d(stem_c)
        )
        c0 = c1 = stem_c
        node_c = init_node_c # node out_channels
        self.cells = nn.ModuleList()
        reduction_prev = False
        reduce_layers = [depth//3, 2*depth//3]
        for i in range(depth):
            if i in reduce_layers:
                node_c *= 2
                reduction = True
            else:
                reduction = False
            cell = SearchedCell(gene, n_nodes, c0, c1, node_c, reduction, reduction_prev, drop_rate)
            reduction_prev = reduction
            self.cells.append(cell)
            c0, c1 = c1, cell.out_channels

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c1, out_channels)

    def forward(self, x):
        x0 = x1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            x0, x1 = x1, cell(x0, x1)
        out = self.global_pooling(x1)
        y = self.classifier(out.view(out.size(0),-1))
        return y
         