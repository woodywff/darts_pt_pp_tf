import torch
import torch.nn as nn
from ops import PRIMITIVES
from cell import Cell
from torch.functional import F
import pdb
from genotype import Genotype, GenoParser

FLAG_DEBUG = False

class KernelNet(nn.Module):
    def __init__(self, in_channels, init_node_c, out_channels, depth, n_nodes):
        '''
        in_channels: RGB channel.
        init_node_c: Initial number of filters or output channels for the node.
        out_channels: How many classes are there in the target.
        depth: Number of cells.
        n_nodes: Number of nodes in each cell.
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
            cell = Cell(n_nodes, c0, c1, node_c, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            c0, c1 = c1, cell.out_channels

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c1, out_channels)

    def forward(self, x, alphas_normal, alphas_reduce):
        x0 = x1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                alphas = F.softmax(alphas_reduce, dim=-1)
            else:
                alphas = F.softmax(alphas_normal, dim=-1)
            x0, x1 = x1, cell(x0, x1, alphas)
        out = self.global_pooling(x1)
        y = self.classifier(out.view(out.size(0),-1))
        return y
    
    
class ShellNet(nn.Module):
    def __init__(self, in_channels, init_node_c, out_channels, depth, n_nodes):
        '''
        This class defines the outlayer of NAS.
        in_channels: RGB channel.
        init_node_c: Initial number of filters or output channels for the node.
        out_channels: How many classes are there in the target.
        depth: Number of cells.
        n_nodes: Number of nodes in each cell.
        '''
        super().__init__()
        self.n_nodes = n_nodes

        self.kernel = KernelNet(in_channels, init_node_c, out_channels, depth, n_nodes)
        self._init_alphas()
        
    def _init_alphas(self):
        n_ops = sum(range(2, 2 + self.n_nodes))
        self.alphas_normal  = nn.Parameter(torch.zeros((n_ops, len(PRIMITIVES))))
        self.alphas_reduce  = nn.Parameter(torch.zeros((n_ops, len(PRIMITIVES))))
        # setup alphas list
        self._alphas = [(name, param) for name, param in self.named_parameters() if 'alpha' in name]
        
    def alphas(self):
        for _, param in self._alphas:
            yield param

    def forward(self, x):
        return self.kernel(x, self.alphas_normal, self.alphas_reduce) 
    
    def get_gene(self):
        geno_parser = GenoParser(self.n_nodes)
        gene_normal = geno_parser.parse(self.alphas_normal)
        gene_reduce = geno_parser.parse(self.alphas_reduce)

        return Genotype(normal=gene_normal, reduce=gene_reduce) 
    

        