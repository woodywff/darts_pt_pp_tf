from .cell import Cell
import pdb
from genotype import Genotype, GenoParser, PRIMITIVES
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import (Layer, Input, Dense, Flatten, Conv2D, MaxPool2D,
                                     GlobalAveragePooling2D, Softmax, ZeroPadding2D)
from .ops import OPS, FactorizedReduce, ReLUConvBN, BatchNorm, l2reg

FLAG_DEBUG = False

class KernelNet(Model):
    def __init__(self, img_size, in_channels, init_node_c, out_channels, depth, n_nodes):
        '''
        img_size: image length or width (length==width)
        in_channels: RGB channel.
        init_node_c: Initial number of filters or output channels for the node.
        out_channels: How many classes are there in the target.
        depth: Number of cells.
        n_nodes: Number of nodes in each cell.
        '''
        super().__init__()
        self.zero_pad = ZeroPadding2D(2) if img_size == 28 else None 
        stem_c = min(in_channels, n_nodes) * init_node_c # stem out_channels
        self.stem = Sequential([
            Conv2D(filters=stem_c, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2reg(), use_bias=False),
            BatchNorm(affine=True)
        ])
#         c0 = c1 = stem_c
        node_c = init_node_c # node out_channels
        self.cells = []
        reduction_prev = False
        reduce_layers = [depth//3, 2*depth//3]
        for i in range(depth):
            if i in reduce_layers:
                node_c *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(n_nodes, node_c, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
#             c0, c1 = c1, cell.out_channels

        self.global_pooling = GlobalAveragePooling2D()
        self.classifier = Dense(out_channels, 
                                kernel_regularizer=l2reg())

    def forward(self, x, alphas_normal, alphas_reduce):
#         pdb.set_trace()
        if self.zero_pad is not None:
            x = self.zero_pad(x) 
        x0 = x1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                alphas = fluid.layers.softmax(alphas_reduce)
            else:
                alphas = fluid.layers.softmax(alphas_normal)
            x0, x1 = x1, cell(x0, x1, alphas)
        out = self.global_pooling(x1)
        out = fluid.layers.squeeze(out, axes=[-1,-2])
        y = self.classifier(out)
        return y
    
    
class ShellNet(Layer):
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
        self.alphas_normal  = fluid.layers.create_parameter(shape=[n_ops, len(PRIMITIVES)],
                                                            dtype="float32",
                                                            default_initializer=ConstantInitializer(value=0))
        self.alphas_reduce  = fluid.layers.create_parameter(shape=[n_ops, len(PRIMITIVES)],
                                                            dtype="float32",
                                                            default_initializer=ConstantInitializer(value=0))
        # setup alphas list
        self._alphas = [self.alphas_normal, self.alphas_reduce]
        
    def alphas(self):
        for param in self._alphas:
            yield param

    def forward(self, x):
        return self.kernel(x, self.alphas_normal, self.alphas_reduce) 
    
    def get_gene(self):
        geno_parser = GenoParser(self.n_nodes)
        gene_normal = geno_parser.parse(fluid.layers.softmax(self.alphas_normal).numpy())
        gene_reduce = geno_parser.parse(fluid.layers.softmax(self.alphas_reduce).numpy())

        return Genotype(normal=gene_normal, reduce=gene_reduce) 
    

        