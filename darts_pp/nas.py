from .cell import Cell
import pdb
from genotype import Genotype, GenoParser, PRIMITIVES
from paddle.fluid.dygraph import Layer, LayerList, Sequential
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import NormalInitializer, MSRAInitializer, ConstantInitializer
import paddle.fluid as fluid
FLAG_DEBUG = False

class KernelNet(Layer):
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
        self.stem = Sequential(
          Conv2D(in_channels, stem_c, 3, padding=1, 
                 param_attr=ParamAttr(initializer=MSRAInitializer()), 
                 bias_attr=False),
          BatchNorm(stem_c)
        )
        c0 = c1 = stem_c
        node_c = init_node_c # node out_channels
        self.cells = LayerList()
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

        self.global_pooling = Pool2D(pool_type='avg', global_pooling=True)
        self.classifier = Linear(input_dim=c1,
                                 output_dim=out_channels,
                                 param_attr=ParamAttr(initializer=MSRAInitializer()),
                                 bias_attr=ParamAttr(initializer=MSRAInitializer()))

    def forward(self, x, alphas_normal, alphas_reduce):
#         pdb.set_trace()
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
    

        