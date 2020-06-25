from .ops import OPS, FactorizedReduce, ReLUConvBN, BatchNorm, l2reg
import pdb
import tensorflow as tf
from tensorflow.keras.layers import (Layer, Dense, Conv2D, Dropout,
                                     GlobalAveragePooling2D, ZeroPadding2D)
from tensorflow.keras import Model, Sequential

FLAG_DEBUG = False

class SearchedCell(Layer):
    def __init__(self, gene, n_nodes, node_c, reduction, reduction_prev, drop_rate=0):
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
            self.preprocess0 = FactorizedReduce(node_c)
        else:
            self.preprocess0 = ReLUConvBN(node_c, 1, 1)
        self.preprocess1 = ReLUConvBN(node_c, 1, 1)
        
        self._ops = [OPS[i[0]](C=node_c, 
                               stride=2 if reduction and i[1] < 2 else 1, 
                               affine=True) for i in self.genolist]
        
        return

#     @property
#     def out_channels(self):
#         return self.n_nodes * self.node_c

    def call(self, x0, x1):
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
                temp = Dropout(self.drop_rate)(temp)
                outputs.append(temp)
                i += 1
            xs.append(sum(outputs)) 
        return tf.concat(xs[-self.n_nodes:], axis=-1) 
            

class SearchedNet(Model):
    def __init__(self, gene, img_size, in_channels, init_node_c, out_channels, depth, n_nodes, drop_rate=0):
        '''
        gene: Genotype, searched architecture of a cell.
        img_size: image length or width (length==width).
        in_channels: RGB channel.
        init_node_c: Initial number of filters or output channels for the node.
        out_channels: How many classes are there in the target.
        depth: Number of cells.
        n_nodes: Number of nodes in each cell.
        drop_rate: dropout rate.
        '''
        super().__init__()
        self.zero_pad = ZeroPadding2D(2) if img_size == 28 else None 
        stem_c = min(in_channels, n_nodes) * init_node_c # stem out_channels
        self.stem = Sequential([
            Conv2D(filters=stem_c, kernel_size=3, strides=1, padding='same',
                   kernel_regularizer=l2reg(), use_bias=False),
            BatchNorm(affine=True)
        ])
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
            cell = SearchedCell(gene, n_nodes, node_c, reduction, reduction_prev, drop_rate)
            reduction_prev = reduction
            self.cells.append(cell)

        self.global_pooling = GlobalAveragePooling2D()
        self.classifier = Dense(out_channels, 
                                kernel_regularizer=l2reg())


    def call(self, x):
        if self.zero_pad is not None:
            x = self.zero_pad(x)
        x0 = x1 = self.stem(x)
        for i, cell in enumerate(self.cells):
            x0, x1 = x1, cell(x0, x1)
        out = self.global_pooling(x1)
        y = self.classifier(out)
        return y
         