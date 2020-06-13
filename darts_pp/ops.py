import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import ConstantInitializer, MSRAInitializer

# genotype.PRIMITIVES is a subset of OPS 
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: Pool2D(pool_size=3,
                                                     pool_type="avg",
                                                     pool_stride=stride,
                                                     pool_padding=1),
    'max_pool_3x3': lambda C, stride, affine: Pool2D(pool_size=3,
                                                     pool_type="max",
                                                     pool_stride=stride,
                                                     pool_padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine),
}

def bn_param_config(affine=False):
    gama = ParamAttr(
        initializer=ConstantInitializer(value=1), trainable=affine)
    beta = ParamAttr(
        initializer=ConstantInitializer(value=0), trainable=affine)
    return gama, beta


class Zero(fluid.dygraph.Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.pool = Pool2D(pool_size=1, pool_stride=2)

    def forward(self, x):
        pooled = self.pool(x)
        x = fluid.layers.zeros_like(x) if self.stride == 1 else fluid.layers.zeros_like(pooled)
        return x

    
class Identity(fluid.dygraph.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    
class ReLUConvBN(fluid.dygraph.Layer):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.conv = Conv2D(num_channels=c_in,
                           num_filters=c_out,
                           filter_size=kernel_size,
                           stride=stride,
                           padding=padding,
                           param_attr=ParamAttr(initializer=MSRAInitializer()),
                           bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn = BatchNorm(num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = fluid.layers.relu(x)
        x = self.conv(x)
        out = self.bn(x)
        return out

    
class DilConv(fluid.dygraph.Layer):
    '''Dilated + Depthwise Separable Conv'''
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.conv1 = Conv2D(num_channels=c_in,
                            num_filters=c_in,
                            filter_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=c_in,
                            use_cudnn=False,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        self.conv2 = Conv2D(num_channels=c_in,
                            num_filters=c_out,
                            filter_size=1,
                            padding=0,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn = BatchNorm(num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = fluid.layers.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.bn(x)
        return out

    
class SepConv(fluid.dygraph.Layer):
    '''Depthwise Separable Conv'''
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.conv1 = Conv2D(num_channels=c_in,
                            num_filters=c_in,
                            filter_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            groups=c_in,
                            use_cudnn=False,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        self.conv2 = Conv2D(num_channels=c_in,
                            num_filters=c_in,
                            filter_size=1,
                            stride=1,
                            padding=0,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn1 = BatchNorm(num_channels=c_in, param_attr=gama, bias_attr=beta)
        self.conv3 = Conv2D(num_channels=c_in,
                            num_filters=c_in,
                            filter_size=kernel_size,
                            stride=1,
                            padding=padding,
                            groups=c_in,
                            use_cudnn=False,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        self.conv4 = Conv2D(num_channels=c_in,
                            num_filters=c_out,
                            filter_size=1,
                            stride=1,
                            padding=0,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn2 = BatchNorm(num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = fluid.layers.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        bn1 = self.bn1(x)
        x = fluid.layers.relu(bn1)
        x = self.conv3(x)
        x = self.conv4(x)
        bn2 = self.bn2(x)
        return bn2


class FactorizedReduce(fluid.dygraph.Layer):
    def __init__(self, c_in, c_out, affine=True):
        super().__init__()
        assert c_out % 2 == 0
        self.conv1 = Conv2D(num_channels=c_in,
                            num_filters=c_out // 2,
                            filter_size=1,
                            stride=2,
                            padding=0,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        self.conv2 = Conv2D(num_channels=c_in,
                            num_filters=c_out // 2,
                            filter_size=1,
                            stride=2,
                            padding=0,
                            param_attr=ParamAttr(initializer=MSRAInitializer()),
                            bias_attr=False)
        gama, beta = bn_param_config(affine)
        self.bn = BatchNorm(num_channels=c_out, param_attr=gama, bias_attr=beta)

    def forward(self, x):
        x = fluid.layers.relu(x)
        out = fluid.layers.concat([self.conv1(x), self.conv2(x)], axis=1)
        out = self.bn(out)
        return out
    
    
