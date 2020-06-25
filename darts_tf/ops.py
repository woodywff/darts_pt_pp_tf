import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Layer, Conv2D, SeparableConv2D, MaxPool2D, 
                                     AveragePooling2D, ReLU, BatchNormalization)

# genotype.PRIMITIVES is a subset of OPS 
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: AveragePooling2D(3, strides=stride, padding='same'),
    'max_pool_3x3': lambda C, stride, affine: MaxPool2D(3, strides=stride, padding='same'),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, 3, stride, affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, 5, stride, affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, 7, stride, affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, 3, stride, 2, affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, 5, stride, 2, affine),
}

def l2reg():
    '''
    We didn't add l2 regularization in this version.
    '''
#     return tf.keras.regularizers.l2()
    return None

def BatchNorm(affine=True):
    '''
    affine: a boolean value that when set to True, this module has
        learnable affine parameters (\gamma, \beta).
    '''
    return BatchNormalization(center=affine, scale=affine)

class Zero(Layer):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def call(self, x):
        if self.strides == 1:
            return x * 0.
        return x[:, ::self.strides, ::self.strides, :] * 0

    
class Identity(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x
    
class FactorizedReduce(Layer):
    def __init__(self, c_out, affine=True, padding='valid'):
        super().__init__()
        assert c_out % 2 == 0
        self.relu = ReLU()
        self.conv1 = Conv2D(filters=c_out // 2,
                            kernel_size=1,
                            strides=2,
                            padding=padding,
                            use_bias=False,
                            kernel_regularizer=l2reg())
        self.conv2 = Conv2D(filters=c_out // 2,
                            kernel_size=1,
                            strides=2,
                            padding=padding,
                            use_bias=False,
                            kernel_regularizer=l2reg())
        self.bn = BatchNorm(affine=affine)

    def call(self, x):
        x = self.relu(x)
        out = tf.concat([self.conv1(x), self.conv2(x)], axis=-1)
        out = self.bn(out)
        return out

class SepConv(Layer):
    '''Depthwise Separable Conv'''
    def __init__(self, c_out, kernel_size, stride, affine=True, padding='same'):
        super().__init__()
        self.op = Sequential([
            ReLU(),
            SeparableConv2D(filters=c_out, 
                            kernel_size=kernel_size, 
                            strides=stride,
                            depthwise_regularizer=l2reg(),
                            pointwise_regularizer=l2reg(),
                            padding=padding, use_bias=False),
            BatchNorm(affine=affine),
            ReLU(),
            SeparableConv2D(filters=c_out, 
                            kernel_size=kernel_size, 
                            strides=1,
                            depthwise_regularizer=l2reg(),
                            pointwise_regularizer=l2reg(),
                            padding=padding, use_bias=False),
            BatchNorm(affine=affine)])

    def call(self, x):
        return self.op(x)


class DilConv(Layer):
    '''Dilated + Depthwise Separable Conv'''
    def __init__(self, c_out, kernel_size, stride, dilation, affine=True, padding='same'):
        super().__init__()
        self.op = Sequential([
            ReLU(),
            SeparableConv2D(filters=c_out, 
                            kernel_size=kernel_size, 
                            strides=stride,
                            depthwise_regularizer=l2reg(),
                            pointwise_regularizer=l2reg(),
                            dilation_rate=dilation, 
                            padding=padding, use_bias=False),
            BatchNorm(affine=affine)])

    def call(self, x):
        return self.op(x)

    
class ReLUConvBN(Layer):
    def __init__(self, c_out, kernel_size, stride, affine=True, padding='valid'):
        super().__init__()
        self.op = Sequential([
            ReLU(),
            Conv2D(filters=c_out, 
                   kernel_size=kernel_size, 
                   strides=stride,
                   kernel_regularizer=l2reg(),
                   padding=padding, 
                   use_bias=False),
            BatchNorm(affine=affine)])

    def call(self, x):
        return self.op(x)

    



    
