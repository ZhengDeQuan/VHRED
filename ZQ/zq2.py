# -*- coding:utf-8 -*-

import theano
import numpy
from theano import tensor as T
from theano.tensor.nnet import conv

# 以下定义只是为了能执行conv2d操作
rng = numpy.random.RandomState(23455)
input = T.tensor4(name='input', dtype='float32')
W = theano.shared(numpy.asarray(
    rng.uniform(
        low=-1.0,
        high=1.0,
        size=(2, 1, 1, 1)),
    dtype='float32'), name='W')
b = theano.shared(numpy.asarray(
    rng.uniform(low=-1, high=1, size=(2,)),
    dtype='float32'), name='b')
conv_out = conv.conv2d(input, W)

########################以下是主要代码，上边代码是借用的python实现卷积的#############################

# 重新定义一个shared变量
b1 = theano.shared(numpy.arange(3).reshape(1, 3), name='b1')

#######一维变换#######
# bb = b.dimshuffle(0)           # 输出b原始结构信息，即shape为(2, )
# bb = b.dimshuffle(0, 'x')      # 输出两维数据，其shape为(2, 1)
# bb = b.dimshuffle('x', 0)      # 输出两维数据，其shape为(1, 2)
# bb = b.dimshuffle(0, 'x', 'x') # 输出三维数据，其shape为(2, 1, 1)
# bb = b.dimshuffle('x', 0, 'x') # 输出三维数据，其shape为(1, 2, 1)
# bb = b.dimshuffle('x', 'x', 0) # 输出三维数据，其shape为(1, 1, 2)

#######二维变换#######
# bb = b1.dimshuffle(0, 1)       # 输出b1原始结构信息，即shape为(1, 3)
# bb = b1.dimshuffle(1, 0)       # 输出b1原始结构信息，即shape为(3, 1)
# bb = b1.dimshuffle(0, 1, 'x')  # 输出b1三维信息，即shape为（1，3，1）
# bb = b1.dimshuffle(0, 'x', 1)  # 输出b1三维信息，即shape为（1，1，3）
# bb = b1.dimshuffle('x', 0, 1)  # 输出b1三维信息，即shape为（1，1，3）
# bb = b1.dimshuffle(1, 0, 'x')  # 输出b1三维信息，即shape为（3，1，1）
# bb = b1.dimshuffle(1, 'x', 0)  # 输出b1三维信息，即shape为（3，1，1）
# bb = b1.dimshuffle('x', 1, 0)  # 输出b1三维信息，即shape为（1，3，1）
# bb = b1.dimshuffle('x', 1, 'x', 0) # 输出b1原始结构信息，即shape为（1，3，1，1）
bb = b1.dimshuffle('x', 0, 'x', 1)  # 输出b1原始结构信息，即shape为（1，3，1，1）

bb_printed = theano.printing.Print('b_1.dimshuffle')(bb)

output = [T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x')), bb_printed * 2]
########################以上是主要测试代码#############################

f = theano.function([input], output)
img2 = numpy.arange(100, dtype='float32').reshape(1, 1, 10, 10)
filtered_img = f(img2)[0]