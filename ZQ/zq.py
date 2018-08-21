# -*-coding:utf-8 -*-
import theano
import numpy
from theano import tensor as T
from theano.tensor.nnet import conv


rng = numpy.random.RandomState(23455)

#以下定义只是为了能执行conv2d操作
input = T.tensor4(name = 'input',dtype = 'float32')
W = theano.shared(value = numpy.asarray(
    rng.uniform(
        low = -1.0,
        high=1.0,
        size=(2,1,1,1)
    ),
    dtype = 'float32') ,name = "W")

b = theano.shared(value = numpy.asarray(
    rng.uniform(low = -1 , high=1, size=(2,)),
    dtype = 'float32'),
    name = 'b')

conv_out = conv.conv2d(input, W)

b1 = theano.shared(numpy.arange(3).reshape(1,3),name='b1')

bb = b1.dimshuffle('x',0,'x',1)
bb_printed = theano.printing.Print('b_1.dimshuffle')(bb)

output = [T.nnet.sigmoid(conv_out + b.dimshuffle('x',0,'x','x')),bb_printed * 2,bb,b1]

f = theano.function([input] , output)
img2 = numpy.arange(100,dtype="float32").reshape(1,1,10,10)
filtered_img = f(img2)[3]
print(filtered_img)