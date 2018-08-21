# -*- coding:utf-8 -*-
'''
函数定义如下

output, update = theano.scan(fn, sequences=None, outputs_info=None, non_sequences=None,n_steps=None,
truncate_gradient=-1, go_backwards=False, mode=None,name=None, profile=False, allow_gc=None, strict=False)
fn就是被执行循环的函数，它接收。sequences是一个变量或者若干个变量组成的list，换句话说，就是for循环里的那个i or j。
outputs_info是循环函数输出的初始值。non_sequences性质与sequences一样，都是指定一个或多个变量名，但是这些变量在循环过程中是作为常量。
n_step指定了循环迭代次数。truncate_gradient与计算梯度有关，-1是指不限制反向传播的长度，反之则在反向传播到一定距离后停止。
go_backwards是指sequences是正向还是反向读取。
'''

'''
关键内容来了，fn的传入参数列表是什么，若scan的写法如下：
'''
"""
scan(fn, sequences = [ dict(input= Sequence1, taps = [-3,2,-1])

                     , Sequence2

                     , dict(input =  Sequence3, taps = 3) ]

       , outputs_info = [ dict(initial =  Output1, taps = [-3,-5])

                        , dict(initial = Output2, taps = None)

                        , Output3 ]
       , non_sequences = [ Argument1, Argument2])
"""

'''
 那么传给fn的参数顺序就是
 1 Sequence1[t-3]
 2 Sequence1[t+2]
 3 Sequence1[t-1]
 4 Sequence2[t]
 5 Sequence3[t+3]
 6 Output1[t-3]
 7 Output1[t-5]
 8 Output3[t-1]
 9 Argument1  
 10 Argument2 
'''
import theano
import theano.tensor as T
import numpy

def ZQ(A,B,C):
    print("A = ",A.ndim,"  A.shape = ",A.shape)
    print("B = ",B.ndim)
    print("C = ",C.ndim)
    D = C
    return A,B,C,D

A = theano.shared(numpy.ones((5,3,3)),borrow = True)
B = theano.shared(numpy.zeros((5,3,3)),borrow = True)
C = theano.shared(numpy.ones((3,3)),borrow = True)
h0 = [C , None ,None ,None]
result , updates = theano.scan(fn = ZQ  ,
                            sequences = [A,B],
                             outputs_info=h0)


fun1 = theano.function(inputs = [],
                       outputs = result)

a = numpy.ones(3)
b = numpy.zeros(3)
c = numpy.ones(3)
aa = [a,a,a,a,a]
bb = [b,b,b,b,b]
res = fun1()
print("res = ",res)


