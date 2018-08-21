# -*- coding:utf-8 -*-

import theano
import numpy
import theano.tensor as T
import numpy as np

a = numpy.asarray([[1,2,3,4,5,0],[1,2,3,4,5,0],[1,2,3,4,5,0],[1,2,3,4,5,0]]) #4个utternace 1个dialogue
# print(a)
# Ta = T.matrix('Ta')
# Tres1 = T.neq(Ta,0)
# fun1 = theano.function(inputs = [Ta],
#                       outputs = Tres1)
# res = fun1(a)
# #print(res)
# Tres2 = Ta.flatten()
# fun2 = theano.function(inputs = [Ta],
#                        outputs=Tres2)
# res2 = fun2(a)
# print(res2)
#
# h_0 = T.alloc(np.int32(0),2,10)
# h_0 = T.alloc(np.float32(1),2,10)
# fun3 = theano.function(inputs = [], outputs = h_0)
# res3 = fun3()
# print(res3)
#
# b = theano.shared(value = numpy.zeros(shape = (3,5),dtype="int32"))
# a = b[0]
# #a = a.dimshuffle(0,'x')
# fun4 = theano.function(inputs = [],outputs = a)
# res4 = fun4()
# print(res4)
#
#
# args = ['h_0', None, None, None]
# args = iter(args)
# h_tm1 = next(args)
# print("h_tm1 = ",h_tm1)


embeding_size = 6
dialogue_length = 4
batch_size = 3
one_mask = [0,0,0,1]#因为dialogue_length = 4
mask = [one_mask,one_mask,one_mask] #因为batch_size = 3
mask = numpy.asarray(mask)
aa = [a,a,a]
aa = numpy.asarray(aa)
print("aa = ",aa)

bb = aa + 1
print("bb = ",bb)


A = T.tensor3("A")
B = T.tensor3("B")
Tmask = T.matrix("mask")
mean= T.mean(A,axis = 2)
tmp = Tmask * mean
sum = T.sum(Tmask * mean)#并没有指定axis，那么就是在所有的维度上进行求和，二维变成0为，变为一个数
maskSum = T.sum(Tmask)
average = sum/maskSum
fun5 = theano.function(inputs=[A,Tmask],outputs =[sum,tmp,average])

# sum,tmp , average= fun5(aa,mask)
# print(sum.ndim)
# print(sum.shape)
# print(sum)
#
# print(tmp.ndim)
# print(tmp.shape)
# print(tmp)
#
# print("average = ",average)

detle = A - B
sum = A + B
prod = A * B
quad = A / B

fun6 = theano.function(inputs = [A,B],outputs = [detle,sum,prod,quad])
detle,sum,prod,quad = fun6(aa,bb)
print "detle = "
print detle
print "sum = "
print sum
print "prod = "
print prod
print "quad = "
print quad



