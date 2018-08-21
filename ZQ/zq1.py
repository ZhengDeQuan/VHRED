# -*-coding:utf-8 -*-
import numpy as np
import numpy
eos_sym = 0
seq = numpy.identity(10,dtype=np.int32)
idx = 2
print(seq)
print(len(seq))
print(seq.shape)
print(seq.shape[0])
print(seq.shape[1])


print(seq[:,2])
for i in range(seq.shape[0]):
    seq[i][2] = i

print(seq)
#eos_indices = numpy.where(seq[:, idx] == eos_sym)[0]
