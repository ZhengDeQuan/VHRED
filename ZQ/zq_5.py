# -*- coding:utf-8 -*-
import numpy
rng = numpy.random.RandomState(1234)
sizeX = 10
sizeY = 10
sparsity = sizeY
values = numpy.zeros((sizeX, sizeY), dtype="float32")
sparsity = numpy.minimum(sizeY, sparsity)

for dx in xrange(sizeX):
    perm = rng.permutation(sizeY)
    new_vals = rng.normal(loc=0, scale=0.01, size=(sparsity,))
    values[dx, perm[:sparsity]] = new_vals
    print(perm)
    print(perm[:sparsity])
    print(len(perm[:sparsity]))
    print(len(perm))
    print(len(new_vals))
    print(new_vals)
    print(values[dx,perm[:sparsity]])

print(values)