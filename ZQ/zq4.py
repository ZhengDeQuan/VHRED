# -*-coding:utf-8 -*-

import numpy as np
import numpy


eos_sym = 10

def reverse_utterances(seq):
    """
    Reverses the words in each utterance inside a sequence of utterance (e.g. a dialogue)
    This is used for the bidirectional encoder RNN.
    """
    #  输入的seq是一个utterance的list，其中的每一个utterance🈶又需要被reverse
    reversed_seq = numpy.copy(seq)
    for idx in range(seq.shape[1]):
        eos_indices = numpy.where(seq[:, idx] == eos_sym)[0]
        prev_eos_index = -1
        for eos_index in eos_indices:
            reversed_seq[(prev_eos_index + 1):eos_index, idx] = (reversed_seq[(prev_eos_index + 1):eos_index, idx])[
                                                                ::-1]
            # 达到[1,2,3,4,0,0,0] --> [4,3,2,1,0,0,0] ，类似于这样的效果，
            # 但是这个是每一列是一个dialogue
            prev_eos_index = eos_index

    return reversed_seq

def mynanem():
    return 1,10

if __name__ == "__main__":
    # utterance1 = numpy.asarray([1,2,3,4,5,6,7,8,9,10],dtype = "int32")
    # utterance2 = numpy.asarray([1,2,3,4,5,6,7,8,10,0],dtype = "int32")
    # utterance3 = numpy.asarray([1,2,3,4,5,6,7,10,0,0],dtype = "int32")
    # utterance4 = numpy.asarray([1,2,3,4,5,6,10,0,0,0],dtype = "int32")
    # utterance1 = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
    # utterances = utterance1 + utterance1 + utterance1 + utterance1
    # print("utterances = ",utterances)
    # seq = [utterances,utterances,utterances,utterances,utterances]
    # seq = numpy.asarray(seq)
    # seq = numpy.transpose(seq,axes=[1,0,2])
    # print(seq)
    # print(seq.shape)
    # temp = seq[:,1]
    # print("temp = ",temp)
    # print("temp.shape = ",temp.shape)
    # eos_indices = numpy.where(seq[:,1] == eos_sym)[0]
    # print(eos_indices)


    # g = [1,2,3,4,5]
    # g= numpy.asarray(g)
    # print(g)
    # g = g ** 2
    # print(g)
    # a = []
    # import copy
    # a.append(copy.deepcopy(g))
    # a.append(g)
    # print(a)
    # g[2] = 1000
    # print(a)
    # a,b, = mynanem()
    # print("a = ",a," b = ",b)
    # import itertools
    # data_x = [[1,2,3,0],[4,5,6,7,0],[8,9,10,11,12,0],[13,14,15,16,17,18,0]]
    # x = list(itertools.chain(data_x))
    # x = numpy.asarray(list(itertools.chain(data_x)))
    #
    # print("x = ",x)
    # lens = numpy.asarray([map(len, x)])
    # print("lens = ",lens)
    # order = numpy.argsort(lens.max(axis=0))
    # print(lens.max(axis = 0))
    # print("order = ",order)


    # a = ['a','a','b','c']
    # from collections import Counter
    # word_counter = Counter()
    # word_counter.update(a)
    # print(word_counter)
    # vocab_count = word_counter.most_common()
    # print(vocab_count)

    dialogue = [1,2,3,4,5,6,7,8,9,10]
    batch = [[dialogue],[dialogue[:8]],[dialogue[:7]],[dialogue[:6]],[dialogue[:5]]]
    data = [batch,batch,batch,batch]
    k_batch = 4
    batch_size = 5
    seq_len = 10
    for i in range(len(data)):
        print(data[i])
    print("******" * 5)
    import itertools

    data = list(itertools.chain.from_iterable(data))
    for i in range(len(data)):
        print(data[i])

    print("******" * 5)
    data_x = []
    for i in range(len(data)):
        data_x.append(data[i][0])
    print(data_x)

    print("******" * 5)
    temp = list(itertools.chain(data_x))
    print(temp)

    print("******" * 5)
    listone = ['a', 'b', 'c']
    listtwo = ['11', '22', 'abc']
    c = [listone,listtwo]
    tmp =  list(itertools.chain(c))
    print(tmp)

    print("******" * 5)
    x = numpy.asarray(list(itertools.chain(data_x)))
    print("x = ",x)

    print("******" * 5)
    lens = numpy.asarray([map(len, x)])
    print(lens)
    print("lens = ",lens.max(axis = 0))
    print("******" * 5)
    order = numpy.argsort(lens.max(axis=0))#sort操作是在做什么
    print(lens.max(axis = 0))
    print(len(lens.max(axis = 0)))
    print("order = ",order)

    print("******" * 5)
    indices = order[0:4]
    tmp = x[indices]
    tmp2 = [tmp]
    print("tmp = ")
    print(tmp)
    print("tmp2 = ")
    print(tmp2)
    print("tmp2[0] = ")
    print(tmp2[0])

    print("******" * 5)
    x = tmp2
    mx = 0
    for idx in xrange(len(x[0])):
        mx = max(mx, len(x[0][idx]))
    mx += 1

    n = 4#n = state['bs']
    X = numpy.zeros((mx, n), dtype='int32')
    Xmask = numpy.zeros((mx, n), dtype='float32')
    X_reversed = numpy.zeros((mx, n), dtype='int32')
    print(X)
    print(Xmask)
    print(X_reversed)
    print("******" * 5)
    X[:mx + 1, idx] = [10000] + x[0][idx][:mx]
    #X[mx-1:, idx] = 10000
    print("X =")
    print(X)
    # eos_indices = numpy.where(X[:, idx] == 10000)[0]
    # print(eos_indices)
    #
    # a = [1,2,3,4,5,5,6,7,8,8,9]
    # a = numpy.asarray(a)
    # print(a[0:6][::-1])
    #
    # for i in range(0,1):
    #     print("i = ",i)
    #
    # import math
    # tmp = math.ceil(float(1)/float(80))
    # print(int(tmp))