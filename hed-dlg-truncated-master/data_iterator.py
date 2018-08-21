# -*- coding:utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

import sys, getopt
import logging

from state import *
from utils import *
from SS_dataset import *

import itertools
import sys
import pickle
import random
import datetime
import math
import copy

logger = logging.getLogger(__name__)


def add_random_variables_to_batch(state, rng, batch, prev_batch, evaluate_mode):
    #外部调用：batch = add_random_variables_to_batch(self.state, self.rng, batch, self.prev_batch, self.evaluate_mode)
    """
    This is a helper function, which adds random variables to a batch.
    We do it this way, because we want to avoid Theano's random sampling both to speed up and to avoid
    known Theano issues with sampling inside scan loops.

    The random variable 'ran_var_constutterance' is sampled from a standard Gaussian distribution, 
    which remains constant during each utterance (i.e. between end-of-utterance tokens).
    
    When not in evaluate mode, the random vector 'ran_decoder_drop_mask' is also sampled. 
    This variable represents the input tokens which are replaced by unk when given to 
    the decoder RNN. It is required for the noise addition trick used by Bowman et al. (2015).
    """

    # If none return none
    if not batch:
        return batch

    # Variable to store random vector sampled at the beginning of each utterance
    Ran_Var_ConstUtterance = numpy.zeros((batch['x'].shape[0], batch['x'].shape[1], state['latent_gaussian_per_utterance_dim']), dtype='float32')

    # Go through each sample, find end-of-utterance indices and sample random variables
    for idx in xrange(batch['x'].shape[1]):
        # Find end-of-utterance indices
        eos_indices = numpy.where(batch['x'][:, idx] == state['eos_sym'])[0].tolist()

        # Make sure we also sample at the beginning of the utterance, and that we stop appropriately at the end
        if len(eos_indices) > 0:
            if not eos_indices[0] == 0:
                eos_indices = [0] + eos_indices
            if not eos_indices[-1] == batch['x'].shape[0]:#最后不是padded 0了吗？所以shape[0],是最后补了sym_eos之后的长度
                                                          #还真不一定，因为在padded函数外面，将一个full_batch,中每个长度超过80的句子都剪断成了几段，这个没一段的前后是没有padd zero的
                eos_indices = eos_indices + [batch['x'].shape[0]]
        else:#这种情况对应于一个utterance的长度就超过了80
            eos_indices = [0] + [batch['x'].shape[0]]

        # Sample random variables using NumPy
        ran_vectors = rng.normal(loc=0, scale=1, size=(len(eos_indices), state['latent_gaussian_per_utterance_dim']))
        #均值为loc = 0  scale = Standard deviation (spread or “width”) of the distribution.
        for i in range(len(eos_indices)-1):
            for j in range(eos_indices[i], eos_indices[i+1]):
                Ran_Var_ConstUtterance[j, idx, :] = ran_vectors[i, :] #第i段话（utterance）的每个单词，都用同样的random_vector,来作为隐变量

        # If a previous batch is given, and the last utterance in the previous batch
        # overlaps with the first utterance in the current batch, then we need to copy over
        # the random variables from the last utterance in the last batch to remain consistent.

        #这个，上一个batch中的最后一句utterance 跟当前的batch中的第一句utterance 发生overlaps 的情况是怎么回事啊，
        #哦，我明白了，因为这里面赋予随机隐变量的逻辑是，每个句子（utterance）中的token都赋予同一个 隐变量，所以之前按照80个token，就一个切割的话，那么最后的一句话，很有可能被切成前后两部分，

        if prev_batch:
            if ('x_reset' in prev_batch) and (not numpy.sum(numpy.abs(prev_batch['x_reset'])) < 1) \
              and ('ran_var_constutterance' in prev_batch):
                prev_ran_vector = prev_batch['ran_var_constutterance'][-1,idx,:] #最后一行，当前的idx列，的这个token的，隐向量
                if len(eos_indices) > 1:
                    for j in range(0, eos_indices[1]):#因为eos_indices[0] 经过前面的逻辑一定是0
                        Ran_Var_ConstUtterance[j, idx, :] = prev_ran_vector
                else: #经过了前面的逻辑判断，那么能走到这个分支的，就只有 len(eos_indices) == 1，但是这个是不可能的啊，经过了上面的if len(eos_indices) > 0 的判断之后，eos_indices要么原来就是，要么被修改为len == 2
                    for j in range(0, batch['x'].shape[0]):
                        Ran_Var_ConstUtterance[j, idx, :] = prev_ran_vector

    # Add new random Gaussian variable to batch
    batch['ran_var_constutterance'] = Ran_Var_ConstUtterance

    # Create word drop mask based on 'decoder_drop_previous_input_tokens_rate' option:
    if evaluate_mode:
        batch['ran_decoder_drop_mask'] = numpy.ones((batch['x'].shape[0], batch['x'].shape[1]), dtype='float32')
    else:
        if state.get('decoder_drop_previous_input_tokens', False):
            ran_drop = rng.uniform(size=(batch['x'].shape[0], batch['x'].shape[1]))
            batch['ran_decoder_drop_mask'] = (ran_drop <= state['decoder_drop_previous_input_tokens_rate']).astype('float32')
        else:
            batch['ran_decoder_drop_mask'] = numpy.ones((batch['x'].shape[0], batch['x'].shape[1]), dtype='float32')


    return batch


def create_padded_batch(state, rng, x, force_end_of_utterance_token = False):
    #外部调用的格式
    #full_batch = create_padded_batch(self.state, self.rng, [x[indices]])
    # Find max length in batch
    mx = 0
    for idx in xrange(len(x[0])):
        mx = max(mx, len(x[0][idx]))

    # Take into account that sometimes we need to add the end-of-utterance symbol at the start
    # 我没想到在什么情况下，需要在句子的开头加入结尾符号，难道是双向LSTM的情况吗，在utterance_encoder中，双向rnn的话那么在句子的开头不就需要一个结尾符么，所以mx 要+1
    mx += 1

    n = state['bs'] 
    
    X = numpy.zeros((mx, n), dtype='int32')
    Xmask = numpy.zeros((mx, n), dtype='float32') 

    # Variable to store each utterance in reverse form (for bidirectional RNNs)
    X_reversed = numpy.zeros((mx, n), dtype='int32')

    # Fill X and Xmask.
    # Keep track of number of predictions and maximum dialogue length.
    num_preds = 0
    max_length = 0
    for idx in xrange(len(x[0])):
        # Insert sequence idx in a column of matrix X
        dialogue_length = len(x[0][idx])

        # Fiddle-it if it is too long ..
        if mx < dialogue_length: 
            continue

        # Make sure end-of-utterance symbol is at beginning of dialogue.
        # This will force model to generate first utterance too
        if not x[0][idx][0] == state['eos_sym']:
            X[:dialogue_length+1, idx] = [state['eos_sym']] + x[0][idx][:dialogue_length]
            #X的所有行的idx列，添加内容。即完成了行列转置，又完成了在开头添加state[state['eos_sym']]
            dialogue_length = dialogue_length + 1
        else:
            X[:dialogue_length, idx] = x[0][idx][:dialogue_length]

        # Keep track of longest dialogue
        max_length = max(max_length, dialogue_length)

        # Set the number of predictions == sum(Xmask), for cost purposes, minus one (to exclude first eos symbol)
        num_preds += dialogue_length - 1
        
        # Mark the end of phrase
        if len(x[0][idx]) < mx:
            if force_end_of_utterance_token: #调用的时候没有显示要求这个，所以默认是false
                X[dialogue_length:, idx] = state['eos_sym']#my model : state['eos_sym'] = 1

        # Initialize Xmask column with ones in all positions that
        # were just set in X (except for first eos symbol, because we are not evaluating this). 
        # Note: if we need mask to depend on tokens inside X, then we need to 
        # create a corresponding mask for X_reversed and send it further in the model
        Xmask[0:dialogue_length, idx] = 1.
        #idx这一列，0:dialogue_length的这些行都设置为1. 即有真实值的都是1.

        # Reverse all utterances
        # TODO: For backward compatibility. This should be removed in future versions
        # i.e. move all the x_reversed computations to the model itself.
        eos_indices = numpy.where(X[:, idx] == state['eos_sym'])[0]
        X_reversed[:, idx] = X[:, idx]
        prev_eos_index = -1
        for eos_index in eos_indices:
            X_reversed[(prev_eos_index+1):eos_index, idx] = (X_reversed[(prev_eos_index+1):eos_index, idx])[::-1]
            #将有值的部分，reverse，但是没有值的部分，padded 0 的部分不动，而且是每个dialogue内部进行reverse，整个dialogue的对话的顺序不变
            prev_eos_index = eos_index
            if prev_eos_index > dialogue_length:
                break


    assert num_preds == numpy.sum(Xmask) - numpy.sum(Xmask[0, :]) #在每个dialogu的第一句utternace的开头会加上一个sym_eos

    batch = {'x': X,                                                 \
             'x_reversed': X_reversed,                               \
             'x_mask': Xmask,                                        \
             'num_preds': num_preds,                                 \
             'num_dialogues': len(x[0]),                             \
             'max_length': max_length                                \
            }

    return batch

class Iterator(SSIterator):
    def __init__(self, dialogue_file, batch_size, **kwargs):
        SSIterator.__init__(self, dialogue_file, batch_size,                          \
                            seed=kwargs.pop('seed', 1234),                            \
                            max_len=kwargs.pop('max_len', -1),                        \
                            use_infinite_loop=kwargs.pop('use_infinite_loop', False))

        self.k_batches = kwargs.pop('sort_k_batches', 20)#state中输入的也是20# Sort by length groups of
        self.state = kwargs.pop('state', None)

        self.batch_iter = None
        self.rng = numpy.random.RandomState(self.state['seed'])

        # Keep track of previous batch, because this is needed to specify random variables
        self.prev_batch = None

        # Store whether the iterator operates in evaluate mode or not
        self.evaluate_mode = kwargs.pop('evaluate_mode', False)
        print 'Data Iterator Evaluate Mode: ', self.evaluate_mode

    def get_homogenous_batch_iter(self, batch_size = -1):
        while True:
            batch_size = self.batch_size if (batch_size == -1) else batch_size 
           
            data = []
            for k in range(self.k_batches):
                batch = SSIterator.next(self)
                if batch:
                    data.append(batch)
            
            if not len(data):
                return
            
            number_of_batches = len(data)
            data = list(itertools.chain.from_iterable(data))

            # Split list of words from the dialogue index
            data_x = []
            for i in range(len(data)):
                data_x.append(data[i][0]) #取第0个这个操作就将SS_dataset.py中class SSFetcher(threading.Thread):的run()函数中的dialogues.append([s])这里面的[s]外的[]消解掉了
            #最终data_x  是一个二维列表，每一个子列表中装一个dialogue

            x = numpy.asarray(list(itertools.chain(data_x)))
            #二维矩阵，每一行是一个dialogue

            lens = numpy.asarray([map(len, x)])
            #[[10  8  7  6  5 10  8  7  6  5 10  8  7  6  5 10  8  7  6  5]]
            #lens.max(axis = 0) 得到的结果 [10  8  7  6  5 10  8  7  6  5 10  8  7  6  5 10  8  7  6  5]
            order = numpy.argsort(lens.max(axis=0))
            #order 的结果是，[ 9 14  4 19  8 18 13  3  2  7 12 17  6 11  1 16  5 10 15  0]，就是从小到大排序之后，返回拍好序的索引
            for k in range(number_of_batches):  #number_of_batches == len(data),就是bacth的数量
                indices = order[k * batch_size:(k + 1) * batch_size]
                full_batch = create_padded_batch(self.state, self.rng, [x[indices]]) #这里返回的是一个完整的batch，装在字典里，有X，X_reversed,X_mask
                # Then split batches to have size 'max_grad_steps'
                splits = int(math.ceil(float(full_batch['max_length']) / float(self.state['max_grad_steps'])))
                #因为使用了math.ceil()，所以splits的最小值是1
                batches = []
                for i in range(0, splits):
                    batch = copy.deepcopy(full_batch)

                    # Retrieve start and end position (index) of current mini-batch
                    start_pos = self.state['max_grad_steps'] * i
                    if start_pos > 0:
                        start_pos = start_pos - 1 #比如 80 * 1 , 那么就应该从79这个index的位置开始算

                    # We need to copy over the last token from each batch onto the next, 
                    # because this is what the model expects.
                    end_pos = min(full_batch['max_length'], self.state['max_grad_steps'] * (i + 1))

                    batch['x'] = full_batch['x'][start_pos:end_pos, :] #我操，一个完整的句子在这里被截断了，分割为两个句子，所以batch['x_reset'] = numpy.zeros(self.state['bs'],dtype = 'float32')才是一个完整的句子，才是一个完整的batch的结束
                    batch['x_reversed'] = full_batch['x_reversed'][start_pos:end_pos, :]
                    batch['x_mask'] = full_batch['x_mask'][start_pos:end_pos, :]
                    batch['max_length'] = end_pos - start_pos
                    batch['num_preds'] = numpy.sum(batch['x_mask']) - numpy.sum(batch['x_mask'][0,:])

                    # For each batch we compute the number of dialogues as a fraction of the full batch,
                    # that way, when we add them together, we get the total number of dialogues.
                    batch['num_dialogues'] = float(full_batch['num_dialogues']) / float(splits)
                    batch['x_reset'] = numpy.ones(self.state['bs'], dtype='float32')#这里是numpy.ones(),下面将最后一个batch的x_reset改成了zeros()

                    batches.append(batch)

                if len(batches) > 0:
                    batches[len(batches)-1]['x_reset'] = numpy.zeros(self.state['bs'], dtype='float32')#这里的把最后一个batch的x_reset置为zeros，但是之前的都是ones

                for batch in batches:
                    if batch:
                        yield batch


    def start(self):
        SSIterator.start(self)
        self.batch_iter = None

    def next(self, batch_size = -1):
        """ 
        We can specify a batch size,
        independent of the object initialization. 
        """
        # If there are no more batches in list, try to generate new batches
        if not self.batch_iter:
            self.batch_iter = self.get_homogenous_batch_iter(batch_size)

        try:
            # Retrieve next batch
            batch = next(self.batch_iter)

            # Add Gaussian random variables to batch. 
            # We add them separetly for each batch to save memory.
            # If we instead had added them to the full batch before splitting into mini-batches,
            # the random variables would take up several GBs for big batches and long documents.
            batch = add_random_variables_to_batch(self.state, self.rng, batch, self.prev_batch, self.evaluate_mode)
            # Keep track of last batch
            self.prev_batch = batch
        except StopIteration:
            return None
        return batch

def get_train_iterator(state):
    train_data = Iterator(
        state['train_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=True,
        max_len=-1,
        evaluate_mode=False)
     
    valid_data = Iterator(
        state['valid_dialogues'],
        int(state['bs']),
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=-1,
        evaluate_mode=True)
    return train_data, valid_data 

def get_test_iterator(state):
    assert 'test_dialogues' in state
    test_path = state.get('test_dialogues')

    test_data = Iterator(
        test_path,
        int(state['bs']), 
        state=state,
        seed=state['seed'],
        use_infinite_loop=False,
        max_len=-1,
        evaluate_mode=True)
    return test_data




