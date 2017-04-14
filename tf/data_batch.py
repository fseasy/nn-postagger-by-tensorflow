#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import random
import copy

class MaxLenStrategy(object):
    '''
    batch data time step padding length strategy.
    Options:
        BatchAdaption: according to the current batch max len
        GlobalMaxLen: use the global max len
        Bucket: use the bucket strategy.
    '''
    BatchAdaption = 0
    GlobalMaxLen = 1
    Bucket = 2

    @classmethod
    def get_batch_len(cls,
            strategy,
            cur_batch_max_len=-1,
            global_len=-1,
            bucket=None):
        '''
        get batch len according to the strategy
        '''
        if strategy == cls.BatchAdaption:
            return cur_batch_max_len
        elif strategy == cls.GlobalMaxLen:
            return global_len
        elif strategy == cls.Bucket:
            # TODO
            return 0
        else:
            raise ValueError("unknown strategy {}".format(strategy))




def batch_training_data_generator(
        data,
        batch_size,
        x_padding_id=0,
        y_padding_id=0, 
        max_len_strategy=MaxLenStrategy.GlobalMaxLen,
        padding_len_bucket=None,
        rng=random.Random(1234),
        use_unk_replace=False,
        word_cnt_dict=None,
        unk_id=-1,
        replace_cnt_lower_bound=0,
        replace_prob_lower_bound=0.):
    '''
    batch training data generator.
    using *padding_id to pad the input(x and y) to be a 2-d matrix(
    raw python list of list).
    if unk_replace is set to True, do unk-replace stratege.
    rng to assign the random engin, which shuffle the order,
    and affect the replace result if do unk-replace.

    Args:
        data: for Train, Dev, a list of Tuple, [(x1, y1), (x2, y2)...],
            where x, y a list of index; for Test, [x1, x2, ...]
        batch_size: batch size
        x_padding_id: padding id for word to pad
        y_padding_id: padding id for tag to pad 
        max_len_strategy: max length strategy. See @MaxLenStrategy
        padding_len_bucket: used only when max_len_strategy == 
                            MaxLenStrategy.Bucket
        rng: random number generator, if None, no shuffle and unk-replace
        use_unk_replace: whether use unk-replace stratege
        replace_cnt_lower_bound: count lower bound when enable replace
        replace_prob_lower_bound: probability lower bound when enable replace
    Returns:
        A tupple, (batch_X, batch_Y, sentenc_len)
        - batch_X: python raw list of list, the inner lists all has the same 
            length because of padding
        - batch_Y: list of list, the sample length for padding
        - sentence_len: list of int. the true length for every sentence in batch
    '''
    data_len = len(data)
    access_order = list(range(data_len))
    rng.shuffle(access_order) # shufle
    batch_X = []
    batch_Y = []
    sample_idx = 0
    batch_sample_cnt = 0
    # max len in current batch
    max_len_in_batch = 0
    # max len for global data
    global_max_len = 0
    if max_len_strategy == MaxLenStrategy.GlobalMaxLen:
        for x, y in data:
            global_max_len = max(global_max_len, len(x))
    def _get_batch_len():
        # can closure use the reference of variable
        # instead of the current variable value?
        # YES
        return MaxLenStrategy.get_batch_len(
                max_len_strategy,
                cur_batch_max_len=max_len_in_batch,
                global_len=global_max_len,
                bucket=padding_len_bucket)
    while sample_idx < data_len:
        # ready a batch
        while (batch_sample_cnt < batch_size and 
               sample_idx < data_len) :
            idx = access_order[sample_idx]
            cur_x, cur_y = data[idx]
            max_len_in_batch = max(max_len_in_batch, len(cur_x))
            batch_X.append(copy.copy(cur_x)) # copy avoid modifying the original data
            batch_Y.append(copy.copy(cur_y))
            batch_sample_cnt += 1
            sample_idx += 1
        # unk-replace
        if use_unk_replace:
            def do_replace_word_id(word_id):
                cnt = word_cnt_dict[word_id]
                random_prob = rng.uniform(0, 1)
                if (cnt < replace_cnt_lower_bound and 
                    random_prob < replace_cnt_lower_bound):
                    return unk_id
                else:
                    return word_id
            iter_num = 0
            while iter_num < len(batch_X):
                x = batch_X[iter_num]
                batch_X[iter_num] = [ do_replace_word_id(word_id) for 
                                      word_id in x]
                iter_num += 1

        # add padding
        cur_batch_len = _get_batch_len()
        sentence_len = _naive_padding(batch_X, x_padding_id,
                                      cur_batch_len)
        _naive_padding(batch_Y, y_padding_id, 
                cur_batch_len)
        yield (batch_X, batch_Y, sentence_len)
        # clear for next
        batch_X = []
        batch_Y = []
        batch_sample_cnt = 0
        max_len_in_batch = 0
    else:
        # outer while end => sample_idx >= data_len
        # all data must have been done!
        assert(len(batch_X) == 0)


def batch_develop_data_generator(
        data,
        batch_size,
        x_padding_id=0,
        y_padding_id=0,
        max_len_strategy=MaxLenStrategy.GlobalMaxLen,
        padding_len_bucket=None):
    '''
    batch develop data generator.
    Args:
        data: [(x1, y1), (x2, y2), ...]
        batch_size: batch size
        x_padding_id: padding id for x
        y_padding_id: y padding id
    Returns:
        A tuple, (batch_X, batch_Y, sentence_len)
    '''
    data_len = len(data)
    max_len_in_batch = 0
    max_len_in_global = 0
    if max_len_strategy == MaxLenStrategy.GlobalMaxLen:
        for x, y in data:
            max_len_in_global = max(len(x),
                    max_len_in_global)
    def _get_batch_len():
        # can closure use the reference of variable
        # instead of the current variable value?
        # YES
        return MaxLenStrategy.get_batch_len(
                max_len_strategy,
                cur_batch_max_len=max_len_in_batch,
                global_len=max_len_in_global,
                bucket=padding_len_bucket)
    batch_X = []
    batch_Y = [] # you can't write X = Y = [] (they refrence same memory)
    for x, y in data:
        batch_X.append(copy.copy(x)) # batch size > 0
        batch_Y.append(copy.copy(y))
        max_len_in_batch = max(max_len_in_batch, len(x))
        if len(batch_X) >= batch_size:
            # padding
            batch_len = _get_batch_len()
            sent_len = _naive_padding(batch_X, x_padding_id, 
                                      batch_len)
            _naive_padding(batch_Y, y_padding_id, batch_len)
            yield (batch_X, batch_Y, sent_len)
            batch_X = []
            batch_Y = []
            max_len_in_batch = 0
    # remains
    if len(batch_X) > 0:
        batch_len = _get_batch_len()
        sent_len = _naive_padding(batch_X, x_padding_id,
                                  batch_len)
        _naive_padding(batch_Y, y_padding_id, batch_len)
        yield (batch_X, batch_Y, sent_len)

def batch_test_data_generator(
        data,
        batch_size, 
        x_padding_id=0,
        max_len_strategy=MaxLenStrategy.GlobalMaxLen,
        padding_len_bucket=None):
    '''
    batch test data generator.
    Args:
        data: [x1, x2, ..]
        batch_size: batch size
        x_padding_id: padding id for x
    Returns:
        A tuple, (batch_X, sentence_length_list)
    '''
    batch_sample_idx = 0
    max_len_in_batch = 0
    max_len_in_global = 0
    def _get_batch_len():
        # can closure use the reference of variable
        # instead of the current variable value?
        # YES
        return MaxLenStrategy.get_batch_len(
                max_len_strategy,
                cur_batch_max_len=max_len_in_batch,
                global_len=max_len_in_global,
                bucket=padding_len_bucket)
    if max_len_strategy == MaxLenStrategy.GlobalMaxLen:
        for x in data:
            max_len_in_global = max(len(x),
                    max_len_in_global)
    batch_X = []
    for instance in data:
        batch_X.append(copy.copy(instance))
        max_len_in_batch = max(max_len_in_batch, len(instance))
        if len(batch_X) >= batch_size:
            batch_len = _get_batch_len()
            sent_len = _naive_padding(batch_X, x_padding_id, batch_len)
            yield(batch_X, sent_len)
            batch_X = []
            max_len_in_batch = 0
    if len(batch_X) > 0:
        batch_len = _get_batch_len()
        sent_len = _naive_padding(batch_X, x_padding_id, batch_len)
        yield (batch_X, sent_len)

####################
# inner function
####################

def _naive_padding(batch_data, padding_id, batch_max_len):
    ''''
    do padding in-place and return sentence length.
    Args:
        batch_data: list of list, batch data, [x1, x2]
        padding_id: int, padding id
        batch_max_len: int, max len
    Returns:
        sentence length. length for every sent of batch
    '''
    sent_len = []
    for one_instance in batch_data:
        instance_len = len(one_instance)
        sent_len.append(instance_len)
        padding_len = batch_max_len - instance_len
        one_instance.extend([padding_id] * padding_len)
    return sent_len
