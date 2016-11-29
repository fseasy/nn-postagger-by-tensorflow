#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division


import sys
import os
import collections
import random
import logging

package_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(package_path)

from dataset_handler.token_processor import TokenProcessor

logging.basicConfig(level=logging.DEBUG)

class MlpData(TokenProcessor):
    
    def __init__(self, rng_seed, window_sz=5, 
                 unkreplace_cnt_threshold=1, unkreplace_prob_threshold=0.2, loglevel=logging.INFO):
        TokenProcessor.__init__(rng_seed, unkreplace_cnt_threshold, unkreplace_prob_threshold) 
        
        self._window_sz = window_sz
        self._rng = random.Random(rng_seed)
        
        logging.getLogger(__name__).setLevel(loglevel)

    def _generate_window_token_list(self, x):
        sent_len = len(x)
        if sent_len == 0:
            return []
        window_token_list = []
        window_queue  = collections.deque()
        half_sz = self._window_sz // 2
        # init the first window
        window_queue.extend( [self._sos_idx] * half_sz )
        window_queue.append(x[0])
        for i in range(1, half_sz + 1):
            window_queue.append( x[i] if i < sent_len else self._eos_idx )
        window_token_list.append(list(window_queue))
        # processing continues
        for center in range(1, sent_len):
            right_most = center + half_sz
            window_queue.append(x[right_most] if right_most < sent_len else self._eos_idx)
            window_token_list.append(list(window_queue))
        return window_token_list

    def batch_data_generator(self, data, batch_sz=128, fill_when_not_enouth=True, has_tag=True):
        '''
        A generator to produce the Mlp window batch data.
        @data original data, probably is the result of `build_training_data`/ `build_devel_data`, `build_test_data`
              has the form [ (X, Y), ... ] if has_tag = True
              or [ X, ...] is has_tag = False
        @batch_sz batch size
        @fill_when_not_enouth if left data can't fill a full batch, how should we do:
                              fill a full batch use duplicated instance if set True, suite for training data.
                              just return un-full batch data if False, suite for devel and test data.
        @has_tag whethere the data has tag. see @data
        @return (batch_x, batch_y) if has_tag == True
                batch_x if has_tag == False
        '''
        data_size = len(data)
        if data_size == 0:
            if has_tag:
                yield ([], [])
            else:
                yield []
            return
        # 1. generate an random access order
        access_order = [i for i in range(0, data_size)] # for compability of Py2 and Py3
        self._rng.shuffle(access_order)
        # 2. generate one window X
        get_x = lambda i: data[access_order[i]]




def test():
    mlp_data = MlpData(window_sz=3, batch_sz=10)
    for i in range(5):
        X, Y = mlp_data.get_mlp_next_batch_training_data()
        for x, y in zip(X, Y):
            print(x, y)
            for wi in x:
                print(mlp_data.convert_idx2word(wi), end=",")
            print("\t", end="")
            print(mlp_data.convert_idx2tag(y))
    print()
    
    devel_data = mlp_data.get_mlp_devel_data()
    print("devel data has instance cnt: {}".format(len(devel_data[0])))
    test_data = mlp_data.get_mlp_test_data()
    print("test data has instance cnt: " , len(test_data))
    print("shuffled_time: ", mlp_data.iterate_time())

if __name__ == "__main__":
    test()
