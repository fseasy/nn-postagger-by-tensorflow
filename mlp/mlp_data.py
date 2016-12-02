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
        TokenProcessor.__init__(self, rng_seed, unkreplace_cnt_threshold, unkreplace_prob_threshold) 
        
        self._window_sz = window_sz
        self._rng = random.Random(rng_seed)
        
        logging.getLogger(__name__).setLevel(loglevel)

    def _generate_window_token_list(self, x):
        sent_len = len(x)
        if sent_len == 0:
            return []
        window_token_list = []
        window_queue  = collections.deque(maxlen=self._window_sz)
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
    
    def replace_low_freq_token_with_unk(self, x):
        if type(x) is list:
            y = [ self.replace_wordidx2unk(idx) for idx in x ]
        else:
            y = self.replace_wordidx2unk(idx)
        return y

    def batch_data_generator(self, data, batch_sz=128, fill_when_not_enouth=True, has_tag=True, has_unk_replace=False):
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
                where batch_x = [ W1, ... ] and W1 is the window of word1; 
                batch_y = [y1, ...] and y1 is the tag of word1
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
        _get_x = lambda i: data[access_order[i]][0] if has_tag else data[access_order[i]] # get original x
        get_x = lambda i: self.replace_low_freq_token_with_unk(_get_x(i)) if has_unk_replace else _get_x(i) # get replaced x
        get_y = lambda i: data[access_order[i]][1] if has_tag else None
        batch_x = []
        batch_y = []
        for i in range(data_size):
            x_window_list = self._generate_window_token_list(get_x(i))
            y = get_y(i)
            for pos, x_window in enumerate(x_window_list):
                if len(batch_x) == batch_sz:
                    if has_tag:
                        yield (batch_x, batch_y)
                    else:
                        yield batch_x
                    batch_x = []
                    batch_y = []
                batch_x.append(x_window)
                if has_tag:
                    batch_y.append(y[pos])
        # process the end
        # 1. no left
        if len(batch_x) == 0: 
            return
        # 2. no need fill
        if not fill_when_not_enouth:
            if has_tag:
                yield (batch_x, batch_y)
            else:
                yield batch_x
        # 3. fill
        else:
            # just copy self
            filled_len = len(batch_x)
            left_len = batch_sz - filled_len
            while filled_len < left_len:
                batch_x.extend(batch_x[:filled_len])
                if has_tag:
                    batch_y.extend(batch_y[:filled_len])
                left_len -= filled_len
                filled_len *= 2
            batch_x.extend(batch_x[:left_len])
            if has_tag:
                batch_y.extend(batch_y[:left_len])
            assert( len(batch_x) == batch_sz)
            if has_tag:
                yield (batch_x, batch_y)
            else:
                yield batch_x
    
    @property
    def window_sz(self):
        return self._window_sz



def unit_test():
    mlp_data = MlpData(1234, window_sz=3)
    training_data = mlp_data.build_training_data()
    for batch_x, batch_y in mlp_data.batch_data_generator(training_data[:1], batch_sz=20):
        print(batch_x)
        print(batch_y)
    test_data = mlp_data.build_test_data()
    for batch_x in mlp_data.batch_data_generator(test_data[-1:], batch_sz=10, has_tag=False, fill_when_not_enouth=False):
        print(batch_x)

if __name__ == "__main__":
    unit_test()
