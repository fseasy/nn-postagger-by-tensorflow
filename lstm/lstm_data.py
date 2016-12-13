#/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys

package_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(package_path)

from dataset_handler.token_processor import TokenProcessor

class LstmData(TokenProcessor):
    '''Lstm Data module '''
    def __init__(self, random_seed, unkreplace_cnt=1, unkreplace_prob=0.2):
        super(LstmData, self).__init__(random_seed, unkreplace_cnt, unkreplace_prob)
        # Adding padding index to word dict
        self._padding_str = "<padding>"
        self._padding_idx = len(self._idx2word)
        self._word2idx[self._padding_str] = self._padding_idx
        self._idx2word.append(self._padding_str)
        self._wordcnt.append(self._unkreplace_cnt + 1)

    
    def batch_data_generator(self, data, batch_sz, 
                             padding_strategy="min", fixed_length=0,
                             fill_when_not_enouth=True, has_tag=True, do_unk_replace=False)
        '''
        batch data generator.
        @data, formating of [ (X, Y), ... ] if has_tag, else [ X, ...]
        @batch_sz, batch size 
        @padding_strategy padding strategy, should be one of the candidate {"none", "min", "fixed"}
                          means specified:
                          "none"  : no padding,
                          "min"   : select the max length of the current batch as the uniform length
                          "fixed" : use the fixed length(specified) as the unifrom length of the batch
        @fixed_length fixed length( > 0) for batch, make sense when padding_strategy="fixed"
        @fill_when_not_enough if True, repeat the samples when no enough samples to fill full the batch
                              else leave it un-full if no enough samples
        @has_tag indicating whether the data has the tag.
        @do_unk_replace whether do unk replace strategy
        '''
        pass
    

