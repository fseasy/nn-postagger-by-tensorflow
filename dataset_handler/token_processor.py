#/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import random
import logging
import collections

_cur_dir = os.path.dirname(__file__)
sys.path.append(_cur_dir)

from reader import ( read_training_data as read_raw_training_data,
                     read_devel_data as read_raw_devel_data,
                     read_test_data as read_raw_test_data)


class DatasetInfo(object):
    '''
    to record the dataset info
    '''
    class TrainingData(object):
        '''
        training data info
        '''
        def __init__(self):
            self.num_all_token = 0
            self.num_uniq_token = 0
            self.num_sent = 0
            self.num_uniq_tag = 0
        def __str__(self):
            d = "training data: all token num({}), uniq token num({}), uniq tag num({}), sent num({})".format(
                    self.num_all_token, self.num_uniq_token, self.num_uniq_tag, self.num_sent
                    )
            return d
        
    class DevelData(object):
        '''
        devel data info
        '''
        def __init__(self):
            self.num_all_token = 0
            self.num_uniq_token = 0
            self.num_uniq_tag = 0
            self.num_sent = 0
        def __str__(self):
            d = "devel data: all token num({}), uniq token num({}), uniq tag num({}), sent num({})".format(
                    self.num_all_token, self.num_uniq_token, self.num_uniq_tag, self.num_sent
                    )
            return d
    
    class TestData(object):
        '''
        test data info
        '''
        def __init__(self):
            self.num_all_token = 0
            self.num_uniq_token = 0
            self.num_sent = 0
        def __str__(self):
            d = "test data: all token num({}), uniq token num({}), sent num({})".format(
                    self.num_all_token, self.num_uniq_token, self.num_sent
                    )
            return d

    def __init__(self):
        self.training_data_info = self.TrainingData()
        self.devel_data_info = self.DevelData()
        self.test_data_info = self.TestData()



class TokenProcessor(object):
    def __init__(self, random_seed, unkreplace_cnt_threshold=1, unkreplace_prob_threshold=0.2,
                 loglevel=logging.DEBUG):
        self._rng = random.Random(random_seed)
        self._unkreplace_cnt = unkreplace_cnt_threshold
        self._unkreplace_prob = unkreplace_prob_threshold
        
        self._raw_training_data = read_raw_training_data()
        self._raw_devel_data = read_raw_devel_data()
        self._raw_test_data = read_raw_test_data()
        
        self._dataset_info = DatasetInfo()
        
        self._word2idx = dict()
        self._idx2word = list()
        self._tag2idx = dict()
        self._idx2tag = list()
        self._wordcnt = list()
        self._unk_str, self._sos_str, self._eos_str = '<unk>', 'sos_str', 'eos_str'
        self._unk_idx = self._sos_idx = self._eos_idx = -1
        self._worddict_sz = self._tagdict_sz = 0
        self._build_worddict()
        self._build_tagdict()

        self._training_data = self._devel_data = self._test_data = None

        logging.getLogger(__name__).setLevel(loglevel)

    def _build_worddict(self):
        logging.getLogger(__name__).info("build word dict... ")
        # training dataset format: 
        # [ ( [W1, W2, ... ], [T1, T2, ... ]  ), ... ] 
        word_counter = collections.Counter()
        for instance in self._raw_training_data:
            word_list = instance[0]
            word_counter.update(word_list)
        # sorted by ( frequent decreasing && unicode point increasing ).
        word_count_list = sorted(word_counter.items(), key=lambda item: (-item[1], item[0]) )
        for word, count in word_count_list:
            self._word2idx[word] = len(self._idx2word)
            self._idx2word.append(word)
            self._wordcnt.append(count)
        self._unk_idx = len(self._idx2word)
        self._word2idx[self._unk_str] = self._unk_idx
        self._idx2word.append(self._unk_str)
        self._wordcnt.append(self._unkreplace_cnt + 1) # hack, aoivd unk replace 

        self._sos_idx = len(self._idx2word)
        self._word2idx[self._sos_str] = self._sos_idx
        self._idx2word.append(self._sos_str)
        self._wordcnt.append(self._unkreplace_cnt + 1)
        
        self._eos_idx = len(self._idx2word)
        self._word2idx[self._eos_str] = self._eos_idx
        self._idx2word.append(self._eos_str)
        self._wordcnt.append(self._unkreplace_cnt + 1)
        
        self._worddict_sz = len(self._idx2word)
        print("+ word dict info: dict size({}) sos_idx({}) eos_idx({}) unk_idx({})".format(
              self._worddict_sz, self._sos_idx, self._eos_idx, self._unk_idx))
        self._dataset_info.training_data_info.num_uniq_token = self._worddict_sz

    def _build_tagdict(self):
        logging.getLogger(__name__).info("build tag dict... ")
        tag_set = set()
        for instance in self._raw_training_data:
            tag_list = instance[1]
            tag_set.update(tag_list)
        ordered_tag_list = sorted(list(tag_set))
        for tag in ordered_tag_list:
            self._tag2idx[tag] = len(self._idx2tag)
            self._idx2tag.append(tag)
        self._tagdict_sz = len(self._idx2tag)
        print("+ tag dict info: dict size({})".format(self._tagdict_sz))
        self._dataset_info.training_data_info.num_uniq_tag = self._tagdict_sz
    
    def convert_word2idx(self, word):
        return self._word2idx[word] if word in self._word2idx else self._unk_idx

    def convert_tag2idx(self, tag):
        return self._tag2idx[tag] # if tag not in tag2idx, just die.

    def convert_idx2word(self, wordidx):
        return self._idx2word[wordidx] if 0 <= wordidx < self._worddict_sz else "<invlaid>"

    def convert_idx2tag(self, tagidx):
        return self._idx2tag[tagidx] if 0 <= tagidx < self._tagdict_sz else "<invalid>"
    
    def _translate_annotated_data(self, dataset):
        ''' 
        raw annotated dataset => index annotated dataset
        index annotated dataset has the format like:
        [ (X1, Y1), ... ], where X1, Y1 is the sequence of one instance of x and y
        @return ( idx_dataset, num_all_token ), where idx_dataset = [ (X1, Y1), ... ]
        '''
        idx_dataset = []
        num_all_token = 0
        for word_list, tag_list in dataset:
            word_idx_list = [ self.convert_word2idx(word) for word in word_list ]
            tag_idx_list = [ self.convert_tag2idx(tag) for tag in tag_list ]
            idx_dataset.append( (word_idx_list, tag_idx_list) )
            num_all_token += len(word_idx_list)
        return (idx_dataset, num_all_token)
    
    def _translate_unannotated_data(self, dataset):
        '''
        raw unannotated dataset => inedx unannotated datasest
        index unannotated dataset has the format like:
        [X1, ...] where X1 is a sequence
        @return (idx_dataset, num_all_token), where idx_dataset = [X1, ...]
        '''
        idx_dataset = []
        num_all_token = 0
        for word_list in dataset:
            word_idx_list = [ self.convert_word2idx(word) for word in word_list ]
            idx_dataset.append(word_idx_list)
            num_all_token += len(word_idx_list)
        return (idx_dataset, num_all_token)

    def build_training_data(self):
        if self._training_data:
            return self._training_data
        logging.getLogger(__name__).info("build training data... ")
        self._training_data, num_all_token = self._translate_annotated_data(self._raw_training_data)
        logging.getLogger(__name__).info("done.")
        info = self._dataset_info.training_data_info
        info.num_sent = len(self._training_data)
        info.num_all_token = num_all_token
        return self._training_data
    

    def build_devel_data(self):
        if self._devel_data:
            return self._devel_data
        logging.getLogger(__name__).info("build devel data... ")
        self._devel_data, num_all_token = self._translate_annotated_data(self._raw_devel_data)
        logging.getLogger(__name__).info("done.")
        return self._devel_data

    def build_test_data(self):
        if self._test_data:
            return self._test_data
        logging.getLogger(__name__).info("build test data... ")
        self._test_data, num_all_token = self._translate_unannotated_data(self._raw_test_data)
        logging.getLogger(__name__).info("done.")
        return self._test_data
    
    def get_training_data_info(self):
        return self._dataset_info.training_data_info
    
    def replace_wordidx2unk(self, wordidx):
        ''' replace word to unk. 
        using the strategy: word_cnt <= replace_cnt && random() <= replace_prob.
        '''
        try:
            if ( self._wordcnt[wordidx] <= self._unkreplace_cnt 
                 and self._rng.random() < self._unkreplace_prob) :
                return self.unk_idx
            else:
                return wordidx
        except IndexError:
            print("invalid word index:{} (word dict size: {}). using <unk> replace.".format(wordidx, self._worddict_sz),file=sys.stderr)
            return self._unk_idx
    
    @property
    def worddict_sz(self):
        return self._worddict_sz
    @property
    def tagdict_sz(self):
        return self._tagdict_sz
    @property
    def unk_idx(self):
        return self._unk_idx
    @property
    def sos_idx(self):
        return self._sos_idx
    @property
    def eos_idx(self):
        return self._eos_idx

def _unit_test():
    tp = TokenProcessor(1234)
    training_data = tp.build_training_data()
    print(tp.get_training_data_info())
    print("1st line sample:")
    print("index token seq:", end=" " )
    print("{}".format( " ".join( map(str,training_data[0][0]) ) ))
    print("str token seq:", end=" ")
    print("{}".format(" ".join([ tp.convert_idx2word(idx) for idx in training_data[0][0] ])))
    print("index tag seq:", end=" ")
    print("{}".format(" ".join( map(str, training_data[0][1])  )))
    print("str tag seq:", end=" ")
    print("{}".format(" ".join([ tp.convert_idx2tag(idx) for idx in training_data[0][1]]) ))

if __name__ == "__main__":
    print("do token processor unit test.", file=sys.stderr)
    logging.basicConfig(level=logging.DEBUG)
    _unit_test()
