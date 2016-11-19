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

class TokenProcessor(object):
    def __init__(self, random_seed, unkreplace_cnt_threshold=1, unkreplace_prob_threshold=0.2):
        self._rng = random.Random(random_seed)
        self._unkreplace_cnt = unkreplace_cnt_threshold
        self._unkreplace_prob = unkreplace_prob_threshold
        
        self._raw_training_data = read_raw_training_data()
        self._raw_devel_data = read_raw_devel_data()
        self._raw_test_data = read_raw_test_data()
        self._training_data = self._devel_data = self._test_data = None
        
        self._word2idx = dict()
        self._idx2word = list()
        self._tag2idx = dict()
        self._idx2tag = list()
        self._wordcnt = list()
        self._unk_str, self._sos_str, self._eos_str = '<unk>', 'sos_str', 'eos_str'
        self._unk_idx = self._sos_idx = self._eos_idx = -1
        self._worddict_sz = self._tagdict_sz = 0

        logging.getLogger(__name__).setLevel(loglevel)

    def _build_worddict(self):
        logging.getLogger(__name__).info("build word dict... ")
        # training dataset format: 
        # [ ( [W1, W2, ... ], [T1, T2, ... ]  ), ... ] 
        word_counter = collections.Counter()
        for instance in self._raw_training_data:
            word_list = instance[0]
            word_counter.update(word_list)
        # now word counter has <word, count> pair list.
        word_count_list = word_counter.most_common() # if we use Coutner.items() for iteration, it'll cause unordered in every time.
                                                     # => lead to word has different id in every run. But We Want The Identity.
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


    def _build_tagdict(self):
        logging.getLogger(__name__).info("build tag dict... ")
        tag_set = set()
        for instance in self._raw_training_data:
            tag_list = instance[1]
            tag_set.update(tag_list)
        ordered_tag_list = sorted(list(tag_set)) # using ordered for identity.
        for tag in ordered_tag_list:
            self._tag2idx[tag] = len(self.idx2tag)
            self._idx2tag.append(tag)
        self._tagdict_sz = len(self._idx2tag)
        print("+ tag dict info: dict size({})".format(self._tagdict_sz))
    
    def convert_word2idx(self, word):
        return self.word2idx[word] if word in self.word2idx else self.unk_idx

    def convert_tag2idx(self, tag):
        return self.tag2idx[tag] # if tag not in tag2idx, just die.

    def convert_idx2word(self, wordidx):
        return self.idx2word[wordidx] if 0 <= wordidx < self.worddict_sz else "<invlaid>"

    def convert_idx2tag(self, tagidx):
        return self.idx2tag[tagidx] if 0 <= tagidx < self.tagdict_sz else "<invalid>"
    
    def _translate_annotated_data(self, dataset):
        ''' raw annotated dataset => index annotated dataset
        '''
        idx_dataset = []
        for word_list, tag_list in dataset:
            word_idx_list = [ self.convert_word2idx(word) for word in word_list ]
            tag_idx_list = [ self.convert_tag2idx(tag) for tag in tag_list ]
            idx_dataset.append( (word_idx_list, tag_idx_list) )
        return idx_dataset
    
    def _translate_unannotated_data(self, dataset):
        ''' raw unannotated dataset => inedx unannotated datasest
        '''
        idx_dataset = []
        for word_list in dataset:
            word_idx_list = [ self.convert_word2idx(word) for word in word_list ]
            idx_dataset.append(word_idx_list)
        return idx_dataset

    def _build_training_data(self):
        logging.getLogger(__name__).info("build training data... ")
        self._training_data = self._translate_annotated_data(self._raw_training_data)
        logging.getLogger(__name__).info("done.")

    def _build_devel_data(self):
        logging.getLogger(__name__).info("build devel data... ")
        self._devel_data = self._translate_annotated_data(self._raw_devel_data)
        logging.getLogger(__name__).info("done.")

    def _build_test_data(self):
        logging.getLogger(__name__).info("build test data... ")
        self._test_data = self._translate_unannotated_data(self._raw_test_data)
        logging.getLogger(__name__).info("done.")
    
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
    
    def init_data(self):
        self._build_training_data()
        self._build_devel_data()
        self._build_test_data()
    
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
