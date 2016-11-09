#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import collections
import random
from dataset_handler.reader import ( read_training_data,
        read_devel_data,
        read_test_data )


RandomSeed = 1234

class MlpData(object):
    
    class BatchReadState(object):
        def __init__(self):
            self.instance_idx = 0
            self.pos = 0
        
        def move2next_instance(self):
            self.instance_idx += 1
            self.pos = 0
        
        def move2next_pos(self):
            self.pos += 1
        
        def has_instance_end(self, instance):
            return self.pos >= len(instance)

        def has_dataset_end(self, dataset):
            return self.instance_idx >= len(dataset)

        def reset(self):
            self.instance_idx = 0
            self.pos = 0
    
    def __init__(self, window_sz=5, batch_sz=128, 
                 unkreplace_cnt_threshold=1, unkreplace_prob_threshold=0.2):
        self.window_sz = window_sz
        self.batch_sz = batch_sz
        self.raw_training_data = read_training_data()
        self.raw_devel_data = read_devel_data()
        self.raw_test_data = read_test_data()
        self.batch_read_state = BatchReadState()
        self.rng = random.Random(RandomSeed)

    def build_worddict(self):
        # training dataset format: 
        # [ ( [W1, W2, ... ], [T1, T2, ... ]  ), ... ] 
        word_counter = collections.Counter()
        for instance in self.raw_training_data:
            word_list = instance[0]
            word_counter.update(word_list)
        # now word counter has <word, count> pair list.
        self.word2idx = dict()
        self.idx2word = list()
        self.counter = word_counter
        for word, count in word_counter.items():
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        self.unk_str = "<UNK>"
        self.unk_idx = len(self.idx2word)
        self.word2idx[self.unk_str] = self.unk_idx
        self.idx2word.append(self.unk_str)

        self.sos_str = "<SOS>"
        self.sos_idx = len(self.idx2word)
        self.word2idx[self.sos_str] = self.sos_idx
        sel.fidx2word.append(self.sos_str)
        
        self.eos_str = "<EOS>"
        self.eos_idx = len(slf.idx2word)
        self.word2idx[self.eos_str] = self.eos_idx
        self.idx2word.append(self.eos_str)
        
        self.worddict_sz = len(self.idx2word)

    def build_tagdict(self):
        tag_set = set()
        for instance in self.raw_training_data:
            tag_list = instance[1]
            tag_set.update(tag_list)
        self.tag2idx = dict()
        self.idx2tag = list()
        for tag in tag_set:
            self.tag2idx[tag] = len(self.idx2tag)
            self.idx2tag.append(tag)
        self.tagdict_sz = len(self.idx2tag)
    
    def _translate_annotated_data(self, dataset):
        ''' raw annotated dataset => index annotated dataset
        '''
        idx_dataset = []
        for word_list, tag_list in dataset:
            word_idx_list = [ self.word2idx[word] for word in word_list ]
            tag_idx_list = [ self.tag2idx[tag] for tag in tag_list ]
            idx_dataset.append( (word_idx_list, tag_idx_list) )
        return idx_dataset
    
    def _translate_unannotated_data(self, dataset):
        ''' raw unannotated dataset => inedx unannotated datasest
        '''
        idx_dataset = []
        for word_list in dataset:
            word_idx_list = [ self.word2idx[word] for word in word_list ]
            idx_dataset.append(word_idx_list)
        return idx_dataset

    def build_training_data(self):
        self.shuffled_training_data = self._translate_annotated_data(self.raw_training_data)

    def build_devel_data(self):
        self.devel_data = self._translate_annotated_data(self.raw_devel_data)

    def build_test_data(self):
        self.test_data = self._translate_unannotated_data(self.raw_test_data)
    
    def _replace_wordidx2unk(self):
        ''' replace word to unk. 
        using the strategy: word_cnt <= replace_cnt && random() <= replace_prob.
        '''
        # TODO

    def get_next_batch_training_data(self):
        X = []
        Y = []
        for i in range(self.batch_sz):
            if self.batch_read_state.has_instance_end() :
                self.batch_read_state.move2next_instance()
            if self.batch_read_state.has_dataset_end():
                # shuffle dataset again and rest batch read state
                self.batch_read_state.reset()
                self.rng.shuffle(self.shuffled_training_data)



