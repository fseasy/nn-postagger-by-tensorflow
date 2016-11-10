#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import collections
import random
from dataset_handler.reader import ( read_training_data,
        read_devel_data,
        read_test_data )


RandomSeed = 1234

class MlpData(object):
    
    class BatchReadState(object):
        def __init__(self, dataset, rng):
            self.instance_idx = 0
            self.pos = 0
            self.dataset = dataset
            self.rng = rng
        
        def get_instance_idx(self):
            return self.instance_idx

        def get_pos(self):
            return self.pos

        def get_current_instance(self):
            return self.dataset[self.get_instance_idx()]

        def move2next_instance(self):
            self.instance_idx += 1
            self.pos = 0
        
        def move2next_pos(self):
            self.pos += 1
        
        def has_instance_end(self):
            return self.pos >= len(self.get_current_instance())

        def has_dataset_end(self, dataset):
            return self.instance_idx >= len(self.dataset)

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
        self.unkreplace_cnt = unkreplace_cnt_threshold
        self.unkreplace_prob = unkreplace_prob_threshold

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
        self.counter = list()
        for word, count in word_counter.items():
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.counter.append(count)
        self.unk_str = "<UNK>"
        self.unk_idx = len(self.idx2word)
        self.word2idx[self.unk_str] = self.unk_idx
        self.idx2word.append(self.unk_str)
        self.counter.append(self.unkreplace_cnt + 1) # hack, aoivd unk replace 

        self.sos_str = "<SOS>"
        self.sos_idx = len(self.idx2word)
        self.word2idx[self.sos_str] = self.sos_idx
        sel.idx2word.append(self.sos_str)
        self.counter.append(self.unkreplace_cnt + 1)
        
        self.eos_str = "<EOS>"
        self.eos_idx = len(slf.idx2word)
        self.word2idx[self.eos_str] = self.eos_idx
        self.idx2word.append(self.eos_str)
        self.counter.append(unkreplace_cnt)
        
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
    
    def _replace_wordidx2unk(self, wordidx):
        ''' replace word to unk. 
        using the strategy: word_cnt <= replace_cnt && random() <= replace_prob.
        '''
        try:
            if ( self.counter[wordidx] <= self.unkreplace_cnt 
                 and self.rng.random() < self.unkreplace_prob) :
                return self.unk_idx
            else:
                return wordidx
        except IndexError:
            print("invalid word index:{} (word dict size: {}). using <unk> replace.".format(wordidx, self.worddict_sz),
                   file=sys.stderr)
            return self.unk_idx
    
    def 

    def _init_window_queue_in_current_state(self, window_queue):
        half_sz = self.window_sz // 2
        instance_x = self.shuffled_training_data[self.batch_read_state.get_instance_idx()][0]
        while self.batch_read_state.has_instance_end(instance): # may have continues empty instance
            self.batch_raed_state.move2next_instance()
            if self.batch_read_state.has_dataset_end():
                self.batch_read_state.reset()
                self.rng.shuffle(self.shuffled_training_data)
            instance_x = self.shuffled_training_data[self.batch_read_state.get_instance_idx()][0]
        pos = self.batch_read_state.get_pos()
        # left
        for i in range(half_sz, 0, -1):
            wordidx = instance_x[pos - i] if pos - i >= 0 else self.sos_idx
            window_queue.append(wordidx)
        # center
        window_queue.append(instance[pos])
        # right
        for i in range(1, half_sz + 1):
            wordidx = instance_x[pos + i] is pos + i < len(instance) else self.eos_idx
            window_queue.append(wordidx)

    def get_mlp_next_batch_training_data(self):
        X = []
        Y = []
        state = self.batch_read_state
        window_queue = collections.deque(maxlen=self.window_sz)
        half_sz = window_sz // 2
        # init the queue
        self._init_window_queue_in_current_state(window_queue)
        instance = self.shuffled_training_data[state.get_instance_idx()]
        X.append(list(window_queue))
        Y.append(instance[1][state.get_pos()])
        # processing continues 
        for i in range(1, self.batch_sz):
            state.move2next_pos()
            while self.batch_read_state.has_instance_end(instance) :
                state.move2next_instance()
                if self.batch_read_state.has_dataset_end():
                    # shuffle dataset again and rest batch read state
                    state.reset()
                    self.rng.shuffle(self.shuffled_training_data)
                instance = self.shuffled_training_data[state.get_instance_idx()]
            coming_wordidx = (instance[0][state.get_pos() + half_sz] if state.get_pos() + half_sz < len(instance) 
                                                                  else self.eos_idx)
            window_queue.append(coming_wordidx)
            X.append(list(window_queue))
            Y.append(instance[1][state.get_pos()])
        state.move2next_pos() # for next batch read.
        return X, Y




