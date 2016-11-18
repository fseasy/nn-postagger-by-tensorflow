#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os
import collections
import random
import logging

package_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
sys.path.append(package_path)

from dataset_handler.reader import ( read_training_data,
        read_devel_data,
        read_test_data )

logging.basicConfig(level=logging.DEBUG)

RandomSeed = 1234

class MlpData(object):
    
    class BatchReadState(object):
        def __init__(self, dataset, rng):
            self.instance_idx = 0
            self.pos = 0
            self.dataset = dataset
            self._rng = rng
            self._nr_shuffled_time = 0
        
        @property
        def nr_shuffled_time(self):
            return self._nr_shuffled_time

        def get_current_pos(self):
            return self.pos
        
        def get_current_instance(self):
            return self.dataset[self.instance_idx] # throw IndexError
        
        def _do_shuffle_transaction(self):
            self._rng.shuffle(self.dataset)
            self.instance_idx = 0
            self._nr_shuffled_time += 1

        def move2next_instance(self):
            self.instance_idx += 1
            self.pos = 0
            # to ensure the current(moved) instance is not empty!
            # 1. check whether next instance exists
            has_shuffled = False
            if self.instance_idx >= len(self.dataset):
                # shuffle again
                self._do_shuffle_transaction()
                has_shuffled = True
            # 2. encure the current instance not empty.
            while len(self.dataset[self.instance_idx][0]) == 0:
                self.instance_idx += 1
                if self.instance_idx >= len(self.dataset):
                    if has_shuffled:
                        raise Exception("dataset empty!")
                    else:
                        self._do_shuffle_transaction()
                        has_shuffled = True
        
        def move2next_pos(self):
            self.pos += 1
        
        def has_instance_end(self):
            return self.pos >= len(self.get_current_instance()[0])

        def reset(self):
            self.instance_idx = 0
            self.pos = 0
    
    def __init__(self, window_sz=5, batch_sz=128, 
                 unkreplace_cnt_threshold=1, unkreplace_prob_threshold=0.2, loglevel=logging.INFO):
        self._window_sz = window_sz
        self._batch_sz = batch_sz
        self._rng = random.Random(RandomSeed)
        self._unkreplace_cnt = unkreplace_cnt_threshold
        self._unkreplace_prob = unkreplace_prob_threshold
        self._build_all_data()
        logging.getLogger(__name__).setLevel(loglevel)

    def _build_worddict(self):
        logging.getLogger(__name__).info("build word dict... ")
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
        self.counter.append(self._unkreplace_cnt + 1) # hack, aoivd unk replace 

        self.sos_str = "<SOS>"
        self.sos_idx = len(self.idx2word)
        self.word2idx[self.sos_str] = self.sos_idx
        self.idx2word.append(self.sos_str)
        self.counter.append(self._unkreplace_cnt + 1)
        
        self.eos_str = "<EOS>"
        self.eos_idx = len(self.idx2word)
        self.word2idx[self.eos_str] = self.eos_idx
        self.idx2word.append(self.eos_str)
        self.counter.append(self._unkreplace_cnt + 1)
        
        self._worddict_sz = len(self.idx2word)
        print("+ word dict info: dict size({}) sos_idx({}) eos_idx({}) unk_idx({})".format(
              self.worddict_sz, self.sos_idx, self.eos_idx, self.unk_idx))


    def _build_tagdict(self):
        logging.getLogger(__name__).info("build tag dict... ")
        tag_set = set()
        for instance in self.raw_training_data:
            tag_list = instance[1]
            tag_set.update(tag_list)
        self.tag2idx = dict()
        self.idx2tag = list()
        for tag in tag_set:
            self.tag2idx[tag] = len(self.idx2tag)
            self.idx2tag.append(tag)
        self._tagdict_sz = len(self.idx2tag)
        print("+ tag dict info: dict size({})".format(self.tagdict_sz))
    
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
        self.shuffled_training_data = self._translate_annotated_data(self.raw_training_data)
        logging.getLogger(__name__).info("done.")

    def _build_devel_data(self):
        logging.getLogger(__name__).info("build devel data... ")
        self.devel_data = self._translate_annotated_data(self.raw_devel_data)
        logging.getLogger(__name__).info("done.")

    def _build_test_data(self):
        logging.getLogger(__name__).info("build test data... ")
        self.test_data = self._translate_unannotated_data(self.raw_test_data)
        logging.getLogger(__name__).info("done.")
    
    def _build_batch_read_state(self):
        self.batch_read_state = self.BatchReadState(self.shuffled_training_data, self._rng)

    def _build_all_data(self):
        
        logging.getLogger(__name__).info("read training data, devel data, test data...")
        self.raw_training_data = read_training_data()
        self.raw_devel_data = read_devel_data()
        self.raw_test_data = read_test_data()
        logging.getLogger(__name__).info("done")
        self._build_worddict()
        self._build_tagdict()
        self._build_training_data()
        self._build_devel_data()
        self._build_test_data()
        self._build_batch_read_state()

    def _replace_wordidx2unk(self, wordidx):
        ''' replace word to unk. 
        using the strategy: word_cnt <= replace_cnt && random() <= replace_prob.
        '''
        try:
            if ( self.counter[wordidx] <= self._unkreplace_cnt 
                 and self._rng.random() < self._unkreplace_prob) :
                return self.unk_idx
            else:
                return wordidx
        except IndexError:
            print("invalid word index:{} (word dict size: {}). using <unk> replace.".format(wordidx, self.worddict_sz),file=sys.stderr)
            return self.unk_idx
    
    def _init_window_queue_in_current_state(self, window_queue, x_list, pos):
        half_sz = self._window_sz // 2
        # left
        for i in range(half_sz, 0, -1):
            wordidx = x_list[pos - i] if pos - i >= 0 else self.sos_idx
            window_queue.append(wordidx)
        # center
        window_queue.append(x_list[pos])
        # right
        for i in range(1, half_sz + 1):
            wordidx = x_list[pos + i] if pos + i < len(x_list) else self.eos_idx
            window_queue.append(wordidx)
    
    def convert_word2idx(self, word):
        return self.word2idx[word] if word in self.word2idx else self.unk_idx

    def convert_tag2idx(self, tag):
        return self.tag2idx[tag] # if tag not in tag2idx, just die.

    def convert_idx2word(self, wordidx):
        return self.idx2word[wordidx] if 0 <= wordidx < self.worddict_sz else "<invlaid>"

    def convert_idx2tag(self, tagidx):
        return self.idx2tag[tagidx] if 0 <= tagidx < self.tagdict_sz else "<invalid>"
    
    def get_mlp_next_batch_training_data(self):
        '''
        generate one batch data for training.
        @return (X, Y) X is list of window training data, y is the corresponding tag list.
        '''
        X = []
        Y = []
        state = self.batch_read_state
        window_queue = collections.deque(maxlen=self._window_sz)
        half_sz = self._window_sz // 2
        # check instance state
        if state.has_instance_end():
            state.move2next_instance()
        original_instance = state.get_current_instance()
        generate_unkreplaced_x = lambda ox: [ self._replace_wordidx2unk(wordidx) for wordidx in ox  ]
        x_list = generate_unkreplaced_x(original_instance[0])
        y_list = original_instance[1]
        # init the queue
        self._init_window_queue_in_current_state(window_queue, x_list, state.get_current_pos())
        X.append(list(window_queue))
        Y.append(y_list[state.get_current_pos()])
        # continues
        instance_cnt = 1
        while instance_cnt < self._batch_sz:
            state.move2next_pos()
            while state.has_instance_end(): # there exits sentence with only 1 token 
                # update x_list, y_list
                state.move2next_instance()
                original_instance = state.get_current_instance()
                x_list = generate_unkreplaced_x(original_instance[0])
                y_list = original_instance[1]
                # re-init the window queue
                self._init_window_queue_in_current_state(window_queue, x_list, state.get_current_pos())
                X.append(list(window_queue))
                Y.append(y_list[state.get_current_pos()])
                instance_cnt += 1
                if instance_cnt >= self._batch_sz:
                    break
                state.move2next_pos() # need move to next pos
            pos = state.get_current_pos()
            # push the comming word index to window queue
            window_queue.append( x_list[pos + half_sz] if pos + half_sz < len(x_list) else self.eos_idx )
            X.append(list(window_queue))
            Y.append(y_list[pos])
            instance_cnt += 1
        state.move2next_pos() # for next batch generate.
        return X, Y

    def get_mlp_devel_data(self):
        '''
        generate devel data for varifying model.
        @return (X, Y), X has format: 
                        [ 
                            [   
                                [Wli1 , Wl2, ..., Wc, ..., Wrj2, Wrj1 ], ... <- window list
                            ]  , ... <- sentence
                        ],
                        Y has format: 
                        [ 
                            [ ...  ], ... <- sentence 
                        ]
        '''
        X = []
        Y = []
        window_queue = collections.deque(maxlen=self._window_sz)
        half_sz = self._window_sz // 2
        for (x_list, y_list) in self.devel_data:
            # processing X
            window_list = []
            # 1. init the window deque
            for i in range(half_sz):
                window_queue.append(self.sos_idx)
            window_queue.append(x_list[0])
            for i in range(1, 1 + half_sz):
                window_queue.append( x_list[i] if i < len(x_list) else self.eos_idx )
            window_list.append(list(window_queue))
            # 2. continues
            for pos in range(1, len(x_list)):
                window_queue.append(x_list[pos + half_sz] if pos + half_sz < len(x_list) else self.eos_idx)
                window_list.append(list(window_queue))
            X.append(window_list)
            # processing Y
            Y.append(y_list)
        return X, Y

    def get_mlp_test_data(self):
        '''
        generate mlp test data.
        @return X, format as: 
                    [ 
                        [ 
                            [ w1, w2, ...w_sz ], ... <- window list  
                        ] ,... <- sentence
                    ]
        '''
        X = []
        window_queue = collections.deque(maxlen=self._window_sz)
        half_sz = self._window_sz // 2
        for x_list in self.test_data:
            window_list = []
            # 1. init the window deque
            for i in range(half_sz):
                window_queue.append(self.sos_idx)
            window_queue.append(x_list[0])
            for i in range(1, 1 + half_sz):
                window_queue.append( x_list[i] if i < len(x_list) else self.eos_idx )
            window_list.append(list(window_queue))
            # 2. continues
            for pos in range(1, len(x_list)):
                window_queue.append(x_list[pos + half_sz] if pos + half_sz < len(x_list) else self.eos_idx)
                window_list.append(list(window_queue))
            X.append(window_list)
        return X
    
    @property
    def window_sz(self):
        return self._window_sz
    
    @property
    def batch_sz(self):
        return self._batch_sz
    
    @property
    def worddict_sz(self):
        return self._worddict_sz
    
    @property
    def tagdict_sz(self):
        return self._tagdict_sz
    
    @property
    def iterate_time(self):
        '''
        count +1 when training data has been read once.
        '''
        return self.batch_read_state.nr_shuffled_time






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
