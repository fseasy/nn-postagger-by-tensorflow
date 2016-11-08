#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from dataset_handler.reader import ( read_training_data,
        read_devel_data,
        read_test_data )


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
    
    def __init__(self, window_sz=5, batch_sz=128):
        self.window_sz = window_sz
        self.batch_sz = batch_sz
        self.shuffled_training_data = read_training_data()
        self.devel_data = read_devel_data()
        self.test_data = read_test_data()
        self.batch_read_state = BatchReadState()

    def get_next_batch_training_data(self):
        X = []
        Y = []
        for i in range(self.batch_sz):
            if self.batch_read_state.has_instance_end() :
                self.batch_read_state.move2next_instance()
            if self.batch_read_state.has_dataset_end():
                # shuffle dataset again and rest batch read state


