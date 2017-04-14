#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import data_process
import data_batch
from rnn_model import (ModelParam, 
                       RNNModel)

DTYPE = ModelParam.DTYPE

def get_optimizer():
    with tf.name_scope("Train"):
        learning_rate = tf.get_variable("learning_rate", 
                shape=[],
                initializer=tf.constant_initializer(0.1),
                dtype=DTYPE)
        opt = tf.trian.GradientDescentOptimizer(learning_rate)

def train(model,
        train_data,
        batch_size=20,
        epoch_num=1,
        learning_rate=0.1,
        seed=1234,
        use_unk_replace=True,
        replace_cnt_lower_bound=2,
        replace_prob_lower_bound=0.4):
    '''
    do train
    '''
    datadef = data_process.get_default_datadef()
    rng = random.Random(seed)
    train_generator_param = {
        "data": train_data,
        "batch_size":  batch_size,
        "x_padding_id": datadef.x_padding_id,
        "y_padding_id": datadef.y_padding_id,
        "rng": rng,
        "use_unk_replace": use_unk_replace,
        "word_cnt_dict": datadef.wordcnt_dict,
        "unk_id": datadef.unk_id,
        "replace_cnt_lower_bound": replace_cnt_lower_bound,
        "replace_prob_lower_bound": replace_prob_lower_bound
    }
    input_placeholder, sequence_len_placeholder = model.build_graph()
    y_placeholder, loss = model.loss()
    sess = tf.Session()
    
