#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import random

import tensorflow as tf

import data_process
import data_batch
from rnn_model import (ModelParam, 
                       RNNModel)

DTYPE = ModelParam.DTYPE

def get_optimizer(loss_expr):
    with tf.name_scope("Train"):
        learning_rate = tf.get_variable("learning_rate",
                shape=[],
                initializer=tf.constant_initializer(0.1),
                dtype=DTYPE)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        back_propagation_op = opt.minimize(loss_expr)
        return back_propagation_op, learning_rate

def get_learning_rate_update_op(lr_var):
    '''
    get learning rate operator.
    Args:
        lr_var: learning rate variable.
    Returns:
        A tuple, (new_lr_placeholder, update_lr_op)
        - new_lr_placeholder: new learning rate placeholder
        - update_lr_op: update learning rate operator
    '''
    new_lr = tf.placeholder(DTYPE, shape=[], name="new_lr")
    update_lr = lr_var.assign(new_lr)
    return new_lr, update_lr


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
    with tf.Session() as sess:
        input_placeholder, sequence_len_placeholder = model.build_graph()
        y_placeholder, loss = model.loss()
        back_prop_op, lr_var = get_optimizer(loss)
        new_lr_placeholder, lr_update_op = get_learning_rate_update_op(lr_var)
        data_generator = data_batch.batch_training_data_generator(
            **train_generator_param
        )
        lr_value = 0.1
        for i, (batch_x, batch_y, sequence_len) in enumerate(data_genrator):
            # update lr
            sess.run(lr_update_op, feed_dict={
                new_lr_placeholder: lr_value
            })
            # mini-batch run
            r = sess.run({"loss": loss, "back_prop": back_prop_op}, feed_dict={
                input_placeholder: batch_x,
                y_placeholder: batch_y   
            })
            print("mini-batch: {0}, loss= {.2f}".format(i, r["loss"]))
            


    
    
