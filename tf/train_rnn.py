#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
train procedure
'''
from __future__ import print_function
from __future__ import division

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
        sess.run(tf.global_variables_initializer())
        lr_value = 0.1
        for nr_epoch in range(epoch_num):
            # init the mini-batch generator
            data_generator = data_batch.batch_training_data_generator(
                **train_generator_param
            )
            # one epoch run
            for i, (batch_x, batch_y, sequence_len) in enumerate(data_generator):
                # update lr
                sess.run(lr_update_op, feed_dict={
                    new_lr_placeholder: lr_value
                })
                # mini-batch run, cal loss and back-propagate
                r = sess.run({"loss": loss, "back_prop": back_prop_op}, feed_dict={
                    input_placeholder: batch_x,
                    y_placeholder: batch_y,
                    sequence_len_placeholder: sequence_len
                })
                print("epoch: {0}, mini-batch: {1}, loss= {2:.2f}".format(
                    nr_epoch, i, r["loss"]))
            lr_value = lr_value / (nr_epoch + 1)

if __name__ == "__main__":
    TEST_TRAIN_FPATH = "data/sample/train.data"
    training_data = data_process.get_training_data(TEST_TRAIN_FPATH)
    datadef = data_process.datadef
    seed = 1234
    model_param = ModelParam(
        rng_seed=seed,
        word_num=datadef.word_num,
        embedding_dim=10,
        x_padding_id=datadef.x_padding_id,
        tag_num=datadef.tag_num,
        y_padding_id=datadef.y_padding_id,
        max_timestep_in_global=data_process.get_annotated_max_len_in_global(
            training_data
        ),
        rnn_h_dim_list=[5, ],
        rnn_dropout_rate_list=[0., ],
        rnn_type="rnn"
    )
    m = RNNModel(model_param)
    train(m, training_data,
        batch_size=2,
        epoch_num=5,
        seed=seed
    )
    
    
