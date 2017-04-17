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

from tag_score import TagScore

DTYPE = ModelParam.DTYPE

def name2optimizer(opt_name):
    name2opt = {
            "sgd": tf.train.GradientDescentOptimizer,
            "adam": tf.train.AdamOptimizer
    }
    opt_name = opt_name.lower()
    default_optimizer = tf.train.GradientDescentOptimizer
    return name2opt.get(opt_name, default_optimizer)

def get_optimizer(loss_expr, opt_type="sgd"):
    with tf.name_scope("Train"):
        learning_rate = tf.get_variable("learning_rate",
                shape=[],
                initializer=tf.constant_initializer(0.1),
                dtype=DTYPE)
        optmizer = name2optimizer(opt_type)
        opt = optmizer(learning_rate)
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
        replace_prob_lower_bound=0.2,
        dev_data=None,
        dev_batch_size=20):
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
    with tf.Session().as_default() as sess:
        input_placeholder, sequence_len_placeholder = model.build_graph()
        y_placeholder, loss = model.loss()
        back_prop_op, lr_var = get_optimizer(loss, "Adam")
        new_lr_placeholder, lr_update_op = get_learning_rate_update_op(lr_var)
        sess.run(tf.global_variables_initializer())
        lr_value = learning_rate
        saver = tf.train.Saver(max_to_keep=2)
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
                #print("epoch: {0}, mini-batch: {1}, loss= {2:.2f}".format(
                #    nr_epoch, i, r["loss"]))
            #lr_value = lr_value * (0.5 ** nr_epoch + 1)
            saver.save(sess, "saver/test", global_step=nr_epoch+1)
            if dev_data:
                holdout_in_train(model,
                        develop_data=dev_data,
                        batch_size=dev_batch_size,
                        sess=sess)

def holdout_in_train(model, 
        develop_data,
        sess,
        batch_size=20):
    '''
    holdout
    '''
    datadef = data_process.get_default_datadef()
    develop_batch_generator_params = {
       "data": develop_data,
       "batch_size": batch_size,
       "x_padding_id": datadef.x_padding_id,
       "y_padding_id": datadef.y_padding_id,
    }
    score_obj = TagScore(datadef.id2tag)
    with sess.as_default():
        input_placeholder, seq_len_placeholder = model.build_graph()
        pred_op = model.predict()
        batch_data_generator = data_batch.batch_develop_data_generator(
            **develop_batch_generator_params         
        )
        for batch_id, (x, y, seq_len) in enumerate(batch_data_generator):
            pred = sess.run(pred_op, feed_dict={
                input_placeholder: x,
                seq_len_placeholder: seq_len
            })
            score_obj.partial_statistic(batch_gold=y, batch_pred=pred, sent_len_list=seq_len)
            #print(y)
            #print(list(pred))
    score_result = score_obj.get_statistic_result() 
    print(score_obj.get_statistic_result_str())
    return score_result 

def holdout(model, 
        develop_data,
        sess=None,
        batch_size=20):
    '''
    holdout
    '''
    datadef = data_process.get_default_datadef()
    develop_batch_generator_params = {
       "data": develop_data,
       "batch_size": batch_size,
       "x_padding_id": datadef.x_padding_id,
       "y_padding_id": datadef.y_padding_id,
    }
    score_obj = TagScore(datadef.id2tag)
    if sess is None:
        sess = tf.Session()
    with sess:
        input_placeholder, seq_len_placeholder = model.build_graph()
        pred_op = model.predict()
        batch_data_generator = data_batch.batch_develop_data_generator(
            **develop_batch_generator_params         
        )
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint("saver")
        print(checkpoint)
        saver.restore(sess, checkpoint)
        for batch_id, (x, y, seq_len) in enumerate(batch_data_generator):
            pred = sess.run(pred_op, feed_dict={
                input_placeholder: x,
                seq_len_placeholder: seq_len
            })
            score_obj.partial_statistic(batch_gold=y, batch_pred=pred, sent_len_list=seq_len)
            #print(y)
            #print(list(pred))
    score_result = score_obj.get_statistic_result() 
    print(score_obj.get_statistic_result_str())
    return score_result 


if __name__ == "__main__":
    TEST_TRAIN_FPATH = "data/ho.2000"
    training_data = data_process.get_training_data(TEST_TRAIN_FPATH)
    develop_data = data_process.get_develop_data(TEST_TRAIN_FPATH)
    datadef = data_process.datadef
    seed = 1234
    model_param = ModelParam(
        rng_seed=seed,
        word_num=datadef.word_num,
        embedding_dim=50,
        x_padding_id=datadef.x_padding_id,
        tag_num=datadef.tag_num,
        y_padding_id=datadef.y_padding_id,
        max_timestep_in_global=data_process.get_annotated_max_len_in_global(
            training_data
        ),
        rnn_h_dim_list=[50, ],
        rnn_dropout_rate_list=[0.1, ],
        rnn_type="rnn"
    )
    m = RNNModel(model_param)
    train(m, training_data,
        batch_size=64,
        epoch_num=13,
        seed=seed,
        learning_rate=0.005,
        dev_data=develop_data,
        dev_batch_size=100
    )
    #holdout(m,
    #    develop_data,
    #    batch_size=2
    #)
