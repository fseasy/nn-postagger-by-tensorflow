#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import os
import math
import tensorflow as tf
# not good
_cur_dir = os.path.dirname(__file__)
sys.path.append(_cur_dir)

from mlp_data import MlpData, RandomSeed

class MlpNet(object):
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.Session()
    
    def build_input(self, batch_sz, window_sz):
        with self.graph.as_default(), tf.device("/cpu:0"):
            with tf.name_scope("input"):
                batch_x_input = tf.placeholder(tf.int32, shape=(batch_sz, window_sz))
                batch_y_input = tf.placeholder(tf.int32, shape=(batch_sz,))
            # export
            self.batch_x_input = batch_x_input
            self.batch_y_input = batch_y_input

    def build_logit(self, batch_sz, window_sz, worddict_sz, embedding_dim, hidden_dim, tagdict_sz):
        with self.graph.as_default(), tf.device("/cpu:0"):
            with tf.name_scope("project"):
                embedding_table = tf.Variable( 
                                    tf.random_uniform( (worddict_sz, embedding_dim), -1., 1. ),
                                    name="embedding_table")
                #batch_x_embedding = tf.nn.embedding_lookup(embedding_table, batch_x_input)
                batch_x_embedding = tf.gather(embedding_table, batch_x_input)
                batch_x_concated = tf.reshape(batch_x_embedding, (batch_sz, embedding_dim * window_sz))
            with tf.name_scope("hidden"):
                w = tf.Variable(
                            tf.truncated_normal( (hidden_dim, embedding_dim * window_sz), 
                                                  stddev=math.sqrt( 6. / (embedding_dim * window_sz))  ),
                            name="w"
                        )
                b = tf.Variable(
                            tf.zeros( (hidden_dim,) ),
                            name="b"
                        )
                batch_net = tf.matmul(batch_x_concated, w, transpose_b=True) + b
                batch_hidden_out = tf.relu(batch_net)
            with tf.name_scope("softmax"):
                w_o = tf.Varibale( 
                            tf.truncated_normal( (tagdict_sz, hidden_dim), 
                                                 stddev=math.sqrt( 6. / hidden_dim )),
                            name="w_softmax"
                        )
                b_o = tf.Variable(
                            tf.zeros( (tagdict_sz,) ),
                            name="b_softmax"
                        )
                batch_logit = tf.matmul(batch_hidden_out, w_o,transpose_b=True) + b_o
           
            # export the symbol-variables
            self.batch_logit = batch_logit # for eval or predict
            self.batch_x_input = batch_x_input
            
        def build_train_op(self, learning_rate):
            with tf.name_scope("loss"):
                batch_loss = tf.sparse_softmax_cross_entropy_with_logits(batch_logits, batch_y_input, name="batch_loss")
                loss = tf.reduce_mean(batch_loss, name="loss")
            tf.summary.scalar('loss', self.loss, name="loss_summary", description="test for loss summary")
            optimizer  = tf.train.GradientDescentOptimizer(learning_rate)
            global_step = tf.Variable(0, name="global_step", trainable=False) # record how much steps updated.
            train_op = optimizer.minimize(loss, global_step=global_step)
            # export the symbol-variable
            self.batch_y_input = batch_y_input
            self.train_op = train_op

        def build_eval_op(self):
            
        

