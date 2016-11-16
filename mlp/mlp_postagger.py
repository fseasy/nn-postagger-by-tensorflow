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

class MlpTagger(object):
    def __init__(self):
        self._graph = tf.Graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(graph=self._graph,config=config)
        self._batch_x_input_expr = None
        self._batch_y_input_expr = None
        self._batch_logit_expr = None
    
    def _build_annotated_input_placeholder(self, window_sz):
        with self._graph.as_default(), tf.device("/cpu:0"):
            with tf.name_scope("input"):
                batch_x_input = tf.placeholder(tf.int32, shape=(None, window_sz))
                batch_y_input = tf.placeholder(tf.int32, shape=(None,))
            # export
            self._batch_x_input_expr = batch_x_input
            self._batch_y_input_expr = batch_y_input
    
    def _build_unannotated_input_placeholder(self, window_sz):
        if self._batch_x_input_expr is not None:
            return
        with self._graph.as_default(), tf.device("/cpu:0"):
            with tf.name_scope("input"):
                batch_x_input = tf.placeholder(tf.int32, shape=(None, window_sz))
        self._batch_x_input_expr = batch_x_input

    def _build_logit_expr(self, window_sz, worddict_sz, embedding_dim, hidden_dim, tagdict_sz):
        with self._graph.as_default(), tf.device("/cpu:0"):
            with tf.name_scope("project_layer"):
                embedding_table = tf.Variable( 
                                    tf.random_uniform( (worddict_sz, embedding_dim), -1., 1. ),
                                    name="embedding_table")
                #batch_x_embedding = tf.nn.embedding_lookup(embedding_table, self._batch_x_input_expr)
                batch_x_embedding = tf.gather(embedding_table, self._batch_x_input_expr)
                batch_x_concated = tf.reshape(batch_x_embedding, (-1, embedding_dim * window_sz))
            with tf.name_scope("hidden_layer"):
                w = tf.Variable(
                            tf.truncated_normal( (hidden_dim, embedding_dim * window_sz), 
                                                  stddev=math.sqrt( 6. / (embedding_dim * window_sz))  ),
                            name="w"
                        )
                b = tf.Variable(
                            tf.zeros( [hidden_dim,] ),
                            name="b"
                        )
                batch_net = tf.matmul(batch_x_concated, w, transpose_b=True) + b
                batch_hidden_out = tf.nn.relu(batch_net)
            with tf.name_scope("softmax"):
                w_o = tf.Variable( 
                            tf.truncated_normal( (tagdict_sz, hidden_dim), 
                                                 stddev=math.sqrt( 6. / hidden_dim )),
                            name="wdddd_softmax"
                        )
                print(tagdict_sz)
                b_o = tf.get_variable("dddd_softmax",[tagdict_sz],tf.float32)
                batch_logit = tf.matmul(batch_hidden_out, w_o,transpose_b=True) + b_o
           
            # export the symbol-variables
            self._batch_logit_expr = batch_logit # for eval or predict
            
    def _build_train_op(self, learning_rate):
        with self._graph.as_default(), tf.device("/cpu:0"):
            with tf.name_scope("loss"):
                # for `sparse_softmax_cross_entropy_with_logits`,
                #     the input label is just the right index
                # for `softmax_cross_entropy_with_logits`,
                #     the input label is the distribution at the label set.
                batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        self._batch_logit_expr, self._batch_y_input_expr, name="batch_loss")
                loss = tf.reduce_mean(batch_loss, name="loss")
            tf.summary.scalar('loss', loss, name="loss_summary", description="test for loss summary")
            optimizer  = tf.train.GradientDescentOptimizer(learning_rate)
            global_step = tf.Variable(0, name="global_step", trainable=False) # record how much steps updated.
            train_op = optimizer.minimize(loss, global_step=global_step)
        # export the symbol-variable
        self._train_op = train_op

    def _build_eval_op(self):
        # eval may be separately, or including in training.
        if self._batch_logit_expr is None:
            self._build_logit_expr()
        with self._graph.as_default() , tf.device("/cpu:0"):
            with tf.name_scope("eval"):
                correct = tf.nn.in_top_k(self._batch_logit_expr, self._batch_y_input_expr, 1, name="is_correct")
                cnt_op = tf.reduce_sum(tf.cast(correct, tf.int32))
        self._devel_op = cnt_op

    def _build_init_op(self):
        with self._graph.as_default(), tf.device("/cpu:0"):
            init_op = tf.initialize_all_variables()
        
        self._init_op = init_op

    def train(self, mlp_data, nr_epoch,
            embedding_dim, hidden_dim, learning_rate=0.01):
        window_sz = mlp_data.window_sz
        batch_sz = mlp_data.batch_sz
        self._build_init_op()
        self._build_annotated_input_placeholder(window_sz)
        self._build_logit_expr(window_sz, mlp_data.worddict_sz, embedding_dim, hidden_dim, mlp_data.tagdict_sz)
        self._build_train_op(learning_rate)
        self._build_eval_op()
        
        self._sess.run(self._init_op)
        while mlp_data.iterate_time < nr_epoch:
            batch_x, batch_y = mlp_data.get_mlp_next_batch_training_data()
            training_feed_dict = {
                        self._batch_x_input_expr: batch_x,
                        self._batch_y_input_expr: batch_y
                    }
            self._sess.run(self._train_op, feed_dict=training_feed_dict)


def main():
    mlp_data = MlpData()
    mlp_tagger = MlpTagger()
    mlp_tagger.train(mlp_data, 15, 50, 100)

if __name__ == "__main__":
    main()

