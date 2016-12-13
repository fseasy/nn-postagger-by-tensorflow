#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import print_function
from __future__ import division

import sys
import os
import math
from contextlib import contextmanager
import tensorflow as tf
# not good
_package_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(_package_dir)

from mlp.mlp_data import MlpData
from utils.tf_utils import TFUtils

RandomSeed = 1234

class MlpTagger(object):
    def __init__(self):
        self._graph = tf.Graph()
        # session and config
        config = tf.ConfigProto(log_device_placement=True)
        config.gpu_options.allow_growth=True
        self._sess = tf.Session(graph=self._graph, config=config)
        
        self._batch_x_input_expr = None
        self._batch_y_input_expr = None
        self._batch_logit_expr = None
    
    @contextmanager
    def _set_context(self):
        g_mgr = self._graph.as_default()
        g = g_mgr.__enter__()
        #d_mgr = tf.device("/gpu:0")
        #d_mgr.__enter__()
        yield g
        #d_mgr.__exit__(*sys.exc_info())
        g_mgr.__exit__(*sys.exc_info())
    
    def _set_random_seed(self, random_seed):
        with self._set_context():
            tf.set_random_seed(random_seed)

    def _build_annotated_input_placeholder(self, window_sz):
        with self._set_context():
            with tf.name_scope("input"):
                batch_x_input = tf.placeholder(tf.int32, shape=(None, window_sz))
                batch_y_input = tf.placeholder(tf.int32, shape=(None,))
            # export
            self._batch_x_input_expr = batch_x_input
            self._batch_y_input_expr = batch_y_input
    
    def _build_unannotated_input_placeholder(self, window_sz):
        if self._batch_x_input_expr is not None:
            return
        with self._set_context():
            with tf.name_scope("input"):
                batch_x_input = tf.placeholder(tf.int32, shape=(None, window_sz))
        self._batch_x_input_expr = batch_x_input

    def _build_logit_expr(self, window_sz, worddict_sz, embedding_dim, hidden_dim, tagdict_sz):
        with self._set_context():
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
                            tf.zeros( (hidden_dim,) ),
                            name="b"
                        )
                batch_net = tf.matmul(batch_x_concated, w, transpose_b=True) + b
                batch_hidden_out = tf.nn.relu(batch_net)
            with tf.name_scope("softmax"):
                w_o = tf.Variable( 
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
            self._batch_logit_expr = batch_logit # for eval or predict
            return batch_logit
            
    def _build_train_op(self, optimizer_constructor, learning_rate):
        with self._set_context():
            with tf.name_scope("loss"):
                # for `sparse_softmax_cross_entropy_with_logits`,
                #     the input label is just the right index
                # for `softmax_cross_entropy_with_logits`,
                #     the input label is the distribution at the label set.
                batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        self._batch_logit_expr, self._batch_y_input_expr, name="batch_loss")
                loss = tf.reduce_mean(batch_loss, name="loss")
            tf.summary.scalar('loss_summary', loss)
            optimizer  = optimizer_constructor(learning_rate)
            global_step = tf.Variable(0, name="global_step", trainable=False) # record how much steps updated.
            train_op = optimizer.minimize(loss, global_step=global_step)
        # export the symbol-variable
        self._train_op = train_op
        return train_op

    def _build_devel_op(self):
        # devel may be separately, or including in training.
        if self._batch_logit_expr is None:
            self._build_logit_expr()
        with self._set_context():
            with tf.name_scope("eval"):
                correct = tf.nn.in_top_k(self._batch_logit_expr, self._batch_y_input_expr, 1, name="is_correct")
                cnt_op = tf.reduce_sum(tf.cast(correct, tf.int32))
        self._devel_op = cnt_op
        return cnt_op

    def _build_init_op(self):
        with self._set_context():
            init_op = tf.initialize_all_variables()
        self._init_op = init_op
        return init_op

    def train(self, mlp_data, nr_epoch=15, batch_sz=64,
            embedding_dim=50, hidden_dim=100, opt_constructor=tf.train.GradientDescentOptimizer,
            learning_rate=0.01, random_seed=RandomSeed):
        '''
        do training.
        @return (best_devel_score, devel_score_list)
        '''
        window_sz = mlp_data.window_sz
        self._set_random_seed(random_seed)
        self._build_annotated_input_placeholder(window_sz)
        self._build_logit_expr(window_sz, mlp_data.worddict_sz, embedding_dim, hidden_dim, mlp_data.tagdict_sz)
        self._build_train_op(opt_constructor, learning_rate)
        self._build_devel_op()
        self._build_init_op()
       
        # build saver
        with self._set_context():
            saver = tf.train.Saver()

        self._sess.run(self._init_op)
        
        batch_cnt = 0
        devel_freq = 400
        best_devel_acc = 0.
        devel_score_list = [] 
        def do_devel(header, devel_score_list):
            print(header)
            devel_acc = self.devel(mlp_data)
            devel_score_list.append(devel_acc)
            if devel_acc > best_devel_acc:
                print("better model found. save it.")
                saver.save(self._sess, "model.ckpt")
            return devel_acc

        training_data = mlp_data.build_training_data()
        for i in range(nr_epoch):
            for batch_x, batch_y in mlp_data.batch_data_generator(training_data, batch_sz):
                training_feed_dict = {
                    self._batch_x_input_expr: batch_x,
                    self._batch_y_input_expr: batch_y
                }
                self._sess.run(self._train_op, feed_dict=training_feed_dict)
                batch_cnt += 1
                #if batch_cnt % devel_freq == 0 :
                #    best_devel_acc = max(best_devel_acc, do_devel("end of another {} batches".format(devel_freq), devel_score_list))
            devel_acc = do_devel("end of epoch {}, do devel".format(i + 1), devel_score_list)
            best_devel_acc = max(best_devel_acc, devel_acc)
        return (best_devel_acc, devel_score_list)
            

    def devel(self, mlp_data):
       devel_batch_sz = 32
       correct_cnt = total_cnt = 0
       devel_data = mlp_data.build_devel_data()
       for batch_x, batch_y in mlp_data.batch_data_generator(devel_data, devel_batch_sz, fill_when_not_enouth=False):
           devel_feed_dict = {
                       self._batch_x_input_expr: batch_x,
                       self._batch_y_input_expr: batch_y
                   }
           correct_cnt += self._sess.run(self._devel_op, feed_dict=devel_feed_dict)
           total_cnt += len(batch_x)
       acc = float(correct_cnt) / total_cnt * 100
       print("devel accuracy: {:.2f}% (correct: {}, total: {})".format(acc, correct_cnt, total_cnt))
       return acc


def train():
    params = {
            "window_sz": 5,
            "nr_epoch": 15,
            "batch_sz": 64,
            "embedding_dim": 50,
            "hidden_dim": 100,
            "opt_name": "adagrad",
            "learning_rate": 0.001
    }
    mlp_data = MlpData(RandomSeed, params["window_sz"] )
    mlp_tagger = MlpTagger()
    best_devel_score, devel_score_list = mlp_tagger.train(mlp_data, nr_epoch=params["nr_epoch"], batch_sz=params["batch_sz"],
                     embedding_dim=params["embedding_dim"], hidden_dim=params["hidden_dim"],
                     opt_constructor=TFUtils.get_optimizer(params["opt_name"]), learning_rate=params["learning_rate"])
    
    print("best devel score: {:.2f}".format(best_devel_score))
    with open("score_list.{}".format(params["learning_rate"]), "wt") as logf:
        for score in devel_score_list:
            print(score, file=logf)

def main():
    train()

if __name__ == "__main__":
    main()

