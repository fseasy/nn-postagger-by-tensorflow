#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import tensorflow as tf

class ModelParam(object):
    DTYPE=tf.float32
    '''
    model param.
    '''
    def __init__(self,
            rng_seed=1234,
            word_num=10,
            embedding_dim=5,
            x_padding_id=0,
            tag_num=4,
            y_padding_id=0,
            max_timestep_in_global=10,
            rnn_h_dim_list=[5,],
            rnn_dropout_rate_list=[0.,],
            rnn_type="rnn",
            **kwargs):
        self.rng_seed = rng_seed
        self.word_num = word_num
        self.embedding_dim = 5
        self.x_padding_id = x_padding_id
        self.tag_num = tag_num
        self.y_padding_id = y_padding_id
        self.max_timestep_in_global = (
                max_timestep_in_global)
        self.rnn_h_dim_list = rnn_h_dim_list
        self.rnn_dropout_rate_list=rnn_dropout_rate_list
        self.rnn_type = rnn_type


class RNNModel(object):
    '''
    RNN model
    '''
    
    def __init__(self, model_param):
        self._p = model_param
        tf.set_random_seed(self._p.rng_seed)

    def _get_rnn_cell(self):
        rnn_type = self._p.rnn_type
        hidden_dim_list = self._p.rnn_h_dim_list
        dropout_rate_list = self._p.rnn_dropout_rate_list
        if len(hidden_dim_list) != len(dropout_rate_list):
            raise ValueError(("un-match param length for "
                    "h dim list({}) and "
                    "dropout rate list({})").format(
                    len(hidden_dim_list),
                    len(dropout_rate_list)))
        if rnn_type == "rnn":
            cell_cls = tf.contrib.rnn.BasicRNNCell
        elif rnn_type == "gru":
            cell_cls = tf.contrib.rnn.GRUCell
        elif rnn_type == "lstm":
            cell_cls = tf.contrib.rnn.LSTMCell
        cell_list = [ cell_cls(hidden_dim) for hidden_dim 
                      in hidden_dim_list ]
        for i in range(len(dropout_rate_list)):
            dropout_rate = dropout_rate_list[i]
            if dropout_rate > 0.:
                cell_list[i] = tf.contrib.rnn.DropoutWrapper(
                        cell_list[i],
                        output_keep_prob=1.-dropout_rate)
        if len(cell_list) > 1:
            cell = tf.contrib.rnn.MultiRNNCell(cell_list)
        else:
            cell = cell_list[0]
        return cell

    def _create_variables(self):
        '''
        init variable if not inited.
        shoulded be called before build_graph.
        '''
        # check flag
        if (hasattr(self, "_has_init_variable") 
            and self._has_init_variable):
            return
        word_num = self._p.word_num
        with tf.variable_scope("Embedding"):
            r = tf.cast(tf.sqrt( 6. / self._p.embedding_dim ), 
                    self._p.DTYPE)
            self._lookup_table = tf.get_variable(
                    "lookup_table",
                    shape=[word_num,
                           self._p.embedding_dim], 
                    initializer=tf.random_uniform_initializer(
                           -r, r),
                    trainable=True,
                    dtype=self._p.DTYPE)
        x_padding_id = self._p.x_padding_id
        raw_mask_lookup = ([[1.]]*x_padding_id + [[0.]] + 
                           [[1.]]*(word_num - x_padding_id -1))
        with tf.variable_scope("Embedding"):
            self.mask_lookup_table = tf.get_variable("mask_lookup_table",
                    initializer=raw_mask_lookup,
                    dtype=self._p.DTYPE,
                    trainable=False)
        # softmax w
        h_dim = self._p.rnn_h_dim_list[-1]
        with tf.variable_scope("Softmax"):
            tag_num = self._p.tag_num
            r = tf.cast(tf.sqrt(6. / (h_dim + tag_num)), self._p.DTYPE)
            W = tf.get_variable("Softmax_W",
                    shape=[h_dim, tag_num],
                    initializer=tf.random_uniform_initializer(
                        -r, r),
                    dtype=self._p.DTYPE)
            b = tf.get_variable("Softmax_b",
                    shape=[tag_num,],
                    initializer=tf.constant_initializer(0.),
                    dtype=self._p.DTYPE)
            self.W = W
            self.b = b
        self.cell = self._get_rnn_cell()
        # set the flag
        self._has_init_variable = True
    
        
    def _build_input(self):
        timestep = self._p.max_timestep_in_global
        self.input_placeholder = tf.placeholder(
                tf.int32,
                shape=[None, timestep],
                name="batch_x")
        self.sequence_len_placeholder = tf.placeholder(
                tf.int32,
                shape=[None,],
                name="sequence_len")
        input_embedding = tf.nn.embedding_lookup(self._lookup_table, 
                self.input_placeholder)
        # mask the padding!
        self.mask_embedding = tf.nn.embedding_lookup(self.mask_lookup_table,
                self.input_placeholder)
        self.batch_embedding = tf.multiply(input_embedding, 
                self.mask_embedding) # broadcast
        return (self.input_placeholder, self.sequence_len_placeholder, self.batch_embedding)
        #return (self.batch_X_id, self.sequence_len, input_embedding)
    

    def build_graph(self):
        '''
        build graph and get the handler for input, sequence_len
        Returns:
            batch_input_placeholder: batch input placeholder
            sequence_len_placeholder: sequence length placeholer
        '''
        if (hasattr(self, "_has_built_graph") and
            self._has_built_graph):
            return self.input_placeholder, self.sequence_len_placeholder
        self._create_variables()
        input_placeholder, sequence_len, input_embedding = self._build_input()
        cell = self.cell
        output_list, _ = tf.nn.dynamic_rnn(cell, input_embedding, 
                sequence_length=sequence_len,
                parallel_iterations=2,
                dtype=self._p.DTYPE)

        # reshape for matmul efficiency
        embedding_dim = self._p.embedding_dim
        timestep = self._p.max_timestep_in_global
        h_dim = self._p.rnn_h_dim_list[-1]
        reshaped_input = tf.reshape(output_list, [-1, h_dim])
        reshaped_logits = tf.matmul(reshaped_input, self.W) + self.b
        tag_num = self._p.tag_num
        self.logits = tf.reshape(reshaped_logits, [-1, timestep, tag_num])
        # set the flag
        self._has_built_graph = True
        return input_placeholder, sequence_len
    
    def loss(self):
        '''
        get loss express and handler for gold tag placeholder.
        Returns:
            y_placeholder: tag placeholder
            loss: loss expression
        '''
        if (hasattr(self, "_has_built_loss") 
            and self._has_built_loss):
            return self.batch_y_placeholder, self.loss
        logits = self.logits
        timestep = self._p.max_timestep_in_global
        self.batch_y_placeholder = tf.placeholder(
                tf.int32,
                shape=[None, timestep])
        # batch_loss shape = [batch_size, time_step]
        batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=self.batch_y_placeholder)
        mask4label = tf.reshape(self.mask_embedding, [-1, timestep])
        masked_loss = tf.multiply(batch_loss, mask4label)
        
        # eevry_sequce_loss shape = [batch_size,]
        every_sequence_loss = tf.reduce_sum(masked_loss, reduction_indices=1)
        # shape = [batch_size, ]
        sequence_len_float = tf.cast(self.sequence_len_placeholder, tf.float32) 
        sequence_mean_loss = every_sequence_loss / sequence_len_float
        loss = tf.reduce_mean(sequence_mean_loss)
        self.loss = loss
        self._has_build_loss = True
        return self.batch_y_placeholder, loss
    
    def predict_prob(self):
        '''
        get every position probability.
        Returns:
            probabilities: expression, 
                           shape = [batch_size, max_timestep, num_tag]
        '''
        if (hasattr(self, "_has_built_predict_prob")
            and self._has_built_predict_prob):
            return self.predict_prob_op
        logits = self.logits
        predict_prob_op = tf.nn.softmax(logits)
        self.predict_prob_op = predict_prob_op
        self._has_built_predict_prob = True
        return self.predict_prob_op

    def predict(self):
        '''
        predict tag for batch sequence.
        Returns:
            tags: expression, 
                  shape = [batch_size, max_timestep]
        '''
        if (hasattr(self, "_has_built_predict")
            and self._has_built_predict):
            return self.predict_op
        probs = self.predict_prob()
        # shape = [batch_size, timesteps]
        predict_op = tf.arg_max(probs, dimension=2)
        self.predict_op = predict_op
        self._has_built_predict = True
        return predict_op

def __unittest_get_loss():
    model_param = ModelParam(
            max_timestep_in_global=4 
    )
    model = RNNModel(model_param)
    input_placeholder, sequence_len_placeholder = model.build_graph()
    y_placeholder, loss = model.loss()
    with tf.Session() as sess:
        x = [ [2, 3, 7, 1],
              [1, 3, 6, 0]]
        y = [ [1, 2, 0, 0],
              [3, 3, 0, 0]]
        seq_len = [4, 2]
        sess.run(tf.global_variables_initializer())
        loss_value = sess.run(loss, feed_dict={
                input_placeholder: x,
                sequence_len_placeholder: seq_len,
                y_placeholder: y
            })
        print(loss_value)


if __name__ == "__main__":
    __unittest_get_loss()
