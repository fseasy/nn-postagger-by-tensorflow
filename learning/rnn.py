# -*- coding: utf-8 -*-

# copying from http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/

# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
import tensorflow as tf
from tf.contrib.rnn import (GRUCell,         # cell
                            DropoutWrapper,  # wrapper
                            MultiRNNCell)    # wrapper

num_neurons = 30 # h_dim 
num_layers = 1 # stacked num
num_class = 10

dropout = tf.placeholder(tf.float32)

cell = GRUCell(num_neurons) # (num_units, input_size, activations)
cell = DropoutWrapper(cell, output_keep_prob=1.-dropout) # cell, input_keep_prob, ouptut_keep_prob
cell = MultiRNNCell([cell] * num_layers) # (cells, state_is_Tuple=True)

max_len = 100
num_input_dim = 28

data = tf.placeholder(tf.float32, shape=[None, max_len, num_input_dim]) # tf.placeholder(dtype, shape=None, name)
target = tf.placeholder(tf.int32, shape=[None, 1])


output, state = tf.nn.dynamic_rnn(cell, data, parallel_iterations=2) # (cell, inputs, sequence_length, initial_state, dtype, parallel_iterations, swap_memory, time_major, scope)

# 转置，按照给定的顺序；如果不给定，则逆序转置
output = tf.transpose(output, [1, 0, 2]) #  tf.transpose(tensor, perm, name="transpose") perm is the permutation of dimentions of tensor

# 提取主列
last = tf.gather(output, int(output.get_shape()[0]) - 1) # (tensor, indices, valid_indices=True, name=None) 

with tf.variable_scope("Softmax_param"):
    W = tf.get_variable("W", [num_neurons, num_class]) # using the default initializer: glorot_uniform_initializer
    b = tf.get_variable("b", [num_class,], initializer=tf.constant_initializer(0.0))

logits = tf.matmul(last, W) + b

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target))

