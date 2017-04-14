'''
rnn learning
'''
# -*- coding: utf-8 -*-

# copying from http://danijar.com/introduction-to-recurrent-networks-in-tensorflow/

# http://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import (GRUCell,         # cell
                            BasicRNNCell,
                            DropoutWrapper,  # wrapper
                            MultiRNNCell)    # wrapper

num_neurons = 30 # h_dim 
num_layers = 1 # stacked num
num_class = 10

dropout = tf.placeholder(tf.float32)

cell = BasicRNNCell(num_neurons) # (num_units, input_size, activations)
cell = DropoutWrapper(cell, output_keep_prob=1.-dropout) # cell, input_keep_prob, ouptut_keep_prob
cell = MultiRNNCell([cell] * num_layers) # (cells, state_is_Tuple=True)

max_len = 20
num_input_dim = 28


#data = tf.placeholder(tf.float32, shape=[None, max_len, num_input_dim], name="data") # tf.placeholder(dtype, shape=None, name)
data = tf.placeholder(tf.float32, shape=[None, max_len, num_input_dim])
target = tf.placeholder(tf.int32, shape=[None,], name="target")


# dtype 与 initial_state 必须指定一个！
output, state = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32, parallel_iterations=2) # (cell, inputs, sequence_length, initial_state, dtype, parallel_iterations, swap_memory, time_major, scope)

# 转置，按照给定的顺序；如果不给定，则逆序转置
output = tf.transpose(output, [1, 0, 2]) #  tf.transpose(tensor, perm, name="transpose") perm is the permutation of dimentions of tensor

# 提取列
last = tf.gather(output, int(output.get_shape()[0]) - 1) # (tensor, indices, valid_indices=True, name=None) 

with tf.variable_scope("Softmax_param"):
    W = tf.get_variable("W", [num_neurons, num_class]) # using the default initializer: glorot_uniform_initializer
    b = tf.get_variable("b", [num_class,], initializer=tf.constant_initializer(0.0))

logits = tf.matmul(last, W) + b

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target))

grad = tf.gradients(loss, data)

trainable_variables = tf.trainable_variables()
variables_gradients = tf.gradients(loss, trainable_variables)

new_learning_rate = tf.placeholder(tf.float32, shape=())

with tf.variable_scope("training"):
    learning_rate = tf.get_variable("learning_rate",shape=(), trainable=False, initializer=tf.constant_initializer(1.0), dtype=tf.float32)

    learning_rate_update = tf.assign(learning_rate, new_learning_rate, name="learning_rate_update")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 10
    max_len_val = 10
    data_np = np.random.ranf((batch_size, max_len_val, num_input_dim))
    labels = np.random.random_integers(0, num_class - 1, (batch_size,))
    dropout_rate = 0.4
    loss_s, g = sess.run([loss, grad], feed_dict={data: data_np, target: labels, dropout: dropout_rate, max_len: 10})
    print(loss_s)
    #print(g)
    print(trainable_variables)
    print(variables_gradients)
    for var in trainable_variables:
        print(var.name)
    
