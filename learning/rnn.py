# -*- coding: utf-8 -*-

from tf.contrib.rnn import (GRUCell,         # cell
                            DropoutWrapper,  # wrapper
                            MultiRNNCell)    # wrapper

num_neurons = 30 # h_dim 
num_layers = 1 # stacked num
dropout = tf.placeholder(tf.float32)

cell = GRUCell(num_neurons) # (num_units, input_size, activations)
cell = DropoutWrapper(cell, output_keep_prob=1.-dropout) # cell, input_keep_prob, ouptut_keep_prob
cell = MultiRNNCell([cell] * num_layers) # (cells, state_is_Tuple=True)

max_len = 100
num_input_dim = 28

data = tf.placeholder(tf.float32, shape=[None, max_len, num_input_dim]) # tf.placeholder(dtype, shape=None, name)

output, state = tf.nn.dynamic_rnn(cell, data, parallel_iterations=2) # (cell, inputs, sequence_length, initial_state, dtype, parallel_iterations, swap_memory, time_major, scope)

# 转置，按照给定的顺序；如果不给定，则逆序转置
output = tf.transpose(output, [1, 0, 2]) #  tf.transpose(tensor, perm, name="transpose") perm is the permutation of dimentions of tensor

# 提取主列
last = tf.gather(output, int(output.get_shape()[0]) - 1) # (tensor, indices, valid_indices=True, name=None) 

