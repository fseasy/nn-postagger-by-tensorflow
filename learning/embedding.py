# -*- coding: utf-8 -*-

'''
Embedding handling.
including creating, initialization, lookup, Mask-Padding-Zero
'''

from __future__ import division

import tensorflow as tf

WORDS_NUM = 10 # including padding-index
EMBEDDING_DIM = 5
PADDING_ID = 0

SEED = 1234

DTYPE = tf.float32

tf.set_random_seed(SEED)

# create and initialize
with tf.variable_scope("Embedding"):
    # see http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
    r = tf.sqrt(tf.cast(6 / EMBEDDING_DIM, dtype=DTYPE)) # => \sqrt( 6 / embedding_dim )
    lookup_table = tf.get_variable("lookup_table", shape=[WORDS_NUM, EMBEDDING_DIM],
                                   initializer=tf.random_uniform_initializer(minval=-r, maxval=r),
                                   trainable=True)


# mask the padding, method 1
# mask_padding_zero_op = tf.scatter_update(lookup_table, PADDING_ID, tf.zeros([EMBEDDING_DIM,], dtype=DTYPE))
# lookup_table = mask_padding_zero_op


# build the raw mask array
raw_mask_array = [[1.]] * PADDING_ID + [[0.]] + [[1.]] * (WORDS_NUM - PADDING_ID - 1)
with tf.variable_scope("Embedding"):
    mask_padding_lookup_table = tf.get_variable("mask_padding_lookup_table",
                                                initializer=raw_mask_array,
                                                dtype=DTYPE,
                                                trainable=False)

id_input = [ [1, 2], [1, 0] ]


embedding_input = tf.nn.embedding_lookup(lookup_table, id_input)

mask_padding_input = tf.nn.embedding_lookup(mask_padding_lookup_table, id_input)

embedding_input = tf.multiply(embedding_input, mask_padding_input) # broadcast

# mask padding zero


if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        embed_input_value = sess.run(embedding_input, feed_dict=None)
        print(embed_input_value)



# http://stackoverflow.com/questions/35769944/manipulating-matrix-elements-in-tensorflow


# https://github.com/tensorflow/tensorflow/issues/206