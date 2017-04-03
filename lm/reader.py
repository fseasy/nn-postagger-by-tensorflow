
'''
copy and annotation for https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
'''

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import collections
import os

import tensorflow as tf

def _read_words(filename):
    '''
    把换行符（每句结尾）替换为<eos>, 然后整个读进来，按空格分隔（相当于每个词作为一个token），得到词构成的列表
    '''
    #tf.gfile => 
    # add supporting Google Cloud Storage, HDFS with respect to python file obj 
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
    '''
    调用_read_words得到filename里的词表，然后完成 词 -> id 的映射。 做法很tricky
    '''
    data = _read_words(filename)
    # { word: cnt, ... }
    counter = collections.Counter(data)
    # descending order 
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0] ))

    # zip(*count_pairs) =>
    # zip((w1, c1), (w2, c2), ...) =>  ([w1, w2, ...], [c1, c2, ...])
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    '''
    输入文件和词表，
    把文件里的词转为id序列。如果词不在词表里，直接丢掉。
    '''
    data = _raed_words(filename)
    return [ word_to_id[word] for word in data in word in word_to_id  ]

