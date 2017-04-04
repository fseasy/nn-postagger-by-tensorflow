# -*- coding: utf-8 -*-

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
    data = _read_words(filename)
    return [ word_to_id[word] for word in data in word in word_to_id  ]

def ptb_raw_data(data_path=None):
    '''
    输入数据集文件夹路径，输出训练数据、开发集数据、测试数据、词表（词个数）
    '''
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    word_to_id = _build_vocab()
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

def  ptb_producer(raw_data, batch_size, num_steps, name=None):
    '''
    这定义了一个OP
    ''' 
    # name_scope(name, default_name, values)
    # values: this values are ensured in the same Graph with the current op
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        # convert_to_tensor, 保证输入的任意数据（原生Python、numpy、tensor）经过该函数转换后都成为tensor类型
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        # tf.size 返回数据中所有元素的个数，整型
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        # 把尾部数据丢掉，修改shape为 batch_size x batch_len
        data = tf.reshape(raw_data[0: batch_size*batch_len], [batch_size, batch_len])
        # 不知道这里为什么要-1 ？？
        # 看后面y的构成，因为要留出一个位置作为预测值，
        # 所以x就得少一个
        epoch_size = (batch_len - 1) // num_steps
        # 断言，不满足时会报错
        assertion = tf.assert_positive(epoch_size, 
            message="epoch_size == 0, decrease batch_size or num_steps")
        # 控制依存, 保证assertion在执行以下操作以前，必须执行了assertion
        with tf.control_dependencies([assertion]):
            # epoch_size 是batch数据的数量
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        # 将range(epoch_size) 加入到QueuRunner中
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        # 虽然strided_slice文档没有看懂，不过这个操作还是可以看明白的：
        # 从data里，slice取  (0, i*num_steps) -> (batch_size, (i+1)*num_steps) 的矩形区域
        x = tf.strided_slice(data, [0, i*num_steps],
            [batch_size, (i+1)*num_steps])
        # 补全shape信息
        x.set_shape([batch_size, num_steps])
        # 把下一个位置的字作为预测gold结果 => 这里可以看到为什么前面batch_len需要 -1 了
        y = tf.strided_slice(data, [0, i*num_steps + 1],
            [batch_size, (i+1)*num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y

if __name__ == "__main__":
    raw_data = list(range(1, 11))
    batch_size = 2
    num_steps = 2
    x, y = ptb_producer(raw_data, batch_size, num_steps)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        xval, yval = sess.run([x, y])
        xval, yval = sess.run([x, y])
        coord.request_stop()
        coord.join(threads)
    
        