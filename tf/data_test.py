#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
data process and batch test
'''
from __future__ import print_function

import random

import data_process
import data_batch

TEST_TRAIN_FPATH = "data/sample/train.data"
TEST_DEV_FPATH = "data/sample/train.data"
TEST_TEST_FPATH = "data/sample/test.data"

def _print_annotated_dataset(X, Y):
    for word_id_list, tag_id_list in zip(X, Y):
        word_list = data_process.x_id_list2text_list(word_id_list)
        tag_list = data_process.y_id_list2text_list(tag_id_list)
        _get_line = lambda tl: u"\t".join(tl).encode("utf-8")
        print("{0}: {1}".format(len(word_list), _get_line(word_list)))
        print("{0}: {1}".format(len(tag_list), _get_line(tag_list)))

def _print_unannotated_dataset(X):
    for word_id_list in X:
        word_list = data_process.x_id_list2text_list(word_id_list)
        _get_line = lambda tl: u"\t".join(tl).encode("utf-8")
        print("{0}".format(_get_line(word_list)))


def _test_data_process():
    train_X, train_Y = data_process.get_training_data(TEST_TRAIN_FPATH)
    dev_X, dev_Y = data_process.get_develop_data(TEST_DEV_FPATH)
    test_X = data_process.get_test_data(TEST_TEST_FPATH)
    assert( len(train_X) == len(train_Y) and
            len(dev_X) == len(dev_Y))
    
    assert(train_X == dev_X)
    print(train_X[0])
    print(train_Y[0])
    _print_annotated_dataset(train_X, train_Y)
    _print_unannotated_dataset(test_X)
    
def _test_data_process_serialization():
    # call get_training_data will construct the datadef
    data_process.get_training_data(TEST_TRAIN_FPATH)
    data_process.save("datadef.dump")
    data_process.datadef = None
    data_process.load("datadef.dump")
    dev_X, dev_Y = data_process.get_develop_data(TEST_DEV_FPATH)
    _print_annotated_dataset(dev_X, dev_Y)

def _test_batch_data():
    train_X, train_Y = data_process.get_training_data(TEST_TRAIN_FPATH)
    dev_X, dev_Y = data_process.get_develop_data(TEST_DEV_FPATH)
    test_X = data_process.get_test_data(TEST_TEST_FPATH)
    datadef = data_process.get_default_datadef()
    rng = random.Random(1234)
    train_ge_param = {
        "data":  zip(train_X, train_Y),
        "batch_size":  2,
        "x_padding_id": datadef.x_padding_id,
        "y_padding_id": datadef.y_padding_id,
        "rng": rng,
        "use_unk_replace": True,
        "word_cnt_dict": datadef.wordcnt_dict,
        "unk_id": datadef.unk_id,
        "replace_cnt_lower_bound": 1,
        "replace_prob_lower_bound": 0.4
    }

    def cycle_run():
        batch_training_ge = data_batch.batch_training_data_generator(
                **train_ge_param)
        
        for idx, (batch_X, batch_Y, sent_len) in enumerate(batch_training_ge):
            print(idx)
            _print_annotated_dataset(batch_X, batch_Y)
            print(sent_len)
    cycle_run()
    cycle_run()

    devel_ge_param = {
            "data": zip(dev_X, dev_Y),
            "batch_size": 2,
            "x_padding_id": datadef.x_padding_id,
            "y_padding_id": datadef.y_padding_id
    }

    def cycle_run():
        batch_devel_ge = data_batch.batch_develop_data_generator(
                **devel_ge_param)
        for idx, (batch_X, batch_Y, sent_len) in enumerate(batch_devel_ge):
            print(idx)
            _print_annotated_dataset(batch_X, batch_Y)
            print(sent_len)
    cycle_run()
    cycle_run()

    test_ge_param = {
            "data": test_X,
            "batch_size": 1,
            "x_padding_id": datadef.x_padding_id
    }
    def cycle_run():
        batch_test_ge = data_batch.batch_test_data_generator(
                **test_ge_param)
        for idx, (batch_X, sent_len) in enumerate(batch_test_ge):
            print(idx)
            _print_unannotated_dataset(batch_X)
            print(sent_len)

    cycle_run()
    cycle_run()

if __name__ == "__main__":
    #_test_data_process()
    #_test_data_process_serialization()
    _test_batch_data()
