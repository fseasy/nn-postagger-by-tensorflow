#!/usr/bin/env python2
# -*- encoding:utf-8 -*-

import os
import gzip

_cur_dir_path = os.path.split(os.path.realpath(__file__))[0]
_data_dir = os.path.join(_cur_dir_path, "../data_from_3rdparty/scir_training_day/3-nlp-practice")

_train_data_path = os.path.join(_data_dir, "penn.train.pos.gz")
_devel_data_path = os.path.join(_data_dir, "penn.devel.pos.gz")
_test_data_path = os.path.join(_data_dir, "penn.test.pos.blind.gz")

def _read_annotated_data(fpath):
    '''read annotated data. (from format 'word/tag ...)' => ( word_list, tag_list )'''
    with gzip.open(fpath, 'rt') as f:
        annotated_dataset = []
        for line in f:
            line = line.decode("utf-8").strip()
            word_tag_pair_list = line.split()
            instance = ([], [])
            for word_tag_pair in word_tag_pair_list:
                (word, tag) = word_tag_pair.rsplit('/', 1) # may have multiple '/'
                instance[0].append(word)
                instance[1].append(tag)
            annotated_dataset.append(instance)
        return annotated_dataset

def _read_unannotated_data(fpath):
    '''read unannotated data. (from format 'word/_ ...') => word_list '''
    with gzip.open(fpath, 'rt') as f:
        unannotated_dataset = []
        for line in f:
            line = line.decode("utf-8").strip()
            word_tag_pair_list = line.split()
            instance = [ pair.rsplit('/', 1)[0] for pair in word_tag_pair_list ]
            unannotated_dataset.append(instance)
        return unannotated_dataset


def read_training_data():
    return _read_annotated_data(_train_data_path)

def read_devel_data():
    return _read_annotated_data(_devel_data_path)

def read_test_data():
    return _read_unannotated_data(_test_data_path)
