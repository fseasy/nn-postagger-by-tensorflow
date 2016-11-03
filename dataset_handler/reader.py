#!/usr/bin/env python3
# -*- encoding:utf-8 -*-

import os
import gzip

_data_dir = "../data_from_3rdparty/scir_training_day/3-nlp-practice"

_train_data_path = os.path.join(_data_dir, "penn.train.pos.gz")
_devel_data_path = os.path.join(_data_dir, "penn.devel.pos.gz")
_test_data_path = os.path.join(_data_dir, "penn.test.pos.blind.gz")

def read_annotated_data(fpath):
    with gzip.open(fpath, 'r', encoding="utf8") as f:
        annotated_dataset = []
        for line in f:
            line = line.strip()
            word_tag_pair_list = line.split()
            instance = [ [ pair[0] for pair in word_tag_pair_list  ], [ pair[1] for pair in word_tag_pair_list ]  ]
            annotated_dataset.append(instance)
        return annotated_dataset

def read_training_data():
    return read_annotated_data(_train_data_path)

def read_devel_data():
    return read_annotated_data(_devel_data_path)

def read_
