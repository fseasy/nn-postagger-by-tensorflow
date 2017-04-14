#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
data processing.
build dict, translate word to id.
ready batch data.
'''

from collections import Counter
try:
    import cPickle as pickle
except:
    import pickle

class DataDef(object):
    '''
    Data definition, including word2id, id2word, word_cnt, seed
    '''
    _PADDING_ID = 0
    _SOS_ID = 1
    _EOS_ID = 2
    _UNK_ID = 3
    _PADDING_WORD_REPR = u"_PADDING_"
    _SOS_WORD_REPR = u"_SOS_"
    _EOS_WORD_REPR = u"_EOS_"
    _UNK_WORD_REPR = u"_UNK_"

    _PADDING_TAG_ID = 0
    _PADDING_TAG_REPR = u"X"
    def __init__(self, seed=1234):
        inf = float("inf")
        self._word2id = {
                    self._PADDING_WORD_REPR: self._PADDING_ID,
                    self._SOS_WORD_REPR: self._SOS_ID,
                    self._EOS_WORD_REPR: self._EOS_ID,
                    self._UNK_WORD_REPR: self._UNK_ID
        }
        self._id2word = [
                        self._PADDING_WORD_REPR, 
                        self._SOS_WORD_REPR, 
                        self._EOS_WORD_REPR,
                        self._UNK_WORD_REPR
        ]
        self._word_cnt = [inf, inf, inf, inf]
        self._tag2id = {
                    self._PADDING_TAG_REPR: self._PADDING_TAG_ID 
        }
        self._id2tag = [self._PADDING_TAG_REPR]
        self._seed = seed

    @property
    def word2id(self):
        return self._word2id

    @property
    def id2word(self):
        return self._id2word

    @property
    def wordcnt_dict(self):
        return self._word_cnt

    @property
    def tag2id(self):
        return self._tag2id

    @property
    def id2tag(self):
        return self._id2tag

    def get_wordid(self, word):
        return self._word2id.get(word, self._UNK_ID)

    def get_tagid(self, tag):
        if tag not in self._tag2id:
            raise ValueError("unkown tag: {0}".format(tag.encode("utf-8")))
        return self._tag2id[tag]
    
    @property
    def word_num(self):
        return len(self._id2word)
    
    @property
    def tag_num(self):
        return len(self._id2tag)

    def get_wordtext(self, wid):
        '''
        Raises:
            IndexError
        '''
        if wid < 0 or wid >= self.word_num:
            raise IndexError("wid: {0} not in valid range: {1}-{2}".format(wid, 0, self.word_num))
        return self._id2word[wid]

    def get_tagtext(self, tid):
        '''
        Raises:
            IndexError
        '''
        if tid < 0 or tid >= self.tag_num:
            raise IndexError("tid: {0} not in valid range: {1}-{2}".format(tid, 0, self.tag_num))
        return self._id2tag[tid]

    @property
    def x_padding_id(self):
        return self._PADDING_ID
    @property
    def y_padding_id(self):
        return self._PADDING_TAG_ID
    @property
    def unk_id(self):
        return self._UNK_ID

# default data definition
datadef = DataDef()

def get_default_datadef():
    '''
    get the default datadef.
    Returns:
        datadef.
    '''
    return datadef

def annotated_data2word_tag_list(word_line, tag_line, delimiter=u"\t"):
    '''split annotated line to word, tag list
    Args:
        word_line: unicode line, seperated by delimiter
        tag_line: unicode line.
        delimiter: delimiter.
    Returns:
        A tuple (word_list, tag_list)
    '''
    word_list = word_line.split(delimiter)
    tag_list = tag_line.split(delimiter)
    return (word_list, tag_list)

def unannotated_data2word_list(word_line, delimiter=u"\t"):
    '''
    split unannotated line to word list.
    Args:
        word_line: unicode line, seperated by delimiter.
        delimietr: delimeiter
    Returns:
        word_list
    '''
    return word_line.split(delimiter)

def build_dict(X_text, Y_text, datadef=datadef):
    '''
    build dict info according to the training text data.
    Args:
        X_text: [ [w1, w2, ..], ...], sentence list, where sentence 
                is the list of word
        Y_text: [ [t1, t2, ...], ..]
        datadef: datadef, optional, using default when no new passing
    Returns:
        datadef
    '''
    word_count = Counter()
    y_set = set()
    for x, y in zip(X_text, Y_text):
        word_count.update(x)
        y_set.update(y)
    word_cnt_decreasing_pair_list = word_count.most_common()
    for word, cnt in word_cnt_decreasing_pair_list:
        idx = len(datadef.id2word)
        datadef.id2word.append(word)
        datadef.word2id[word] = idx
        datadef.wordcnt_dict.append(cnt)
    for y in y_set:
        idx = len(datadef.id2tag)
        datadef.id2tag.append(y)
        datadef.tag2id[y] = idx
    return datadef

def annotated_text_sample2id_sample(x_text, y_text, datadef=datadef):
    '''
    [w1, w2] => [id1, id2, ...], [t1, t2] => [tid1, tid2, ...]
    Args:
        x_text: list of word text.
        y_text: list to tag list.
        datadef: data definition
    Returns:
        A tuple, (x_id_list, y_id_list)
    '''
    x_id_list = [datadef.get_wordid(word) for word in x_text]
    y_id_list = [datadef.get_tagid(tag) for tag in y_text]
    return (x_id_list, y_id_list)

def unannotated_text_sample2id_sample(x_text, datadef=datadef):
    ''' [w1, w2, ..] => [id1, id2, ...]
    Args:
        x_text: list of word text
        datadef: data definition
    Returns:
        A list, x_id_list
    '''
    return [ datadef.get_wordid(word) for word in x_text] 

def _read_annotated_text_sample_generator(fpath, encoding):
    '''
    a generator to read x(word list), and y(tag list) form fpath.
    Args:
        fpath: annotated data file path
        encoding: file encoding
    Returns:
        A tuple, (word_list, tag_list)
    '''
    with open(fpath) as f:
        while True:
            x_text = f.readline()
            y_text = f.readline()
            if x_text == "" or y_text == "":
                break
            x_text = x_text.decode(encoding).strip()
            y_text = y_text.decode(encoding).strip()
            word_list, tag_list = annotated_data2word_tag_list(
                    x_text, y_text)
            yield (word_list, tag_list)

def _read_unannotated_text_sample_generator(fpath, encoding):
    '''
    a generator to read x(word list) from file.
    Args:
        fpath: unnotated data file path
        encoding: file encoding
    Returns:
        word_list
    '''
    with open(fpath) as f:
        for line in f:
            line = line.decode(encoding).strip()
            word_list = unannotated_data2word_list(line)
            yield word_list

def get_training_data(training_fpath, encoding="utf-8", datadef=datadef):
    '''
    read training data file and generate training data( list of id list).
    Args:
        training_fpath: training data file path.
        encoding: file encoding.
        datadef: datadef
    Returns:
        A tuple, (X, Y)
        - X, list of x, x = [id1, id2]
        - Y, list of y, y = [tid1, tid2]
    '''
    X_text = []
    Y_text = []
    sample_ge = _read_annotated_text_sample_generator(training_fpath, encoding)
    for word_list, tag_list in sample_ge:
            X_text.append(word_list)
            Y_text.append(tag_list)
    build_dict(X_text, Y_text, datadef)
    X = []
    Y = []
    for x_text, y_text in zip(X_text, Y_text):
        x, y = annotated_text_sample2id_sample(x_text, y_text, datadef)
        X.append(x)
        Y.append(y)
    return X, Y

def get_develop_data(dev_fpath, encoding="utf-8", datadef=datadef):
    '''
    read dev data file and generate dev data( list of id list).
    Args:
        dev_fpath: dev data file path.
        encoding: file encoding.
        datadef: datadef
    Returns:
        A tuple, (X, Y)
        - X, list of x, x = [id1, id2]
        - Y, list of y, y = [tid1, tid2]
    '''
    X = []
    Y = []
    dev_sample_ge = _read_annotated_text_sample_generator(dev_fpath, encoding)
    for word_list, tag_list in dev_sample_ge:
        x, y = annotated_text_sample2id_sample(word_list, tag_list, datadef)
        X.append(x)
        Y.append(y)
    return (X, Y)

def get_test_data(test_fpath, encoding="utf-8", datadef=datadef):
    '''
    read test data file and generate test data(list of id list)
    Args:
        test_fpath: test data file path
        encoding: file encoding
        datadef: datadef
    Returns:
        X, list of id list, [ [id1, id2, ...], ...]
    '''
    X = []
    test_sample_ge = _read_unannotated_text_sample_generator(test_fpath, encoding)
    for word_list in test_sample_ge:
        X.append(unannotated_text_sample2id_sample(word_list, datadef))
    return X

################
# save & load data definition
################

def save(f, datadef=datadef):
    '''
    save data definition.
    Args:
        f: file path to store the datadef or file obj
        datadef: datadef
    '''
    if isinstance(f, file):
        pickle.dump(datadef, f)
    else:
        with open(f, "wb") as of:
            pickle.dump(datadef, of)

def load(f, use_default=True):
    '''
    load data definition
    Args:
        f: file path to load datadef or file obj
        use_default: is True, set the inner datadef object
            when load done. else not.

    Returns:
        DataDef object.
    '''
    global datadef
    if isinstance(f, file):
        local_datadef = pickle.load(f)
    else:
        with open(f) as pf:
            local_datadef = pickle.load(pf)
    if use_default:
        datadef = local_datadef
    return local_datadef

#########
# 4 debug
#########

def x_id_list2text_list(word_id_list, datadef=datadef):
    '''
    [id1, id2, ..] => [w1, w2]
    '''
    return [ datadef.get_wordtext(wid) for wid in word_id_list]

def y_id_list2text_list(tag_id_list, datadef=datadef):
    '''
    [tid1, tid2, ..] => [tag1, tag2]
    '''
    return [datadef.get_tagtext(tid) for tid in tag_id_list]
