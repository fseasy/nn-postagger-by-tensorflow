#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO

class _TagCnt(object):
    def __init__(self):
        self.tp_cnt = 0
        self.gold_cnt = 0
        self.pred_cnt = 0

def calc_prf(tag_cnt):
    '''
    calc p, r, f for _TagCnt project.
    Args:
        tag_cnt: Tag Cnt.
    Returns:
        [p, r, f, tp_cnt, gold_cnt, pred_cnt], 
        p, r, f is percent representation( XX% )
    '''
    tp_cnt = tag_cnt.tp_cnt
    gold_cnt = tag_cnt.gold_cnt
    pred_cnt = tag_cnt.pred_cnt

    if pred_cnt != 0:
        p = tp_cnt / pred_cnt
    else:
        if tp_cnt == 0:
            p = 1.
        else:
            p = 0.
    if gold_cnt != 0:
        r = tp_cnt / gold_cnt
    else:
        r = float(tp_cnt == 0)
    if p + r == 0.:
        f = float(gold_cnt + pred_cnt == 0)
    else:
        f = 2 * p * r / (p + r)
    prf = [ float_v * 100  for float_v in [p, r, f]]
    return prf + [tp_cnt, gold_cnt, pred_cnt]

class TagScore(object):
    def __init__(self, tag_id2str=None):
        '''
        init data.
        '''
        self._tag_cnt = dict()
        if not tag_id2str:
            id2str = lambda x: x
        else:
            id2str = lambda x: tag_id2str[x]
        self._id2str = id2str

    def partial_statistic(self, batch_gold, batch_pred, sent_len_list):
        '''
        patial statistic.
        only update the inner cnt structure
        Args:
            batch_gold: batch gold tag sequence.
            batch_pred: batch pred tag sequence.
            sent_len_list: length for every sentence in batch.
        '''
        batch_size = len(batch_gold)
        assert(batch_size == len(batch_pred) == len(sent_len_list))
        for i in range(batch_size):
            gold_seq = batch_gold[i]
            pred_seq = batch_pred[i]
            cur_len = sent_len_list[i]
            tag_idx = 0
            while tag_idx < cur_len:
                gold_tag = gold_seq[tag_idx]
                pred_tag = pred_seq[tag_idx]
                tag_cnt4gold = self._tag_cnt.setdefault(gold_tag, _TagCnt())
                tag_cnt4pred = self._tag_cnt.setdefault(pred_tag, _TagCnt())
                tag_cnt4gold.gold_cnt += 1
                tag_cnt4pred.pred_cnt += 1
                if gold_tag == pred_tag:
                    tag_cnt4gold.tp_cnt += 1
                tag_idx += 1

    def get_statistic_result(self):
        '''
        calc p, r, f for every tag.
        Returns:
            A dict, key=tag, value=[p, r, f, tp_cnt, gold_cnt, pred_cnt], prf is percent representation
        '''
        result = {}
        for tag_id, cnt_obj in self._tag_cnt.items():
            prf = calc_prf(cnt_obj)
            tag = self._id2str(tag_id)
            result[tag] = prf
        return result

    def get_statistic_result_str(self, result=None, encoding="utf-8"):
        '''
        result string for every tag p, r, f.
        Returns:
            str, result string
        '''
        if not result:
            result = self.get_statistic_result()
        mem_of = StringIO()
        mem_of.write(u"----- Statistic Info -----\n".encode(encoding))
        for tag, [p, r, f, tp_cnt, gold_cnt, pred_cnt] in sorted(result.items()):
            u = ("{tag}: p: {p:.2f}({tp_cnt}/{pred_cnt})%,"
                 " r: {r:.2f}({tp_cnt}/{gold_cnt})% f: {f:.2f}%\n").format(**locals())
            s = u.encode(encoding)
            mem_of.write(s)
        return mem_of.getvalue()


def __unittest():
    tag_score = TagScore()
    batch_prd = [ [1,2,2,0,0,0],
                  [2,1,0,0,0,0]]
    batch_gold = [ [2,2,2,0,0,0],
                   [1,1,0,0,0,0]]
    sent_len = [3, 2]
    # 1: gold_cnt = 2, pred_cnt= 2, tp_cnt=1, p=1/2=0.5, r=1/2=0.5, f1=0.5
    # 2: gold_cnt = 3, pred_cnt= 3, tp_cnt=2, p=2/3=0.67, r=2/3=0.67, f1=0.67
    tag_score.partial_statistic(batch_gold, batch_prd, sent_len)
    result_str = tag_score.get_statistic_result_str()
    print(result_str)

if __name__ == "__main__":
    __unittest()
