#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   eval.py
@Time    :   2020/01/07 12:05:17
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from rouge import Rouge

def rouge_score(reference, hypothesis):
    """
    reference: truth
    hypothesis: summary

    return: {
        "rouge-1": {"p": 1.0, "r": 1.0, "f": s}
        "rouge-2": ...,
        "rouge-l": ...,
    }
    """

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]

def pymaid():
    pass

def fresch():
    pass



def evaluation(truth, out_sentences):
    Score = namedtuple(precision, recall, fscore)


    {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-3": [],
    }
    
    precisions, recalls, fs = [], [], []
    
    for summary in reduce(lambda pre_s, s: f"{pre_s} {s}", out_sentences):
        rouge_score(truth, summary)
        




