#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   metrics.py
@Time    :   2019/12/10 14:04:19
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from rouge import Rouge

def rouge_score(reference, hypothesis, rouge_type=None):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]