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

def rouge_score(reference, hypothesis, rouge_type : str):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)

    # rouge-1, rouge-2, rouge-l
    # "r", "p", "f"
    #print(scores)
    return scores[0][f"rouge-{rouge_type}"]["r"]