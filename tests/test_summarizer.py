#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_summarizer.py
@Time    :   2020/01/07 11:09:19
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from code.summ import *
from code.corpus import ACL2014
from pathlib import Path

def test_summ():
    corpus = ACL2014(Path("/home/lee/workspace/RST_summary/data/acl2014_2"))
    summ = Summarizer()

    for item in corpus.items_generator():
        s = summ.summ_single(item)
        print(s)
        print(list(s))
        break