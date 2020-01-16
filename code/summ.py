#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summ.py
@Time    :   2020/01/07 21:35:57
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from lexrank import LexRank
from pathlib import Path
from summa.summarizer import summarize as textrank_f


class Summarizer:

    def __init__(self):
        pass

class TextRank(Summarizer):

    def summarize(self, sentences):
        return textrank_f(". ".join(sentences), words=120)

class LexRank(Summarizer):

    def __init__(self):
        documents = [
            file_path.open(mode='rt', encoding='utf-8').readlines()
            for file_path in Path("../bbc/tech").glob("*.txt")
        ]
        self.lxr = LexRank(documents)

    def summarize(self, sentences):
        return self.lxr.get_summary(sentences, summary_size=5)
    
