#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_corpus.py
@Time    :   2019/12/10 17:17:01
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''


from code.data_loader import Corpus
import pytest
from scipy import stats

mean = lambda lst: sum(lst) / len(lst)

@pytest.fixture
def corpus():
    return Corpus()


def test_keywords(corpus):
    corpus.keywords("chi2")
    corpus.keywords("tf")
    corpus.keywords("gi")
    corpus.keywords("idf")


def test_distribution(corpus):
    s = [[], [], []]
    for item in corpus.items:
        record = item.abstract_distribution()
        for i in range(3):
            #if record[1][i][2]!=0:
            s[i].append(record[1][i][2])
    print(list(map(mean, s)))
    t2, p2 = stats.ttest_ind(s[0], s[1])
    print(t2, " ", p2)
    t2, p2 = stats.ttest_ind(s[2], s[1])
    print(t2, " ", p2)

    return 
