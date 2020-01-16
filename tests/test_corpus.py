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


from code.data_loader import Corpus, tokenize
from functools import partial
import pytest
from scipy import stats

mean = lambda lst: sum(lst) / len(lst)


@pytest.fixture
def tokenizer_():
    return partial(tokenize, remove_stop=True)

@pytest.fixture
def corpus():
    return Corpus(partial(tokenize, remove_stop=True))


def test_summarize(corpus, tokenizer_):
    
    p, r = corpus.summarize("chi2", range(60, 210, 10))
    print(p, r)
    p, r = corpus.summarize("idf", range(60, 210, 10))
    print(p, r)
    p, r = corpus.summarize("gi", range(60, 210, 10))
    print(p, r)
    p, r = corpus.summarize("tf", range(60, 210, 10))
    print(p, r)
    
    return



    item = corpus.items[0]

    def prec_recall(score, filter="rouge-1"):
        rouge_1 = score[0][filter]
        return rouge_1["p"], rouge_1["r"]

    import matplotlib.pyplot as plt
    plt.figure(1) # 创建图表1
    plt.title('Precision/Recall Curve')# give plot a title
    plt.xlabel('Recall')# make axis labels
    plt.ylabel('Precision')

    ps, rs = [], []
    for desired in range(60, 210, 10):
        score = item.summarize("idf", tokenizer_, desired)
        p, r = prec_recall(score)
        ps.append(p)
        rs.append(r)
    

    plt.plot(rs, ps, color='green', label='training accuracy')
    plt.show()


def test_keywords(corpus):
    corpus.keywords("chi2")
    return
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
