#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   info.py
@Time    :   2019/12/09 11:35:41
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from copy import deepcopy
from functools import partial
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelBinarizer

vectorizer = CountVectorizer(lowercase=False)

def _log(prob, smooth=1e-5):
    prob[prob == 0] = prob[prob == 0] + smooth
    return np.log(prob)


# 摘要与正文之间的比较
# 分类性能越好的词，即显著性越高，那么代表这个词越不重要
def chi2_f(inputs):
    X = vectorizer.fit_transform(inputs.data)
    Y = inputs.target

    chi2_, pval_ = chi2(X, inputs.target)

    return zip(vectorizer.get_feature_names(), chi2_)

def idf_f(inputs):
    idf_vectorizer = deepcopy(vectorizer)
    idf_vectorizer.binary = True
    X = idf_vectorizer.fit_transform(inputs.data)

    # Count every token occurs times per document
    count = X.sum(axis=0)
    idfs = np.log(len(inputs.data) / count).tolist()[0]

    return zip(idf_vectorizer.get_feature_names(), idfs)
    

def tf_f(inputs):
    X = vectorizer.fit_transform(inputs.data)
    count = X.sum(axis=0).tolist()[0]
    
    return zip(vectorizer.get_feature_names(), count)


def gi_f(inputs):
    """Gain Information Calculator
    
    The value of GI is equal to mutual information, 
    but the GI metric requires classification labels.
    """
    X = vectorizer.fit_transform(inputs.data)

    gi_ = mutual_info_classif(X, inputs.target, discrete_features=True)

    return zip(vectorizer.get_feature_names(), gi_)


def llr_f(inputs):
    """Log Likelihood Ratio test

    This test can elimate the count gap between two categories.
    """
    X = vectorizer.fit_transform(inputs.data).toarray()
    Y = np.array(inputs.target)
    Y = LabelBinarizer().fit_transform(Y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    freq = np.dot(Y.T, X)
    N = freq.sum()
    col_prob = freq / np.sum(freq, axis=0)
    row_prob = freq / np.sum(freq, axis=1).reshape(-1, 1)
    row_sum = np.sum(freq, axis=1).reshape(-1, 1)

    log = partial(_log, smooth=1 / N)

    row_entropy = np.sum(freq * log(row_prob) + (row_sum - freq) * log(1 - row_prob), axis=0)
    col_entropy = np.sum(freq * log(col_prob), axis=0) + np.sum((row_sum - freq) * log((row_sum - freq) / np.sum(row_sum - freq, axis=0)), axis=0)
    mat_entropy = np.sum(freq * log(freq / N), axis=0) + np.sum((row_sum - freq) * log((row_sum - freq) / N), axis=0)

    llr_ = - 2 * (mat_entropy - row_entropy - col_entropy)
    return zip(vectorizer.get_feature_names(), llr_)
