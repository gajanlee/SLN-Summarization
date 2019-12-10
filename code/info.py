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
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif

vectorizer = CountVectorizer(lowercase=True,stop_words='english')


# 摘要与正文之间的比较
# 分类性能越好的词，即显著性越高，那么代表这个词越不重要
def chi_square_f(inputs):
    X = vectorizer.fit_transform(inputs.data)
    Y = inputs.target

    chi2_, pval_ = chi2(X, inputs.target)

    return zip(vectorizer.get_feature_names(), chi2_)


def idf_f(inputs):
    new_vectorizer = deepcopy(vectorizer)
    new_vectorizer.binary = True
    X = new_vectorizer.fit_transform(inputs.data)
    vocab = new_vectorizer.get_feature_names()

    # Count every token occurs times per document
    count = X.sum(axis=0)
    idfs = np.log(len(inputs.data) / count).tolist()[0]

    return zip(vectorizer.get_feature_names(), idfs)
    

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

