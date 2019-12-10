#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_info.py
@Time    :   2019/12/10 14:09:25
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from codes.data_loader import Input
from codes.info import *

test_input = Input(
    ["Yes, you are right",
    "do you like me",
    "yes it is good"],
    [1, 0, 1]
)

def test_chi_square():
    chi2_ = chi_square_f(test_input)
    assert chi2_["like"] > chi2_["yes"] > chi2_["good"] == chi2_["right"]

def test_idf():
    idf_ = idf_f(test_input)
    assert idf_["good"] == idf_["right"] == idf_["right"] > idf_["yes"]

def test_tf():
    tf_ = tf_f(test_input)
    assert tf_["yes"] > tf_["right"] == tf_["good"] == tf_["like"]

def test_gi():
    gi_ = gi_f(test_input)
    
    assert gi_["like"] == gi_["yes"] > gi_["good"] == gi_["right"]
