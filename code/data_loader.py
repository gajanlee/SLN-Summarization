#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dataloader.py
@Time    :   2019/12/10 14:11:52
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from .info import chi_square_f, gi_f, idf_f, tf_f

from collections import Counter, namedtuple
from functools import partial
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from pathlib import Path
from string import punctuation
import re

BASE_PATH = Path("/home/lee/workspace/RST_summary/data/acl2014_2")
ABSTRACT_PATH = BASE_PATH / "abstract"
CONTENT_PATH = BASE_PATH / "content"
RST_PATH = BASE_PATH / "summary"

RE_SENTENCE = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)')

#STOP_TOKENS = set(stopwords.words("english") + list(punctuation))
STOP_TOKENS = set(Path("/home/lee/workspace/summarization/res/stopwords.txt").read_text().split("\n") + list(punctuation))

Input = namedtuple("Input", ["data", "target"])
round = partial(round, ndigits=4)


def tokenize(text, level="sentence"):
    if level == "sentence":
        return [match.group() for match in RE_SENTENCE.finditer(text)]

def metric_factory(metric):
    mapper = {
        "chi2": chi_square_f,
        "gi": gi_f,
        "idf": idf_f,
        "tf": tf_f,
    }
    return mapper.get(metric, None)

class CorpusItem:

    def __init__(self, filename):
        self.filename = filename
        self.abstract = (ABSTRACT_PATH / filename).read_text().lower()
        self.title, self.introduction, *self.sections, self.conclusion = (CONTENT_PATH / filename).read_text().lower().split("\n")
        self.sections = " ".join(self.sections)


    def transform(self):
        """Transform to Input tuple
        按照一定形式组织输入，如短语级、句子级、段落级等，
        标签使用启发式方法，引言与结论作为1，细节描述作为0,。
        """
        intro_units = tokenize(self.introduction)
        sec_units = tokenize(self.sections)
        conc_units = tokenize(self.conclusion)

        data = intro_units + sec_units + conc_units
        target = [1] * len(intro_units) + [0] * len(sec_units) + [1] * len(conc_units)

        return Input(data, target)


    def abstract_distribution(self):
        word_tokenize_stem = lambda text: list(map(PorterStemmer().stem, word_tokenize(text)))

        filter_tokenize = lambda text: list(
            filter(lambda tok: tok not in STOP_TOKENS, word_tokenize_stem(text)))

        abstract_toks = filter_tokenize(self.abstract)

        dists = []
        for dist in map(filter_tokenize, 
            [self.introduction, self.sections, self.conclusion]):
            uniq_count = len(set(dist) & set(abstract_toks))
            dup_count = sum(dist.count(tok) for tok in set(abstract_toks))
            portion = dup_count / len(dist) if len(dist) else 0
            dists.append((uniq_count, dup_count, portion))
        return len(set(abstract_toks)), dists


class Corpus:

    def __init__(self):
        self.items = [CorpusItem(filename.name) for filename in CONTENT_PATH.glob("*.txt")]

    def summarize(self, metric):
        pass

    def keywords(self, metric):
        func = metric_factory(metric)
        for i, item in enumerate(self.items):
            input = item.transform()
            tok_val_list = func(input)
            tok_val_list = sorted(tok_val_list, key=lambda tok_val: tok_val[1], reverse=True)

            print(tok_val_list[:10])
            print(tok_val_list[-10:])
            print(item.title)
            break
