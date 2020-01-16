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

from .info import chi2_f, gi_f, idf_f, llr_f, tf_f
from .metrics import rouge_score

from collections import Counter, namedtuple
from functools import partial
from itertools import chain
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
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
word_tokenize_stem = lambda text: list(map(PorterStemmer().stem, word_tokenize(text)))


def segment(text, level="sentence"):
    if level == "sentence":
        return [match.group() for match in RE_SENTENCE.finditer(text)]

def tokenize(text, join=True, stem=False, remove_stop=False):
    tokenizer_ = word_tokenize_stem if stem else word_tokenize
    tokens = filter(lambda tok: tok not in STOP_TOKENS, tokenizer_(text)) if remove_stop else tokenizer_(text)
    return " ".join(tokens) if join else list(tokens)

def metric_factory(metric):
    mapper = {
        "chi2": chi2_f,
        "gi": gi_f,
        "idf": idf_f,
        "tf": tf_f,
        "llr": llr_f,
    }
    return mapper.get(metric, None)

class CorpusItem:

    def __init__(self, filename, tokenizer):
        """
        tokenizer: 分词器，所有参数内置
        """
        self.filename = filename
        self.abstract = (ABSTRACT_PATH / filename).read_text().lower()
        self.title, self.introduction, *self.sections, self.conclusion = (CONTENT_PATH / filename).read_text().lower().split("\n")
        self.sections = " ".join(self.sections)

        self._keywords_cache = {}
        self.tokenizer_ = tokenizer

    def transform(self):
        """Transform to Input tuple
        按照一定形式组织输入，如短语级、句子级、段落级等，
        标签使用启发式方法，引言与结论作为1，细节描述作为0。
        """
        segment_units = lambda text: list(map(self.tokenizer_, segment(text)))

        intro_units = segment_units(self.introduction)
        sec_units = segment_units(self.sections)
        conc_units = segment_units(self.conclusion)

        data = intro_units + sec_units + conc_units
        target = [1] * len(intro_units) + [0] * len(sec_units) + [1] * len(conc_units)

        return Input(data, target)

    def abstract_distribution(self):
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

    def summarize(self, metric):
        tok_val_dict = self.keywords(metric)
        
        sentences = segment(self.introduction + self.conclusion)
        sent_toks = [self.tokenizer_(sent, join=False) for sent in sentences]
        #weights = [sum(tok_val_dict.get(tok, 0) for tok in sent) / len(sent) for sent in sent_toks]

        weights = []
        for toks in sent_toks:
            toks = list(filter(lambda tok: tok in tok_val_dict, toks))
            weight = sum(tok_val_dict[tok] for tok in toks) / len(toks) if len(toks) != 0 else 0
            weights.append(-weight)
        

        #sort = np.random.permutation(len(weights))
        #print(len(sentences))
        sort = np.argsort(weights)

        return [sent_toks[i] for i in sort]

        
        " ".join([" ".join(sent_toks[i]) for i in np.argsort(weights[::-1])])

        weights = [sum(tok_val_dict.get(tok, 0) for tok in sent) / len(sent) for sent in sent_toks]
        choices = []
        for i in np.argsort(weights[::-1]):
            if len(choices) >= desired_length:
                break
            choices += sent_toks[i]
        summary = " ".join(choices)
        s = rouge_score(tokenizer_(self.abstract), summary)
        return s

    def summarize_wrapper(self, metric, desired_lengths):
        sent_toks = self.summarize(metric)
        truth = self.tokenizer_(self.abstract)

        scores = {}
        for desired in desired_lengths:
            choices = []
            for toks in sent_toks:
                if len(choices) >= desired:
                    break
                choices += toks
            summary = " ".join(choices)
            scores[desired] = rouge_score(truth, summary)
        return scores


    def keywords(self, metric):
        if metric in self._keywords_cache:
            return self._keywords_cache[metric]
        metric_f = metric_factory(metric)
        
        inputs = self.transform()
        tok_val_dict = dict(metric_f(inputs))

        keys = set(self.tokenizer_(self.introduction + self.abstract, join=False))
        for key in tok_val_dict.keys():
            #if key not in keys:
            #    tok_val_dict[key] /= 4
            #if key.isdigit():
            #    tok_val_dict[key] = 0
            pass

        self._keywords_cache[metric] = tok_val_dict
        return tok_val_dict


class Corpus:

    def __init__(self, tokenizer):
        self.items = [CorpusItem(filename.name, tokenizer) for filename in CONTENT_PATH.glob("*.txt")]

    def summarize(self, metric, desired_lengths, pr=True):
        desired_lengths = list(desired_lengths)[:]
        scores = [item.summarize_wrapper(metric, desired_lengths) for item in self.items[:]]

        precs, recalls = {}, {}
        for score in scores:
            for desired in desired_lengths:
                pr = score[desired][0]["rouge-1"]
                p, r = pr["p"], pr["r"]
                precs[desired] = precs.get(desired, []) + [p]
                recalls[desired] = recalls.get(desired, []) + [p]
        mean = lambda lst: sum(lst) / len(lst)

        return list(map(mean, precs.values())), list(map(mean, recalls.values()))


    def keywords(self, metric):
        tokenizer_ = partial(tokenize, remove_stop=False)

        for i, item in enumerate(self.items):
            item.keywords(metric, tokenizer_)

            # counter = Counter(word_tokenize(item.introduction + item.sections + item.conclusion))
            # tokens = set(tok for tok, cnt in counter.items() if cnt >= 1)
            
