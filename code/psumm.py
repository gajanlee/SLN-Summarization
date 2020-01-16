#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   summ.py
@Time    :   2020/01/07 10:32:37
@Author  :   gajanlee 
@Version :   1.0
@Contact :   lee_jiazh@163.com
@Desc    :   None
'''

from collections import namedtuple
from copy import deepcopy
from eval_metric import rouge_score
from functools import partial
from lexrank import LexRank
import math
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
from pathlib import Path
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.preprocessing import LabelBinarizer
from string import punctuation
from summa.summarizer import summarize as textrank_f
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.reduction import ReductionSummarizer  # Graph-Rank with neighbors
from sumy.nlp.tokenizers import Tokenizer as STokenizer


STOP_TOKENS = set(Path("/home/lee/workspace/summarization/res/stopwords.txt").read_text().split("\n") + list(punctuation))

RE_SENTENCE = re.compile('(\S.+?[.!?])(?=\s+|$)|(\S.+?)(?=[\n]|$)')

Input = namedtuple("Input", ["data", "target"])
Score = namedtuple("Score", ["content", "score"])
EvalS = namedtuple("EvalS", ["precision", "recall", "fscore"])

def preprocess_section(section, *args, **kwargs):
    return [preprocess_sentence(match.group(), *args, **kwargs) 
        for match in RE_SENTENCE.finditer(section)]

def preprocess_sentence(sentence, remove_stop=True, stem=True):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)

    if remove_stop:
        tokens = filter(lambda tok: tok not in STOP_TOKENS and tok.isalpha(),
                    tokens)
    
    if stem: 
        tokens = map(PorterStemmer().stem, tokens)
    
    return " ".join(tokens)


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
    


class Summarizer:

    def __init__(self):
        self._info_F = InfoF()
    

    def _centroid_sentence_scores(self, sentences, tok_importance):
        return sorted([Score( sentence, 
                            sum(tok_importance.get(tok, 0)
                                for tok in sentence.split(" ")),
                            ) for sentence in sentences],
                    key=lambda s: s.score, 
                    reverse=True)


    def _summ_centroid(self, introduction, section, conclusion, 
                    tok_importance, summary_size):
        return [s.content 
            for s in self._centroid_sentence_scores(
                introduction+section+conclusion, tok_importance)][:summary_size]
    


    def _summ_mmr(self, introduction, section, conclusion,
                    tok_importance, summary_size):
        """
        Reference: Murray, Gabriel & Renals, Steve & Carletta, Jean. (2005). Extractive summarization of meeting recordings. 593-596. 
        """
        
        sentences = introduction + section + conclusion
        summarys = []


        sentence_vector = {}
        for tok in " ".join(sentences).split(" "):
            sentence_vector[tok] = sentence_vector.get(tok, 0) + tok_importance.get(tok, 0)

        eu_dis = math.sqrt(sum(x**2 for x in sentence_vector.values()))
        print(eu_dis)


        similarity_matrix = {}
        for sentence in sentences:
            vector = {}
            for tok in sentence.split(" "):
                vector[tok] = vector.get(tok, 0) + tok_importance.get(tok, 0)
            dis = math.sqrt(sum(x**2 for x in vector.values()))

            multi = sum(vector.get(tok, 0)*sentence_vector.get(tok, 0) for tok in vector.keys())
            sim = multi / (dis * eu_dis)

            similarity_matrix[sentence] = (sim, vector)


        lamda = 0.7
        for i in range(summary_size):

            summary_vector = {}
            for tok in " ".join(summarys).split(" "):
                summary_vector[tok] = summary_vector.get(tok, 0) + tok_importance.get(tok, 0)
            
            s_dis = math.sqrt(sum(x**2 for x in summary_vector.values()))

            sentence_scores = []

            for sentence in sentences:
                sim, vec = similarity_matrix[sentence]

                dis = math.sqrt(sum(x**2 for x in vec.values()))
                multi = sum(vec.get(tok, 0)*summary_vector.get(tok, 0) for tok in vec.keys())

                sim_s = multi / (dis * s_dis)

                score = lamda * sim - (1 - lamda) * sim_s
                if len(summarys) == 0:
                    score = sim

                sentence_scores.append(Score(sentence, score))

            summary = sorted(sentence_scores, key=lambda sc: sc.score, reverse=True)[0].content
            sentences.remove(summary)
            summarys.append(summary)


        
        return summarys
                





    def _summ_centroid_prob(self, introduction, section, conclusion,
                    tok_importance, summary_size):
        sentences = introduction + section + conclusion
        summarys = []

        for i in range(summary_size):
            for sentence_score in self._centroid_sentence_scores(
                sentences, tok_importance):
                sentence = sentence_score.content
                sentences.remove(sentence)

                if sentence in summarys: continue
                summarys.append(sentence)
                for tok in sentence.split(" "):
                    tok_importance[tok] = tok_importance.get(tok, 0) / 2
                break
        return summarys


    def _summ_kl(self, *args, **kwargs):
        return self._summy_funcs(LsaSummarizer, *args, **kwargs)

    def _summ_lsa(self, *args, **kwargs):
        return self._summy_funcs(LsaSummarizer, *args, **kwargs)

    def _summ_luhn(self, *args, **kwargs):
        return self._summy_funcs(LuhnSummarizer, *args, **kwargs)

    def _summ_reduction(self, *args, **kwargs):
        return self._summy_funcs(ReductionSummarizer, *args, **kwargs)

    def _summy_funcs(self, summyCls, introduction, section, conclusion, summary_size):
        sentences = summyCls()(PlaintextParser.from_string(
            ". ".join(introduction + section + conclusion), STokenizer("english")).document, summary_size
        )[:summary_size]
        return " ".join([s._text for s in sentences]).split(". ")

    def summ_item(self, item, info="chi2", method="centroid", summary_size=5):
        summ_f = getattr(self, f"_summ_{method}")
        if not summ_f: 
            raise Exception("can't support [{method}]")

        introduction, section, conclusion = map(preprocess_section, 
            [item.introduction, item.section, item.conclusion])

        if method in ["kl", "lsa", "luhn", "reduction"]:
            return summ_f(introduction, section, conclusion, 
                        summary_size=summary_size)

        info_f = getattr(self._info_F, f"{info}_f")
        if not info_f:
            raise Exception("can't support [{info}]")

        _input = Input(introduction + section + conclusion,
            [1]*len(introduction) + [0]*len(section) + [1]*len(conclusion))
        
        token_importance = info_f(_input)
        # abstract = " ".join(preprocess_section(item.abstract))

        summary = summ_f(introduction, section, conclusion,
                            dict(token_importance), summary_size=summary_size)

        return summary

    def summ_items(self, items, *args, **kwargs):
        """
        items: array of items
        -----
        return a map generator of (abstract_sentences, summary_sentences) tuple
        """
        return map(
            lambda item: (preprocess_section(item.abstract), 
                        self.summ_item(item, *args, **kwargs)),
            items)

    def _eval_rouge(self, abstract_sents, summary_sents):
        rs = rouge_score(" ".join(abstract_sents), " ".join(summary_sents))

        return dict([
            (metric, EvalS(rs[metric]["p"], rs[metric]["r"], rs[metric]["f"]))
            for metric in ("rouge-1", "rouge-2", "rouge-l")
        ])


    def evaluate(self, pairs, metric="rouge"):
        """Evaluate the performance a batch
        pairs: every pair consists of a tuple generated from summ_items
        """
        eval_f = getattr(self, f"_eval_{metric}")
        
        recorder = {}

        for i, pair in enumerate(pairs):
            score = eval_f(*pair)
            if i == 0:
                for key in score.keys():
                    recorder[key] = []
            
            for key, evals in score.items():
                recorder[key].append(evals)

        def mean_evals(evals):
            precision = recall = fscore = 0
            for eval in evals:
                precision += eval.precision
                recall += eval.recall
                fscore += eval.fscore
            
            return EvalS(precision / len(evals), 
                        recall / len(evals),
                        fscore / len(evals))

        for key in recorder.keys():
            recorder[key] = mean_evals(recorder[key])
        
        return recorder
    

class InfoF:

    def __init__(self):
        self.vectorizer = CountVectorizer(lowercase=False)

    #` 摘要与正文之间的比较
    # 分类性能越好的词，即显著性越高，那么代表这个词越不重要
    def chi2_f(self, inputs):
        X = self.vectorizer.fit_transform(inputs.data)
        Y = inputs.target

        chi2_, pval_ = chi2(X, inputs.target)

        return zip(self.vectorizer.get_feature_names(), chi2_)

    def idf_f(self, inputs):
        idf_vectorizer = deepcopy(self.vectorizer)
        idf_vectorizer.binary = True
        X = idf_vectorizer.fit_transform(inputs.data)

        # Count every token occurs times per document
        count = X.sum(axis=0)
        idfs = np.log(len(inputs.data) / count).tolist()[0]

        return zip(idf_vectorizer.get_feature_names(), idfs)
        

    def tf_f(self, inputs):
        X = self.vectorizer.fit_transform(inputs.data)
        count = X.sum(axis=0).tolist()[0]
        
        return zip(self.vectorizer.get_feature_names(), count)


    def gi_f(self, inputs):
        """Gain Information Calculator
        
        The value of GI is equal to mutual information, 
        but the GI metric requires classification labels.
        """
        X = self.vectorizer.fit_transform(inputs.data)

        gi_ = mutual_info_classif(X, inputs.target, discrete_features=True)

        return zip(self.vectorizer.get_feature_names(), gi_)


    def llr_f(self, inputs):
        """Log Likelihood Ratio test

        This test can elimate the count gap between two categories.
        """
        def _log(prob, smooth=1e-5):
            prob[prob == 0] = prob[prob == 0] + smooth
            return np.log(prob)

        X = self.vectorizer.fit_transform(inputs.data).toarray()
        Y = np.array(inputs.target)
        Y = LabelBinarizer().fit_transform(Y)
        if Y.shape[1] == 1:
            Y = np.append(1 - Y, Y, axis=1)

        freq = np.dot(Y.T, X)
        N = freq.sum()
        col_prob = freq / np.sum(freq, axis=0)
        row_prob = freq / np.sum(freq, axis=1).reshape(-1, 1)
        row_sum = np.sum(freq, axis=1).reshape(-1, 1)

        _log = partial(_log, smooth=1 / N)

        row_entropy = np.sum(freq * _log(row_prob) + (row_sum - freq) * _log(1 - row_prob), axis=0)
        col_entropy = np.sum(freq * _log(col_prob), axis=0) + np.sum((row_sum - freq) * _log((row_sum - freq) / np.sum(row_sum - freq, axis=0)), axis=0)
        mat_entropy = np.sum(freq * _log(freq / N), axis=0) + np.sum((row_sum - freq) * _log((row_sum - freq) / N), axis=0)

        llr_ = - 2 * (mat_entropy - row_entropy - col_entropy)
        return zip(self.vectorizer.get_feature_names(), llr_)


    def random_f(self, inputs):
        """
        The weights is assigned random.
        """

        self.vectorizer.fit(inputs.data)
        weights_ = np.random.random(len(self.vectorizer.get_feature_names()))

        return zip(self.vectorizer.get_feature_names(), weights_)
