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
import nltk
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


STOP_TOKENS = set(Path("/home/lee/workspace/projects/SLN-Summarization/res/stopwords.txt").read_text().split("\n") + list(punctuation))

RE_SENTENCE = re.compile('(\S.+?[.!?`])(?=\s+|$)|(\S.+?)(?=[\n]|$)')

Input = namedtuple("Input", ["data", "target"])
Score = namedtuple("Score", ["content", "score"])
EvalS = namedtuple("EvalS", ["precision", "recall", "fscore"])

def preprocess_section(section, *args, **kwargs):
    return [preprocess_sentence(match.group(), *args, **kwargs) 
        for match in RE_SENTENCE.finditer(section.replace(".", ". "))]

def preprocess_sentence(sentence, remove_stop=False, stem=False):
    sentence = sentence.lower()
    sentence = sentence.replace("\n", "").replace("`", "").replace("'", "").strip()
    tokens = word_tokenize(sentence)

    if remove_stop:
        tokens = filter(lambda tok: tok not in STOP_TOKENS and tok.isalpha(),
                    tokens)
    
    if stem: 
        tokens = map(PorterStemmer().stem, tokens)
    
    return " ".join(tokens)

class Summarizer:

    def __init__(self):
        self._info_F = InfoF()

        documents = [
            file_path.open(mode='rt', encoding='utf-8').readlines()
            for file_path in Path("/media/lee/辽东铁骑/codes/summarization/bbc/tech").glob("*.txt")
        ]
        self.lxr = LexRank(documents)

    def summarize(self, sentences):
        return self.lxr.get_summary(sentences, summary_size=5)
    

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


    def _summ_textrank(self, sentences, summary_size=5):
        return textrank_f(sentences, words=summary_size*40)

    def _summ_lexrank(self, sentences, summary_size=5):
        return self.lxr.get_summary(sentences.split(". "), summary_size)


    def _summ_kl(self, *args, **kwargs):
        return self._summy_funcs(KLSummarizer, *args, **kwargs)

    def _summ_lsa(self, *args, **kwargs):
        return self._summy_funcs(LsaSummarizer, *args, **kwargs)

    def _summ_luhn(self, *args, **kwargs):
        return self._summy_funcs(LuhnSummarizer, *args, **kwargs)

    def _summ_reduction(self, *args, **kwargs):
        return self._summy_funcs(ReductionSummarizer, *args, **kwargs)

    def _summy_funcs(self, summyCls, text, summary_size):
        sentences = summyCls()(PlaintextParser.from_string(
            text, STokenizer("english")).document, summary_size
        )
        return [s._text for s in sentences]
        return " ".join([s._text for s in sentences]).split(". ")


    def summarize_item(self, item, method="kl", summary_size=5):
        summ_f = getattr(self, f"_summ_{method}")
        if not summ_f: 
            raise Exception("can't support [{method}]")

        if method in ["kl", "lsa", "luhn", "reduction", "textrank", "lexrank"]:
            
            text = " ".join(preprocess_section(item.introduction + item.section + item.conclusion))
            
            return summ_f(text, summary_size), None



    def summ_item(self, item, info="chi2", method="centroid", summary_size=5):
        summ_f = getattr(self, f"_summ_{method}")
        if not summ_f: 
            raise Exception("can't support [{method}]")

        introduction, section, conclusion = map(preprocess_section, 
            [item.introduction, item.section, item.conclusion])

        if method in ["kl", "lsa", "luhn", "reduction"]:
            return summ_f(" ".join(introduction + section + conclusion), 
                        summary_size=summary_size), None

        info_f = getattr(self._info_F, f"{info}_f")
        if not info_f:
            raise Exception("can't support [{info}]")

        _input = Input(introduction + section + conclusion,
            [1]*len(introduction) + [0]*len(section) + [1]*len(conclusion))
        
        token_importance = dict(info_f(_input))
        # abstract = " ".join(preprocess_section(item.abstract))

        summary = summ_f(introduction, section, conclusion,
                            token_importance, summary_size=summary_size)

        return summary, token_importance

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
tokens = informative_tokens + detailed_tokens
    all_counter = Counter(all_tokens)
    informative_counter = Counter(informative_tokens)
    detailed_counter = Counter(informative_tokens)

    def _smi(token):
        p_x_yi = informative_counter.get(token, 0) / len(informative_tokens)
        p_x_yd = detailed_counter.get(token, 0) / len(detailed_tokens)

        p_x = all_counter.get(token) / len(all_tokens)
        p_yi = len(informative_tokens) / len(all_tokens)
        p_yd = len(detailed_tokens) / len(all_tokens)

        return p_x_yi * log(p_x_yi / (p_x * p_yi)) - lamda * p_x_yd * log(p_x_yd / (p_x * p_yd))

    return {
        token: _smi(token) for token in all_counter
    }
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

        # TODO: 越高的独立值，互相依赖性越高，试试负号
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

    



class SLNSummarize:
    def __init__(self):
        self.define_link()


    def define_link(self):
        LINK = namedtuple("LINK", ["name", "clue_words"])

        TEMPORAL = LINK("TEMPORAL_LINK", ["before", "after", "overlap", "when", "during"])
        CAUSE_EFFECT = LINK("CE_LINK", ["because", "since", "so", "lead to", "therefore", "thus", "hence"])
        PURPORSE = LINK("PURPOSE_LINK", ["for", "to", "in order to", "so that"])
        MEANS = LINK("MEANS_LINK", ["by", "through"])
        CONDITION = LINK("CONDITION_LINK", ["if", "only if", "only when"])
        SEQUENTIAL = LINK("SEQUENTIAL_LINK", ["and", "also", "or", "then"])
        OWN = LINK("OWN_LINK", ["of", "belong to"])

        self.clueWord_to_link = {}
        for link in [TEMPORAL, CAUSE_EFFECT, PURPORSE, MEANS, CONDITION, SEQUENTIAL, OWN]:
            for clueWord in link.clue_words:
                self.clueWord_to_link[clueWord] = link.name
        self.clue_words = sorted(self.clueWord_to_link.keys())

    def extract_link(self, tokens):
        phrases = []; tags = []
        token_tags = nltk.pos_tag(tokens)
        i = 0
        while i < len(tokens):
            for clue_word in self.clue_words:
                if " ".join(tokens[i:]).startswith(clue_word):
                    i += len(clue_word.split(" "))
                    phrases.append(clue_word)
                    tags.append("SLN-LINK")
                    break
            else:
                phrases.append(tokens[i])
                tags.append(token_tags[i][1])
                i += 1

        return list(zip(phrases, tags))


    def keep(self, tag):
        if tag[:2] in ["VB", "NN", "JJ", "RB"]: return True
        return False
        # not 保留
        # 动词, 名词, 形容词, 副词
        if tag[:2] in ["VB", "NN", "JJ", "RB"]: return True
        # 所有格： you, your
        if tag[:3] == "PRP": return True

        # 情态动词
        if tag[:2] in ["MD"]: return False
        return False
    
    def test2(self, sentences, token_importance, mmr=True, summary_size=5):
        summarys = []

        for sentence in sentences:
            summary = ""; score = 0
            for i, (token, tag) in enumerate(self.extract_link(sentence.split(" "))):
                if tag == "SLN-LINK" or self.keep(tag):
                    summary += token + " "
                    score += 1 if tag == "SLN-LINK" else token_importance.get(token, 0)
            summarys.append((summary, score))
            
        return sorted(summarys, key=lambda x: x[1], reverse=True)
    
    def __call__(self, sentences, token_importance, mmr=True, summary_size=5):
        from copy import deepcopy
        token_importance = deepcopy(token_importance)

        final_summarys = []
        summary_words_set = set()

        def get_sentence_score(sentence):
            if not sentence: return ("", -1)
            summary = ""; score = 0
            for token, tag in nltk.pos_tag(sentence.strip().split(" ")):
                if self.keep(tag) or (token not in summary_words_set):
                    summary += f"{token} "
                    score += token_importance.get(token, 0)
                elif token in self.clue_words:
                    summary += f"{token} "
                    score += 1
            
            return summary, score


        while len(final_summarys) < summary_size + 2:
            sentences = list(map(lambda x: x[0],
                sorted(map(get_sentence_score, sentences), key=lambda x: x[1], reverse=True)
            ))

            summary = sentences[0]
            sentences = sentences[1:]
            final_summarys.append(summary)

            for token in summary.split(" "):
                summary_words_set.update(token)
                token_importance[token] =  token_importance.get(token, 0) / 4

            for sentence in sentences:
                for token in sentence.split(" "):
                    if token in summary.split(" "):
                        for token in sentence.split(" "):
                            token_importance[token] = token_importance.get(token, 0) * 2
                        break

        return [". ".join(final_summarys)]

        return [". ".join(
            list(map(lambda x: x[0],
                sorted(map(get_sentence_score, sentences), key=lambda x: x[1], reverse=True)
            ))[:summary_size]
            )]

        # 加入mmr算法
        final_summarys = []
        summary_words_set = set()

        while len(final_summarys) < summary_size:

            sentences = []
            
            summarys = []
            for sentence in sentences:
                if not sentence: continue

                summary = ""; score = 0
                for i, (token, tag) in enumerate(self.extract_link(sentence.split(" "))):
                    if tag == "SLN-LINK" or self.keep(tag):
                        summary += token + " "
                        score += 1 if tag == "SLN-LINK" else token_importance.get(token, 0)
                summarys.append((summary, score))
            summary = sorted(summarys, key=lambda x: x[1], reverse=True)[0][0]

            for token in summary.split(" "):
                token_importance[token] = token_importance.get(token, 0) / 2
                summary_words_set.update(token)

            final_summarys.append(summary)

        return [". ".join(final_summarys)]






        

        while len(final_summarys) < summary_size:
            summarys = []

            for sentence in sentences:
                if not sentence: continue

                summary = ""; score = 0
                for i, (token, tag) in enumerate(self.extract_link(sentence.strip().split(" "))):
                    if tag == "SLN-LINK" or self.keep(tag) or (token not in summary_words_set):
                        summary += token + " "
                        score += 1 if tag == "SLN-LINK" else token_importance.get(token, 0)
                summarys.append((summary, score))
            
            summary = sorted(summarys, key=lambda x: x[1], reverse=True)[0][0]
            for token in summary.split(" "):
                token_importance[token] = token_importance.get(token, 0) / 4
                summary_words_set.update(token)
        
            final_summarys.append(summary)
            
            sentences = list(map(lambda s: s[0], filter(lambda s: s[0] != summary and s[0].strip() != "", summarys)))
            
            summ_tokens = summary.split(" ")
            for sentence in sentences:
                for summ_token in summ_tokens:
                    if summ_token in sentence.split(" "):
                        for token in sentence.split(" "):
                            if token not in summ_tokens:
                                token_importance[token] = token_importance.get(token, 0) * 4
                        break
        
        return [". ".join(final_summarys)]
        