import copy
import json
import math
from collections import Counter
from functools import partial
from pathlib import Path
from tqdm import tqdm
from itertools import chain
from pathlib import Path
from sln_summ.corpus import read_bbc_file_paths
from sln_summ.tokenizer import tokenize_sentences_and_words

log = lambda x: 0 if x == 0 else math.log(x)

def normalize_scoring_dict(score_dict):
    score_dict = copy.deepcopy(score_dict)
    max_score, min_score = max(score_dict.values()), min(score_dict.values())

    if max_score == min_score:
        return score_dict

    for word in score_dict:
        score_dict[word] = (score_dict[word] - min_score) / (max_score - min_score)
    
    return score_dict

def smi(informative_tokens, detailed_tokens, lamda=0.3, normalize=True, debug=False):
    all_tokens = informative_tokens + detailed_tokens
    all_counter = Counter(all_tokens)
    informative_counter = Counter(informative_tokens)
    detailed_counter = Counter(detailed_tokens)

    def _smi(token):
        p_x_yi = informative_counter.get(token, 0) / len(all_tokens)
        p_x_yd = detailed_counter.get(token, 0) / len(all_tokens)

        p_x = all_counter.get(token) / len(all_tokens)
        p_yi = len(informative_tokens) / len(all_tokens)
        p_yd = len(detailed_tokens) / len(all_tokens)

        if p_x == 0 or p_yi == 0 or p_yd == 0:
            return 0

        return p_x_yi * log(p_x_yi / (p_x * p_yi)) - lamda * p_x_yd * log(p_x_yd / (p_x * p_yd))

    score_dict = {
        token: _smi(token) for token in all_counter
    }
    if normalize:
        if not score_dict:
            return score_dict

        max_score, min_score = max(score_dict.values()), min(score_dict.values())

        if max_score == min_score:
            return score_dict

        score_dict = {
            token: (value - min_score) / (max_score - min_score)
            for token, value in score_dict.items()
        }

    if debug:
        for word, _ in sorted(score_dict.items(), key=lambda x: x[1]):
            print(f"{word}\t{informative_counter.get(word, 0)}\t{detailed_counter.get(word, 0)}\t{all_counter.get(word, 0)}\t{score_dict[word]}")

        print(len(informative_tokens), len(detailed_tokens), len(all_tokens))

    return score_dict

pmi = partial(smi, lamda=-1)

def centroid_smi(informative_tokens, detailed_tokens, lamda=0.3):
    all_tokens = informative_tokens + detailed_tokens
    all_counter = Counter(all_tokens)
    smi_dict = smi(informative_tokens, detailed_tokens, lamda=lamda)
    return {
        token: all_counter.get(token) * smi_dict.get(token) for token in all_tokens
    }
    
def tf(tokens):
    return Counter(tokens)

bbc_idf_dict = None
def init_idf_dict():
    global bbc_idf_dict
    token_occurrence = {}

    for file_path in tqdm(read_bbc_file_paths(), desc="bbc idf calculation"):
        text = file_path.read_text()
        for token in set(chain(*tokenize_sentences_and_words(text))):
            token_occurrence[token] = token_occurrence.get(token, 0) + 1

    for token in token_occurrence:
        token_occurrence[token] = 1 / token_occurrence[token]
    bbc_idf_dict = token_occurrence

def tf_idf(tokens):
    global bbc_idf_dict
    if not bbc_idf_dict:
        init_idf_dict()
    tokens = list(tokens)
    token_counter = Counter(tokens)
    return {
        token: token_counter.get(token) * bbc_idf_dict.get(token, 0) for token in tokens
    }


def test_smi():
    informative_tokens = [
        "georgia", "mayor", "assails", "governor's", "move", "to", "reopen", "beaches"
    ]
    detailed_tokens = "the tybee island city council voted to close the beaches on march 20 but georgia gov brian kemp issued a statewide shelter-in-place executive order which supersedes all local orders relating to coronavirus and also opened up the state's beaches".split(" ")
    score_dict = smi(informative_tokens, detailed_tokens, normalize=True, debug=True)
    print(score_dict)

def test_smi2():
    from sln_summ.corpus import read_paper_corpus
    import itertools

    abstract, text, informative, detailed = read_paper_corpus()[0]
    score_dict = smi(list(itertools.chain(*tokenize_sentences_and_words(informative))), list(itertools.chain(*tokenize_sentences_and_words(detailed))), normalize=True, debug=True)
    print(score_dict)


if __name__ == "__main__":
    test_smi2()
