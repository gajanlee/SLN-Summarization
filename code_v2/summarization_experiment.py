import json
import sys
from corpus import read_cnn_corpus, read_paper_corpus, read_legal_corpus
from functools import partial
from itertools import chain, product
from tqdm import tqdm
from pathlib import Path
from sln import SLN
from summarizers import SLN_summarizer
from tokenizer import tokenize_sentences_and_words
from word_scoring import smi, tf, tf_idf
from metrics import rouge_perl

def rouge_score(reference, hypothesis):
    """
    reference: truth
    hypothesis: summary

    return: {
        "rouge-1": {"p": 1.0, "r": 1.0, "f": s}
        "rouge-2": ...,
        "rouge-l": ...,
    }
    """
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores[0]

def sln_summarize_item(item, strategies=["concise", "diverse", "coherent"], use_smi=True, smi_lambda=0.3):
    abstract, introduction, middle, conclusion = item

    introduction_sentences_tokens = tokenize_sentences_and_words(introduction, remove_stop=False)
    middle_sentences_tokens = tokenize_sentences_and_words(middle, remove_stop=False)
    conclusion_sentences_tokens = tokenize_sentences_and_words(conclusion, remove_stop=False)

    if use_smi:
        score_dict = smi(
            list(chain(*(introduction_sentences_tokens + conclusion_sentences_tokens))),
            list(chain(*middle_sentences_tokens)),
            lamda=smi_lambda
        )
    else:
        score_dict = tf(list(chain(
        *(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens)
    )))

    sents, _ = SLN_summarizer(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens, score_dict, strategies=["concise", "diverse", "coherent"])

    # print(sent)
    # print(abstract_sentences_tokens)
    # summary_sln = [" ".join(tokens) for tokens in sent]
    # print(summary_sln)
    # from rouge.rouge import Rouge
    # rouge = Rouge("./rouge/RELEASE-1.5.5/")
    # score = rouge(' '.join(["a d", "g f"]), ["a b c"], 150)
    # print('%s: ROUGE-1: %4f %4f %4f, ROUGE-2: %4f %4f %4f, ROUGE-L: %4f %4f %4f' % ("SLN", 
    #     score['rouge_1_f_score'], score['rouge_1_precision'], score['rouge_1_recall'],
    #         score['rouge_2_f_score'], score['rouge_2_precision'], score['rouge_2_recall'],
    #         score['rouge_l_f_score'], score['rouge_l_precision'], score['rouge_l_recall']))

    return sents


def summarize_from_library(item, method):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.kl import KLSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.tokenizers import Tokenizer as STokenizer

    abstract, introduction, middle, conclusion = item
    text = introduction + middle + conclusion

    sentences = {
        "lsa": LsaSummarizer,
        "kl": KLSummarizer,
        "luhn": LuhnSummarizer,
        "lexrank": LexRankSummarizer,
        "textrank": TextRankSummarizer,
    }[method]()(
        PlaintextParser.from_string(text, STokenizer("english")).document,
        sentences_count=3,
    )

    sentences = tokenize_sentences_and_words(" ".join([s._text for s in sentences]), remove_stop=False)
    summary_sents = []
    while sum(len(summary_sent) for summary_sent in summary_sents) < 150 and len(summary_sents) < len(sentences):
        summary_sents.append(sentences[len(summary_sents)])
    # summary_sents = [" ".join(tokens) for tokens in summary_sents]

    return summary_sents

def summarize_items(items, func):
    scores = {
        "rouge-1": {"f": 0, "p": 0, "r": 0},
        "rouge-2": {"f": 0, "p": 0, "r": 0},
        "rouge-l": {"f": 0, "p": 0, "r": 0},
    }

    for i, item in enumerate(tqdm(items)):
        abstract, introduction, middle, conclusion = item
        abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
        abstract = ". ".join(" ".join(tokens) for tokens in abstract_sentences_tokens)

        summary = func(item)
        summary = ". ".join(" ".join(tokens) for tokens in summary)

        score = rouge_score(abstract, summary)
        # rouge_score = sln_summarize_item(item)
        # score = summarize_from_library(text, abstract, method)

        for k in ['1', '2', 'l']:
            for m in ['f', 'p', 'r']:
                scores[f"rouge-{k}"][m] += score[f'rouge-{k}'][m]
    
    for k in ['1', '2', 'l']:
            for m in ['f', 'p', 'r']:
                scores[f"rouge-{k}"][m] /= (i + 1)
    print(scores, i)


def summarize_items_perl_eval(items, func, experimental_name):
    refs = []
    hyps = []

    for i, item in enumerate(tqdm(items)):
        abstract, introduction, middle, conclusion = item
        abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
        abstract = [" ".join(tokens) for tokens in abstract_sentences_tokens]

        summary = func(item)
        summary = [" ".join(tokens) for tokens in summary]

        refs.append(abstract)
        hyps.append(summary)

    score = rouge_perl(refs, hyps, experimental_name)
    print(experimental_name, score)
    return score

def run_all_library_summarizations():
    model_names = ["lsa", "kl", "luhn", "lexrank", "textrank"]
    datasets = {
        "paper": read_paper_corpus(200),
        "cnn": read_cnn_corpus(1000),
        "legal": read_legal_corpus(1000),
    }
    for model_name, data_name in product(model_names, datasets.keys()):
        print(model_name, data_name)
        func = partial(summarize_from_library, method=model_name)
        score = summarize_items_perl_eval(datasets[data_name], func, f"{model_name}_{data_name}")
        json.dump(score, open(f"scores/{model_name}_{data_name}_score.json", "w"))


def run_sln_experiments():
    datasets = {
        "paper": read_paper_corpus(50),
        "cnn": read_cnn_corpus(200),
        "legal": read_legal_corpus(50),
    }
    for data_name in datasets.keys():
        print(data_name)
        func = partial(sln_summarize_item, strategies=["concise", "diverse", "coherent"], use_smi=True, smi_lambda=0.3)
        score = summarize_items_perl_eval(datasets[data_name], func, f"sln_{data_name}")
        json.dump(score, open(f"scores/sln_{data_name}_score.json", "w"))

if __name__ == "__main__":
    if (expr := sys.argv[1]) == "all":
        run_all_library_summarizations()
    elif expr == "sln":
        run_sln_experiments()
