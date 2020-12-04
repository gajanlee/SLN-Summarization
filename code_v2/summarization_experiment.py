from corpus import read_cnn_corpus, read_paper_corpus
from functools import partial 
from itertools import chain
from tqdm import tqdm
from pathlib import Path
from sln import SLN
from summarizers import SLN_summarizer
from tokenizer import tokenize_sentences_and_words
from word_scoring import smi, tf, tf_idf

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

    sent = SLN_summarizer(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens, score_dict, strategies=["concise", "diverse", "coherent"])

    # print(sent)
    # print(abstract_sentences_tokens)
    summary_sln = ". ".join(" ".join(tokens) for tokens in sent)
    # from rouge.rouge import Rouge
    # rouge = Rouge("./rouge/RELEASE-1.5.5/")
    # score = rouge(' '.join(["a d", "g f"]), ["a b c"], 150)
    # print('%s: ROUGE-1: %4f %4f %4f, ROUGE-2: %4f %4f %4f, ROUGE-L: %4f %4f %4f' % ("SLN", 
    #     score['rouge_1_f_score'], score['rouge_1_precision'], score['rouge_1_recall'],
    #         score['rouge_2_f_score'], score['rouge_2_precision'], score['rouge_2_recall'],
    #         score['rouge_l_f_score'], score['rouge_l_precision'], score['rouge_l_recall']))

    return summary_sln


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
    summary_sents = ". ".join(" ".join(tokens) for tokens in summary_sents)

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

        score = rouge_score(abstract, summary)
        # rouge_score = sln_summarize_item(item)
        # score = summarize_from_library(text, abstract, method)

        for k in ['1', '2', 'l']:
            for m in ['f', 'p', 'r']:
                scores[f"rouge-{k}"][m] += score[f'rouge-{k}'][m]
    
    for k in ['1', '2', 'l']:
            for m in ['f', 'p', 'r']:
                scores[f"rouge-{k}"][m] += score[f'rouge-{k}'][m] / (i + 1)
    print(scores, i)

# func = sln_summarize_item
# summarize_items(read_paper_corpus()[:10], func)

func = partial(summarize_from_library, method="lsa")
summarize_items(read_paper_corpus()[:10], func)