from corpus import read_cnn_corpus
from itertools import chain
from pathlib import Path
from sln import SLN
from summarizers import SLN_summarizer
from tokenizer import tokenize_sentences_and_words
from word_scoring import smi, tf, tf_idf

item = read_cnn_corpus(1)[0]
abstract, introduction, middle, conclusion = item

abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
introduction_sentences_tokens = tokenize_sentences_and_words(introduction, remove_stop=False)
middle_sentences_tokens = tokenize_sentences_and_words(middle, remove_stop=False)
conclusion_sentences_tokens = tokenize_sentences_and_words(conclusion, remove_stop=False)

smi_dict = smi(
    list(chain(*(introduction_sentences_tokens + conclusion_sentences_tokens))),
    list(chain(*middle_sentences_tokens)),
    lamda=0.3
)

sent = SLN_summarizer(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens, smi_dict, strategies=["concise", "diverse", "coherent"])

print(sent)
print(abstract_sentences_tokens)


from rouge.rouge import Rouge
rouge = Rouge("./rouge/RELEASE-1.5.5/")
score = rouge(' '.join(["a d", "g f"]), ["a b c"], 150)
print('%s: ROUGE-1: %4f %4f %4f, ROUGE-2: %4f %4f %4f, ROUGE-L: %4f %4f %4f' % ("SLN", 
    score['rouge_1_f_score'], score['rouge_1_precision'], score['rouge_1_recall'],
        score['rouge_2_f_score'], score['rouge_2_precision'], score['rouge_2_recall'],
        score['rouge_l_f_score'], score['rouge_l_precision'], score['rouge_l_recall']))
