from corpus import read_cnn_corpus
from sln import SLN
from summarizers import SLN_summarizer
from word_scoring import smi, tf, tf_idf
from 

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

SLN_summarizer(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens, smi_dict)