from corpus import read_cnn_corpus
from sln import SLN
from summarizers import SLN_summarizer

item = read_cnn_corpus(1)[0]
abstract, introduction, middle, conclusion = item

SLN_summarizer(introduction, middle, conclusion)

