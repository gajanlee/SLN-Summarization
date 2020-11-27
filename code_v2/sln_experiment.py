from sln import SLN
from itertools import chain
from collections import Counter
from corpus import read_paper_corpus, read_legal_corpus, read_cnn_corpus
from functools import partial
from tokenizer import tokenize_sentences_and_words
from word_scoring import smi, tf, pmi, tf_idf
from tqdm import tqdm


def stat_sln_item(item):
    abstract, introduction, middle, conclusion = item

    abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=False)
    introduction_sentences_tokens = tokenize_sentences_and_words(introduction, remove_stop=False)
    middle_sentences_tokens = tokenize_sentences_and_words(middle, remove_stop=False)
    conclusion_sentences_tokens = tokenize_sentences_and_words(conclusion, remove_stop=False)

    words = list(chain(*(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens)))
    s = SLN(words)
    s.construct()
    
    c = Counter([e.element_type for e in s.semantic_elements])
    return c

def stat_sln_items(items):
    counter = Counter()
    for item in tqdm(items):
        c = stat_sln_item(item)
        counter.update(c)
    
    return {
        link: count / len(items) for link, count in dict(counter).items()
    }
        
        


if __name__ == "__main__":
    # items = read_paper_corpus() # + read_cnn_corpus() + read_legal_corpus()
    result = {}

    for name, items in {
        "cnn": read_cnn_corpus(258),
        "legal": read_legal_corpus(139),
        "acl": read_paper_corpus(),
    }.items():

        result[name] = stat_sln_items(items)
    
    print(result)