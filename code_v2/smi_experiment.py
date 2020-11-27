# SMI的实验：
# 1、在三个数据集合中，把词按照SMI score的大小排序
# 2、和tf, idf, pmi进行对比

from itertools import chain
from corpus import read_paper_corpus, read_legal_corpus, read_cnn_corpus
from functools import partial
from tokenizer import tokenize_sentences_and_words
from word_scoring import smi, tf, pmi, tf_idf, dsmi
from tqdm import tqdm

def stat_item(item, lamda, topKs):
    abstract, introduction, middle, conclusion = item

    abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
    introduction_sentences_tokens = tokenize_sentences_and_words(introduction, remove_stop=False)
    middle_sentences_tokens = tokenize_sentences_and_words(middle, remove_stop=False)
    conclusion_sentences_tokens = tokenize_sentences_and_words(conclusion, remove_stop=False)

    smi_dict = smi(
        list(chain(*(introduction_sentences_tokens + conclusion_sentences_tokens))),
        list(chain(*middle_sentences_tokens)),
        lamda=lamda
    )
    dsmi_dict = dsmi(
        list(chain(*(introduction_sentences_tokens + conclusion_sentences_tokens))),
        list(chain(*middle_sentences_tokens)),
        lamda=lamda
    )
    tf_dict = tf(list(chain(
        *(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens)
    )))
    idf_dict = tf_idf(list(chain(
        *(introduction_sentences_tokens + middle_sentences_tokens + conclusion_sentences_tokens)
    )))
    pmi_dict = pmi(
        list(chain(*(introduction_sentences_tokens + conclusion_sentences_tokens))),
        list(chain(*middle_sentences_tokens)),
    )

    abstract_tokens = list(chain(*abstract_sentences_tokens))
    # print(abstract_tokens)

    topK_fn = lambda sd, topK: [tp[0] for tp in sorted(sd.items(), key=lambda x: x[1], reverse=True)[:topK]]
    recall_fn = lambda lst: len(set(abstract_tokens) & set(lst))

    # print(top20_fn(tf_dict))
    # print(set(abstract_tokens) & set(top20_fn(tf_dict)))
    # print(top20_fn(smi_dict))
    # print(set(abstract_tokens) & set(top20_fn(smi_dict)))

    return {
        f"top{topK}": {
            "tf": recall_fn(topK_fn(tf_dict, topK)),
            "pmi": recall_fn(topK_fn(pmi_dict, topK)),
            "smi": recall_fn(topK_fn(smi_dict, topK)),
            "tf_idf": recall_fn(topK_fn(idf_dict, topK)),
            "dsmi": recall_fn(topK_fn(dsmi_dict, topK)),
        } for topK in topKs
    }

def stat_items(items, lamda, topKs):
    stat_dict = {
        f"top{topK}": {
            "tf": [],
            "pmi": [],
            "smi": [],
            "tf_idf": [],
            "dsmi": [],
        } for topK in topKs
    }
    for item in tqdm(items, desc="running"):
        result_dict = stat_item(item, lamda, topKs)

        for topK in topKs:
            stat_dict[f"top{topK}"]["tf"].append(result_dict[f"top{topK}"]["tf"])
            stat_dict[f"top{topK}"]["pmi"].append(result_dict[f"top{topK}"]["pmi"])
            stat_dict[f"top{topK}"]["smi"].append(result_dict[f"top{topK}"]["smi"])
            stat_dict[f"top{topK}"]["tf_idf"].append(result_dict[f"top{topK}"]["tf_idf"])
            stat_dict[f"top{topK}"]["dsmi"].append(result_dict[f"top{topK}"]["dsmi"])

    return {
            f"top{topK}": {
            "tf": sum(stat_dict[f"top{topK}"]["tf"]) / len(stat_dict[f"top{topK}"]["tf"]),
            "pmi": sum(stat_dict[f"top{topK}"]["pmi"]) / len(stat_dict[f"top{topK}"]["pmi"]),
            "smi": sum(stat_dict[f"top{topK}"]["smi"]) / len(stat_dict[f"top{topK}"]["smi"]),
            "tf_idf": sum(stat_dict[f"top{topK}"]["tf_idf"]) / len(stat_dict[f"top{topK}"]["tf_idf"]),
            "dsmi": sum(stat_dict[f"top{topK}"]["dsmi"]) / len(stat_dict[f"top{topK}"]["dsmi"]),
        } for topK in topKs
    }

if __name__ == "__main__":
    # items = read_paper_corpus() # + read_cnn_corpus() + read_legal_corpus()
    result = {}

    for name, items in {
        "cnn": read_cnn_corpus(3000),
        "legal": read_legal_corpus(3000),
        "acl": read_paper_corpus(),
    }.items():
        result[name] = stat_items(items, lamda=0.3, topKs=[5, 10, 20, 30])
    
    print(result)