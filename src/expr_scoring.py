# SMI的实验：
# 1、在三个数据集合中，把词按照SMI score的大小排序
# 2、和tf, idf, pmi进行对比
import pandas as pd
from functools import partial
from itertools import chain
from sln_summ.corpus import read_cnn_corpus, read_paper_corpus, read_legal_corpus
from sln_summ.tokenizer import tokenize_sentences_and_words
from sln_summ.word_scoring import tf, tf_idf, pmi, smi, centroid_smi
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
    dsmi_dict = centroid_smi(
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
    result = {}

    for name, items in {
        "cnn": (read_cnn_corpus(300),
        "legal": read_legal_corpus(300),
        "acl": read_paper_corpus(200),
    }.items():
        result[name] = stat_items(items, lamda=-0.5, topKs=[5, 10, 20, 30])

    score_csv = pd.DataFrame()
    for name in result:
        for top_N in result[name]:
            for metric, value in result[name][top_N].items():
                score_csv = score_csv.append(
                    {"name": name, "topN": top_N, "metric": metric, "value": value},
                    ignore_index=True,
                )
    score_csv.to_csv("expr_scoring.csv", index=False)