# SMI的实验：
# 1、在三个数据集合中，把词按照SMI score的大小排序
# 2、和tf, idf, pmi进行对比
import pandas as pd
from pathlib import Path
from functools import partial
from itertools import chain
from sln_summ.corpus import read_cnn_corpus, read_paper_corpus, read_legal_corpus
from sln_summ.tokenizer import tokenize_sentences_and_words
from sln_summ.word_scoring import tf, tf_idf, pmi, smi, centroid_smi
from tqdm import tqdm

def stat_item(item, name, topKs, **kwargs):
    abstract, text, informative, detailed = item

    abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
    text_sentence_tokens = tokenize_sentences_and_words(text, remove_stop=False)
    informative_sentence_tokens = tokenize_sentences_and_words(informative, remove_stop=False)
    detailed_sentence_tokens = tokenize_sentences_and_words(detailed, remove_stop=False)

    if name == "tf":
        score_dict = tf(list(chain(*text_sentence_tokens)))
    elif name == "idf":
        score_dict = tf_idf(list(chain(*text_sentence_tokens)))
    elif name == "smi":
        score_dict = smi(
            list(chain(*informative_sentence_tokens)),
            list(chain(*detailed_sentence_tokens)),
            lamda=kwargs["lamda"]
        )
    elif name == "dsmi":
        score_dict = centroid_smi(
            list(chain(*informative_sentence_tokens)),
            list(chain(*detailed_sentence_tokens)),
            lamda=kwargs["lamda"]
        )

    abstract_tokens = list(chain(*abstract_sentences_tokens))

    topK_fn = lambda sd, topK: [tp[0] for tp in sorted(sd.items(), key=lambda x: x[1], reverse=True)[:topK]]
    recall_fn = lambda lst: len(set(abstract_tokens) & set(lst))

    return {
        topK: recall_fn(topK_fn(score_dict, topK))
        for topK in topKs
    }

def stat_items(items, *args, **kwargs):
    score_dict = {}
    for item in tqdm(items, desc=f"running, {args}, {kwargs}"):
        item_score_dict = stat_item(item, *args, **kwargs)

        for key, val in item_score_dict.items():
            score_dict[key] = score_dict.get(key, 0) + val

    for key in score_dict:
        score_dict[key] /= len(items)

    return score_dict

if __name__ == "__main__":
    data_items = {
        # "cnn": read_cnn_corpus(300),
        "legal_cite": read_legal_corpus(100),
        # "acl": read_paper_corpus(200),
    }
    # result = {"cnn": {}, "legal": {}, "acl": {}}
    result = {key: {} for key in data_items}
    topKs = [5, 10, 20, 30]
    
    if (score_path := Path("./expr_scoring.csv")).exists():
        score_csv = pd.read_csv(score_path)
    else:
        score_csv = pd.DataFrame()


    for lamda in [-3, -1, -0.7, -0.5, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 3]:
        for name in ["smi", "dsmi"]:
            for data_name, items in data_items.items():
                if name not in result[data_name]:
                    result[data_name][name] = {}
                if lamda not in result[data_name][name]:
                    result[data_name][name][lamda] = {}
                result[data_name][name][lamda] = stat_items(items, name, topKs=topKs, lamda=lamda)

                for top_K in topKs:
                    score_csv = score_csv.append(
                        {"data": data_name, "metric": name, "lambda": lamda, "topK": top_K, "value": result[data_name][name][lamda][top_K]},
                        ignore_index=True,
                    )

    for name in ["tf", "idf"]:
        for data_name, items in data_items.items():
            if name not in result[data_name]:
                result[data_name][name] = {}
            result[data_name][name] = stat_items(items, name, topKs=[5, 10, 20, 30], lamda=lamda)

            for top_K in topKs:
                score_csv = score_csv.append(
                    {"data": data_name, "metric": name, "topK": top_K, "value": result[data_name][name][top_K]},
                    ignore_index=True,
                )
    score_csv.to_csv(score_path, index=False)
