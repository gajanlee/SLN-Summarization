import itertools
import sys
import pandas as pd
from pathlib import Path
from sln_summ.corpus import *
from sln_summ.metrics import rouge_perl
from sln_summ.sln_construction import *
from sln_summ.summarizers import *
from sln_summ.tokenizer import *
from sln_summ.word_scoring import *

def lib_summarization(items, model_name):
    summarys = []

    for item in items:
        abstract, text, informative, detailed = item
        summary = summarize_from_library(text, model_name)
        summarys.append(summary)

    return summarys

def run_all_library_summarizations(items, base_dir):
    for model_name, desired_length in tqdm(list(itertools.product(["luhn", "lsa", "kl", "lexrank", "textrank", "textteaser"], [50, 100, 120, 150])), desc="library summarizing"):
        key = f"{model_name}_{desired_length}"
        print(f"running {key} {base_dir}")
        if (base_dir / key).exists() and len(list((base_dir / key).glob("*"))) != 0:
            print(f"{key} exists")
            continue
            
        summarys = lib_summarization(items, model_name)
        summarys = [truncate_summary_by_words(tokenize_sentences_and_words(summary), desired_length) for summary in summarys]
        summarys_sentences = [[" ".join(words) + "." for words in summary] for summary in summarys]
        yield key, summarys_sentences

def sln_summarization_items(items, score_approach, lamda, *args, **kwargs):
    summarys = []

    for item in tqdm(items, desc="sln summarizing"):
        abstract, text, informative, detailed = item
        sentences = tokenize_sentences_and_words(text)

        abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
        text_sentence_tokens = tokenize_sentences_and_words(text, remove_stop=False)
        informative_sentence_tokens = tokenize_sentences_and_words(informative, remove_stop=False)
        detailed_sentence_tokens = tokenize_sentences_and_words(detailed, remove_stop=False)

        if score_approach == "tf":
            score_dict = tf(list(chain(*text_sentence_tokens)))
        elif score_approach == "idf":
            score_dict = tf_idf(list(chain(*text_sentence_tokens)))
        elif score_approach == "smi":
            score_dict = smi(
                list(chain(*informative_sentence_tokens)),
                list(chain(*detailed_sentence_tokens)),
                lamda=lamda,
            )
        elif score_approach == "dsmi":
            score_dict = centroid_smi(
                list(chain(*informative_sentence_tokens)),
                list(chain(*detailed_sentence_tokens)),
                lamda=lamda
            )
        kwargs["verbose"] = True
        summary = sln_summarize(sentences, score_dict, *args, **kwargs)

        summarys.append(summary)
    return summarys

def combo_params(*params):
    for i in range(len(params)):
        # Ith parameters product
        for alternative in params[i]:
            yield [alts[0] for alts in params[:i]] + [alternative] + [alts[0] for alts in params[i+1:]]

def run_all_sln_summarzations(items, base_dir):
    score_approaches = ["dsmi", "tf", "idf", "smi"]
    strategies = [
        ["simplification", "diversity", "coherence"],
        ["simplification"],
        ["diversity"],
        ["coherence"]
    ]
    sentence_normalized_alternatives = [True, False]
    make_sln_funcs = [
        make_sln_noun_verb_pronoun_prep_adj_adv,
        make_sln_noun_verb_pronoun,
        make_sln_noun_verb_pronoun_rich,
        make_sln_noun_verb_pronoun_prep,
        make_sln_noun_verb_pronoun_prep_adj,
    ]
    score_thresholds = [0.3, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
    decay_rates = [0.3, 0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
    desired_lengths = [150, 50, 100, 120]
    lamdas = [0.3, -0.3, -0.5, -0.8, 0.1, 0.5, 0.8]

    summary_recorder = {}

    for (score_approach, strategy, sentence_normalized, make_sln_func,
        score_threshold, decay_rate, desired_length, lamda) in tqdm(list(combo_params(
        score_approaches, strategies, sentence_normalized_alternatives,
        make_sln_funcs, score_thresholds, decay_rates, desired_lengths, lamdas
    )), desc="sln summarizing"):
        key = f"{score_approach}_{'_'.join(strategy)}_{sentence_normalized}_{make_sln_func.__name__}_{score_threshold}_{decay_rate}_{desired_length}"
        print(f"running {key} base_dir")
        if (base_dir / key).exists() and len(list((base_dir / key).glob("*"))) != 0:
            print(key, " exists")
            continue
        yield (key, sln_summarization_items(items, score_approach, lamda,
            strategies=strategy,
            sentence_normalized=sentence_normalized,
            make_sln_func=make_sln_func,
            score_threshold=score_threshold,
            decay_rate=decay_rate,
            desired_length=desired_length,))

if __name__ == "__main__":
    expr = sys.argv[1]
    output_dir = Path("/home/ubuntu/SLN_Summarization_output")
    if (score_path := Path(f"./expr_summarization_{expr}.csv")).exists():
        score_csv = pd.read_csv(score_path)
    else:
        score_csv = pd.DataFrame()

    datasets = {
        "paper": read_paper_corpus(200),
        "cnn": read_cnn_corpus(200),
        "legal": read_legal_corpus(50),
    }

    for data_name, items in datasets.items():
        base_dir = output_dir / data_name
        if expr == "lib":
            summary_iterator = run_all_library_summarizations(items, base_dir)
        elif expr == "sln":
            summary_iterator = run_all_sln_summarzations(items, base_dir)
        
        for key, summarys in summary_iterator:
            abstracts = [tokenize_sentences_and_words(item[0], cut_words=False) for item in items]
            
            _abstracts, _summarys = [], []
            for abstract, summary in zip(abstracts, summarys):
                if abstract and summary:
                    _abstracts.append(abstract)
                    _summarys.append(summary)
            print(f"len abstracts {len(_abstracts)}, len summarys {len(_summarys)}")
            score = rouge_perl(_abstracts, _summarys, base_dir / key)

            score_csv = score_csv.append({
                "dataset": f"{data_name}/{len(items)}",
                "key": key,
                "rouge-1(F/R/P)": f"{score['rouge-1']['f']}/{score['rouge-1']['r']}/{score['rouge-l']['p']}",
                "rouge-2(F/R/P)": f"{score['rouge-2']['f']}/{score['rouge-2']['r']}/{score['rouge-2']['p']}",
                "rouge-l(F/R/P)": f"{score['rouge-l']['f']}/{score['rouge-l']['r']}/{score['rouge-l']['p']}",
            }, ignore_index=True)
    score_csv.to_csv(score_path, index=False)
