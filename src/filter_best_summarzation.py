from sln_summ.corpus import *
from sln_summ.metrics import rouge_score
from sln_summ.sln_construction import *
from sln_summ.summarizers import *
from sln_summ.tokenizer import *
from sln_summ.word_scoring import *
from pathlib import Path
from tqdm import tqdm

def rouge_summarization_selection(items):
    records = []
    for item in tqdm(list(items)):
        abstract, text, informative, detailed = item
        sentences = tokenize_sentences_and_words(text)

        abstract_sentences_tokens = tokenize_sentences_and_words(abstract, remove_stop=True)
        text_sentence_tokens = tokenize_sentences_and_words(text, remove_stop=False)
        informative_sentence_tokens = tokenize_sentences_and_words(informative, remove_stop=False)
        detailed_sentence_tokens = tokenize_sentences_and_words(detailed, remove_stop=False)
        score_dict = centroid_smi(
                list(chain(*informative_sentence_tokens)),
                list(chain(*detailed_sentence_tokens)),
                lamda=0.1
            )
        for summary in sln_summ():
            summary = sln_summarization_items(items, score_approach, 0.1,
                strategies=["simplification", "diversity", "coherence"],
                sentence_normalized=True,
                make_sln_func=make_sln_noun_verb_pronoun_prep_adj_adv,
                score_threshold=0.3,
                decay_rate=0.1,
                desired_length=150)
        
