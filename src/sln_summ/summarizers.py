import copy
import itertools
from sln_summ.sln_construction import SLN, get_node_tokens, get_link_tokens, make_sln_noun_verb_pronoun_prep_adj_adv
from sln_summ.word_scoring import normalize_scoring_dict

def truncate_summary_by_words(summary, desired_length=100):
    truncated_summary = []
    length = 0
    for words in summary:
        truncated_summary.append(words)
        length += len(words)
        if length > desired_length:
            break
    
    return truncated_summary


def sln_summarize(sentences, word_score_dict, strategies,
                sentence_normalized=True,
                make_sln_func=make_sln_noun_verb_pronoun_prep_adj_adv,
                score_threshold=0.3, decay_rate=0.5, desired_length=150, verbose=False, call_back_fn=None):
    """
    sentences = [[words], ...]
    Expected:
    strategies = {
        "completeness": {
            "sentence_normalized": True
        },
        "simplification": {
            "score_threshold": 0.3,
        },
        "conciseness": {},
        "diversity": {},
        "
    }
    """
    if verbose:
        v_log = lambda *a: print(a)
    else:
        v_log = lambda *a: None

    word_score_dict = normalize_scoring_dict(word_score_dict)
    slns = [make_sln_func(sentence) for sentence in sentences]

    summary_sentences = []
    score_iteration_list = [("init", copy.deepcopy(word_score_dict))]
    merged_SLN = sum(slns, start=SLN([]))

    iteration_step = 1
    while sentences and sum(len(sentence) for sentence in summary_sentences) < desired_length:
        index = select_sentence(sentences, slns, word_score_dict)
        candidate_words = sentences[index]

        v_log(f"Step {iteration_step} Selection")
        v_log(f"{index}th sentence: {candidate_words}")

        if "simplification" in strategies:
            kept_word_set = {word for word, score in word_score_dict.items() if score >= score_threshold}
            simplified_words = simplify_sentence(candidate_words, kept_word_set)
        
            # 删除了一些关键词以后导致SLN无法构造回去了
            if len(ineq_words := find_inequivalence_words(
                slns[index],
                make_sln_func(simplified_words),
            )) != 0:
                for _word in candidate_words:
                    word_score_dict[_word] = max(word_score_dict.get(_word, 0), score_threshold)
                # for _word in ineq_words:
                #     word_score_dict[_word] = max(word_score_dict.get(_word), score_threshold)
                score_iteration_list.append(("simp", copy.deepcopy(word_score_dict)))

                v_log(f"Step {iteration_step} Simplification Failed.")
                v_log(f"{ineq_words} are set as score threshold {score_threshold}.")
                v_log(simplified_words)

                iteration_step += 1
                continue
            
            candidate_words = simplified_words

            v_log(f"Step {iteration_step} Simplification Passed.")
            v_log(f"simplified_sentence is {candidate_words}")
        
        if "diversity" in strategies:
            word_score_dict = decrease_redundancy(slns[index], word_score_dict, decay_rate)
            score_iteration_list.append(("div", copy.deepcopy(word_score_dict)))
            v_log(f"Step {iteration_step} Diversity Passed.")

        if "coherence" in strategies:
            word_score_dict = increase_coherence(slns[index], merged_SLN, word_score_dict, decay_rate)
            score_iteration_list.append(("coh", copy.deepcopy(word_score_dict)))
            v_log(f"Step {iteration_step} Coherence Passed.")

        sentences = sentences[:index] + sentences[index+1:]
        slns = slns[:index] + slns[index+1:]

        summary_sentences.append(candidate_words)

        # score dict post-normalization
        word_score_dict = normalize_scoring_dict(word_score_dict)

        iteration_step += 1
    
    summary_sentences = [" ".join(words) + "." for words in summary_sentences]

    return "".join(summary_sentences)

def select_sentence(sentences, slns, word_score_dict, sentence_normalized=True):
    """
    return the index of most informative sentence
    """
    def get_SI_score(sln):
        token_set = set()
        for from_node, link, to_node in sln:
            tokens = get_node_tokens(from_node) + get_link_tokens(link) + get_node_tokens(to_node)
            token_set.update(tokens)
        try:
            return sum(
                word_score_dict[word] for word in token_set)
        except Exception as ex:
            print(ex)
            return sum(
                word_score_dict.get(word, 0) for word in token_set)

    sentence_scores = [
        (index, get_SI_score(sln)) for index, sln in enumerate(slns) if (score := get_SI_score) != 0
    ]

    if sentence_normalized:
        sentence_scores = [
            (index, score / len(sentences[index])) for index, score in sentence_scores
        ]
    
    sentence_scores = sorted(sentence_scores, key=lambda tp: tp[1], reverse=True)

    if len(sentence_scores) == 0:
        raise ValueError("Error Input with non-sense sentences")

    # return the index
    return sentence_scores[0][0]

def simplify_sentence(sentence, kept_word_set):
    return [
        word for word in sentence if word in kept_word_set
    ]

def find_inequivalence_words(original_sln, simplified_sln):
    if original_sln == simplified_sln: 
        return []

    original_word_set, simplified_word_set = set(), set()

    for from_node, link, to_node in original_sln:
        original_word_set.update(get_link_tokens(link) + get_node_tokens(from_node) + get_node_tokens(to_node))
    for from_node, link, to_node in simplified_sln:
        simplified_word_set.update(get_link_tokens(link) + get_node_tokens(from_node) + get_node_tokens(to_node))

    return original_word_set - simplified_word_set

def decrease_redundancy(sln, word_score_dict, decay_rate):
    word_score_dict = copy.deepcopy(word_score_dict)

    tokens = set()
    for from_node, link, to_node in sln:
        tokens.update(
            get_node_tokens(from_node) + get_link_tokens(link) + get_node_tokens(to_node)
        )
    
    for token in tokens:
        if token in word_score_dict:
            word_score_dict[token] *= (1 - decay_rate)

    return word_score_dict

def increase_coherence(sln, merged_sln, word_score_dict, decay_rate):
    word_score_dict = copy.deepcopy(word_score_dict)

    selected_node_set = set()
    selected_word_set = set()
    for from_node, link, to_node in sln:
        selected_node_set.update([from_node, to_node])
        selected_word_set.update(
            get_node_tokens(from_node) + get_link_tokens(link) + get_node_tokens(to_node)
        )

    connected_token_set = set()
    for node in selected_node_set:
        connected_nodes = merged_sln.find_connected_nodes(node)
        connected_links = merged_sln.find_connected_links(node)

        connected_token_set.update(
            itertools.chain(*(get_node_tokens(_node) for _node in connected_nodes))
        )
        connected_token_set.update(
            itertools.chain(*(get_link_tokens(_link) for _link in connected_links))
        )

    for token in connected_token_set - selected_word_set:
        if token in word_score_dict:
            word_score_dict[token] /= 1 - decay_rate

    return word_score_dict

def summarize_from_library(text, method, desired_length=150):
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.lsa import LsaSummarizer
    from sumy.summarizers.kl import KLSummarizer
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.tokenizers import Tokenizer as STokenizer
    from textteaser import TextTeaser

    if method == "textteaser":
        sentences = TextTeaser().summarize("title test", text)
        return " ".join(sentences)

    sentences = {
        "luhn": LuhnSummarizer,
        "lsa": LsaSummarizer,
        "kl": KLSummarizer,
        "lexrank": LexRankSummarizer,
        "textrank": TextRankSummarizer,
    }[method]()(
        PlaintextParser.from_string(text, STokenizer("english")).document,
        sentences_count=5,
    )

    return " ".join([s._text for s in sentences])

    # sentences = tokenize_sentences_and_words(" ".join([s._text for s in sentences]), remove_stop=False)
    # summary_sents = []
    # while sum(len(summary_sent) for summary_sent in summary_sents) < desired_length and len(summary_sents) < len(sentences):
    #     summary_sents.append(sentences[len(summary_sents)])

    # return summary_sents
