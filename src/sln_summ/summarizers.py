import copy
import itertools
# import sln_summ.tokenizer
# from sln_summ.sln_construction import make_sln_noun_verb_pronoun_prep_adj_adv as make_sln
from sln_summ.sln_construction import SLN, get_node_tokens, get_link_tokens

def sln_summarize(slns, sentences, word_score_dict, strategies, score_threshold=0.3, decay_rate=0.5, desired_length=150):
    """
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
    summary_sentences = []

    merged_SLN = sum(slns, start=SLN([]))
    while sum(len(sentence) for sentence in summary_sentences) < desired_length:
        index = select_sentence(sentences, slns, word_score_dict)

    score_iteration_list = [copy.deepcopy(word_score_dict)]

    while sum(len(sentence) for sentence in summary_sentences) < desired_length:
        index = select_sentence(sentences, slns, word_score_dict)
        candidate_sentence = sentences[index]

        if "simplification" in strategies:
            kept_word_set = {word for word, score in word_score_dict.item() if score >= score_threshold}
            simplify_sentence(candidate_sentence, kept_word_set)
        
            # check_equivalence(SLN(generated_sentence, construct=True), SLNs[index])
            # 调整重新设置阈值的策略
            # continue
        
        if "diversity" in strategies:
            word_score_dict = decrease_redundancy(slns[index], word_score_dict, decay_rate)

        if "coherent" in strategies:
            word_score_dict = increase_coherence(slns[index], merged_SLN, word_score_dict, decay_rate)

        sentences = sentences[:index] + sentences[index+1:]
        slns = slns[:index] + slns[index+1:]

        summary_sentences.append(candidate_sentence)

        # score dict post normalized
        score_iteration_list.append(copy.deepcopy(word_score_dict))

    return summary_sentences, score_iteration_list

def select_sentence(sentences, slns, word_score_dict, sentence_normalized=True):
    """
    return the index of most informative sentence
    """
    def get_SI_score(sln):
        token_set = set()
        for from_node, link, to_node in sln:
            tokens = get_node_tokens(from_node) + get_link_tokens(link) + get_node_tokens(to_node)
            token_set.update(tokens)
        return sum(
            word_score_dict[word] for word in token_set
        )

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
    if original_sln == simplified_sln: return True

    words = set()
    for from_node, link, to_node in original_sln - simplified_sln:
        words.update(get_link_tokens(link))
        words.update(get_node_tokens(from_node) + get_node_tokens(to_node))
    
    return list(words)

def decrease_redundancy(sln, word_score_dict, decay_rate):
    word_score_dict = copy.deepcopy(word_score_dict)

    for from_node, link, to_node in sln:
        tokens = get_node_tokens(from_node) + get_link_tokens(link) + get_node_tokens(to_node)
        for token in tokens:
            word_score_dict[token] *= (1 - decay_rate)

    return word_score_dict

def increase_coherence(sln, merged_sln, word_score_dict, decay_rate):
    word_score_dict = copy.deepcopy(word_score_dict)

    selected_node_set = set()
    for from_node, _, to_node in sln:
        selected_node_set.add([from_node, to_node])

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

    for token in connected_token_set:
        word_score_dict[token] /= decay_rate

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
        sentences = TextTeaser().summarize("", text)
        return sentences

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

    return sentences

    # sentences = tokenize_sentences_and_words(" ".join([s._text for s in sentences]), remove_stop=False)
    # summary_sents = []
    # while sum(len(summary_sent) for summary_sent in summary_sents) < desired_length and len(summary_sents) < len(sentences):
    #     summary_sents.append(sentences[len(summary_sents)])

    # return summary_sents