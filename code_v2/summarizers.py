from copy import deepcopy
from sln import *

def SLN_summarizer(sentences, word_score_dict, strategies=[], desired_length=150, threshold=0.3, decay_rate=0.5):
    SLNs = [SLN(sentence) for sentence in sentences]
    for sln in SLNs:
        sln.construct()
    summary_sentences = []

    while sum(len(sentence) for sentence in summary_sentences) < desired_length:
        index = select_sentence(sentences, SLNs, word_score_dict)

        generated_sentence = sentences[index]

        if "concise" in strategies:
            generated_sentence = simplify_sentence(generated_sentence, word_score_dict, threshold)

        check_equivalence(SLN(generated_sentence, construct=True), SLNs[index])
        # 调整重新设置阈值的策略
        if "diverse" in strategies:
            word_score_dict = decrease_redundancy(SLNs[index], word_score_dict, decay_rate)

        if "coherent" in strategies:
            word_score_dict = increase_coherence(SLN(generated_sentence, construct=True), SLNs, word_score_dict, decay_rate)

        summary_sentences.append(generated_sentence)
        sentences = sentences[:index] + sentences[index+1:]
    
    return summary_sentences

def select_sentence(sentences, SLNs, word_score_dict):
    """
    return the index of most informative sentence
    """
    index_score_list = []
    for index, (sentence, SLN) in enumerate(zip(sentences, SLNs)):
        sentence_score = 0
        for element in SLN.semantic_elements:
            node_score = sum(word_score_dict[word] for word in element.literal.split(" "))
            sentence_score += node_score
        index_score_list.append((index, sentence_score / len(SLN.semantic_elements), sentence))
    
    # 尝试根据长度再次筛选句子
    sorted_list = sorted(index_score_list, key=lambda t: t[1], reverse=True)
    return sorted_list[0][0]

def simplify_sentence(sentence, word_score_dict, threshold):
    simplified_token_sequence = []
    for word in sentence:
        if word_score_dict.get(word) >= threshold:
            simplified_token_sequence.append(word)
    return  simplified_token_sequence

def check_equivalence(SLN_1, SLN_2):
    return True
    return new_threshold

def decrease_redundancy(sln, word_score_dict, decay_rate=0.2):
    word_score_dict = deepcopy(word_score_dict)

    for element in sln.semantic_elements:
        if not is_node(element): continue
        
        for word in element.literal.split(" "):
            word_score_dict[word] *= decay_rate

    return word_score_dict

def increase_coherence(selected_SLN, SLNs, word_score_dict, decay_rate=0.2):
    word_score_dict = deepcopy(word_score_dict)
    nodes = [ele for ele in selected_SLN.semantic_elements if is_node(ele)]
    for node in nodes:
        for cnode in connected_nodes(node, SLNs):
            for word in cnode.split(" "):
                word_score_dict[word] /= decay_rate

    return word_score_dict