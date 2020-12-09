import nltk
from sln_definitions import *

class SLN:

    def __init__(self, tuples):
        self.nodes = set()
        self.outgoing_links, self.incoming_links = {}, {}
        self.tuples = []

        self.update_tuples(tuples)

    def update_tuples(self, tuples):
        for from_node, link, to_node in tuples:
            self.nodes.update([from_node, to_node])

            if from_node not in self.outgoing_links:
                self.outgoing_links[from_node] = {}
            if to_node not in self.incoming_links:
                self.incoming_links[to_node] = {}

            if to_node not in self.outgoing_links[from_node]:
                self.outgoing_links[from_node][to_node] = []
            if from_node not in self.incoming_links[to_node]:
                self.incoming_links[to_node][from_node] = []
            
            self.outgoing_links[from_node][to_node].append(link)
            self.incoming_links[to_node][from_node].append(link)

    @property
    def link_count(self):
        pass

    def apply_reasoning(self):
        pass
    
    def to_neo4j_input(self):
        pass

    def __add__(self, other):
        # sln1 + sln2
        assert isinstance(other, SLN)

        self.update_tuples(other.tuples)

    def __str__(self):
        strings = []
        for from_node in self.outgoing_links:
            for to_node in self.outgoing_links[from_node]:
                string = f"{from_node} -{str(self.outgoing_links[from_node][to_node])}> {to_node}"
                strings.append(string)

        return "\n".join(strings)

    def __repr__(self):
        return self.__str__()

def make_sln_noun_verb(words):
    return make_sln(words, [])

def make_sln_noun_verb_prep(words):
    return make_sln(words, [
        CONDITION_LINK_NAME, CAUSE_EFFECT_LINK_NAME, PART_OF_LINK_NAME,
        PURPOSE_LINK_NAME, MEANS_LINK_NAME, SITUATION_LINK_NAME,
    ])

def make_sln_noun_verb_prep_adj(words):
    pass

def make_sln_noun_verb_prep_adj_adv(words):
    pass

def normalize_word_pos_tag(words):
    def _normalize_word(word):
        if word in ["am", "is", "are", "was", "were"]:
            return "be"
        return word

    def _normalize_pos(pos):
        if pos.startswith("NN"): return "NN"
        # if pos.startswith("VB"): return "VB"
        return pos

    word_pos_tag = [
        (_normalize_word(word), _normalize_pos(pos))
        for word, pos in nltk.pos_tag(words)
    ]
    return word_pos_tag

def is_be(word):
    return word in ["be", "am", "is", "are", "was", "were"]

def is_noun(pos):
    return pos.startswith("NN")

def is_verb(pos):
    return pos.startswith("VB")

def is_passive_verb(pos):
    return pos == "VBD"

def is_adj(pos):
    return pos == "JJ"

def is_pronoun(pos):
    return pos.startswith("PRP")

def merge_neighbor_identical_tag(word_pos_tags):
    # word_pos_tags: [(word1, pos1), (word2, pos1), (word3, pos2), ...]
    # Return [("word1 word2", pos1), (word3, pos2), ...]
    new_word_pos_tags = []

    candidate_phrase = ""
    for i, (word, pos) in enumerate(word_pos_tags):
        if i < len(word_pos_tags) - 1 and pos == word_pos_tags[i+1][1]:
            candidate_phrase += f"{word} "
        else:
            candidate_phrase += f"{word} "
            new_word_pos_tags.append((candidate_phrase.strip(), pos))
            candidate_phrase = ""
    
    return new_word_pos_tags

def merge_neighbor_clue_words(word_pos_tags, link_words_mapper):
    word_link_mapper = {}
    for link, words in link_words_mapper.items():
        for word in words:
            word_link_mapper[word] = link
    
    new_word_pos_tags = []
    
    i = 0
    while i < len(word_pos_tags):
        for j in [3, 2, 1]:
            word = " ".join([word_pos[0] for word_pos in word_pos_tags[i:i+j]])
            if word in word_link_mapper:
                new_word_pos_tags.append((word, word_link_mapper[word]))
                i += j
                break
        else:
            new_word_pos_tags.append((word, word_pos_tags[i][1]))
            i += 1

    return new_word_pos_tags

def make_sln(words, candidate_link_names):
    word_pos_tags = normalize_word_pos_tag(words)
    word_pos_tags = merge_neighbor_identical_tag(word_pos_tags)
    word_pos_tags = merge_neighbor_clue_words(
        word_pos_tags,
        {l: link_clue_words[l] for l in ALL_LINK_NAMES if l in candidate_link_names}
    )

    candidate_tuples = []
    from_node, links, to_node, appendix = "", [], "", []
    i = 0
    while i < len(word_pos_tags):
        word, pos = word_pos_tags[i]
        next_word, next_pos = word_pos_tags[i+1] if i + 1 < len(word_pos_tags) else (None, None)
        if (is_be(word) ) and is_adj(next_pos):
            links.append(make_link(ATTRIBUTE_LINK_NAME, word))
            to_node = next_word
            i += 1
        elif is_noun(pos) or is_pronoun(pos):
            if from_node and not links:
                links.append(make_link(CO_OCCUR_LINK_NAME, ""))
                to_node = word
        elif pos == NEGATIVE_LINK_NAME:
            appendix.append(make_link(NEGATIVE_LINK_NAME, word))
        elif is_verb(pos):
            link = make_link(PASSIVE_LINK_NAME, word) if is_passive_verb(pos) else make_link(ACTION_LINK_NAME, word)
            links.append(link)
        elif is_link_name(pos):
            links.append(make_link(pos, word))
        else:
            print("dropped:", word, pos)

        if not from_node and to_node:
            from_node = to_node
            to_node = ""

        if to_node or (i >= len(word_pos_tags) - 1 and from_node and links):
            for link in links:
                for app_link in appendix:
                    link.add_appendix(app_link)
                candidate_tuples.append((from_node, link, to_node))
            links = []
            from_node = to_node
            to_node = ""
            appendix = []
        
        i += 1

    return SLN(candidate_tuples)