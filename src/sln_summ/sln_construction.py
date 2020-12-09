import nltk

PLACEHOLDER_NODE_NAME = "_placeholder"

CO_OCCUR_LINK_NAME = "_CO_OCC"
ACTION_LINK_NAME = "_ACT"
PASSIVE_LINK_NAME = "_PASS"
NEGATIVE_LINK_NAME = "_NEG"
ATTRIBUTE_LINK_NAME = "_ATTR"
SEQUENTIAL_LINK_NAME = "_SEQ"
CONDITION_LINK_NAME = "_COND"
CAUSE_EFFECT_LINK_NAME = "_CE"
EFFECT_CAUSE_LINK_NAME = "_EC"
TEMPORAL_LINK_NAME = "_TEMP"
PART_OF_LINK_NAME = "_PARTOF"
SIMILAR_LINK_NAME = "_SIM"
PURPOSE_LINK_NAME = "_PUR"
MEANS_LINK_NAME = "_MEANS"
OWN_LINK_NAME = "_OWN"
SITUATION_LINK_NAME = "_SIT"

# Replace to regrex
link_clue_words = {
    "NODE": ["I", "you", "he", "we", "they", "she", "It", "him"],
    # Looking from sentiment lexicons
    NEGATIVE_LINK_NAME: ["not", "don't", "isn't", "is not", "am not", "are not", "arn't", "was not", "were not", "not", "do not", "no", "nothing", "never", "neither", "rarely", "seldom", "scarcely"],
    ATTRIBUTE_LINK_NAME: ["am", "is", "are", "feel", "sound", "hear", "keep", "remain", "look", "smell", "become", "grow", "fall", "get", "go", "come"],
    SEQUENTIAL_LINK_NAME: ["and", "then", "or", "as well as", "beside", "besides", "except"],
    CONDITION_LINK_NAME: ["if", "only if", "when", "unless", "while"],
    CAUSE_EFFECT_LINK_NAME: ["so", "therefore", "thus", "lead to", "attributable to", "result in"],
    EFFECT_CAUSE_LINK_NAME: ["because", "because of", "due to",],
    TEMPORAL_LINK_NAME: ["second", "minute", "hour", "day", "year", "month", "before", "after", "same time", "time", "seconds", "minutes", "hours", "days", "years", "months", "from"],
    PART_OF_LINK_NAME: ["kind of", "part of", "part", "kind", "such as", "subclass", "child"],
    SIMILAR_LINK_NAME: ["similar to", "like", "same as", "resemblance to", "resemble", "comprable with"],
    PURPOSE_LINK_NAME: ["for", "to", "in order to", "aim to", "so as to"],
    MEANS_LINK_NAME: ["by", "through", "via", "use", "make use of", "with"],
    OWN_LINK_NAME: ["of", "'s", "his", "their", "own", "her", "our", "your", "have", "has", "had"],
    SITUATION_LINK_NAME: ["in", "at"],
}

ALL_LINK_NAMES = list(link_clue_words.keys())

class SemanticLink:

    def __init__(self, link_name, literal, appendix=None):
        self.link_name = link_name
        self.literal = literal
        self.appendix_links = appendix if appendix else []
    
    def add_appendix(self, appendix_link):
        self.appendix_links.append(appendix_link)
    
    def __eq__(self, other):
        return self.link_name == other.link_name and self.literal == other.literal and other.appendix_links == other.appendix_links
    
    def __str__(self):
        string = f"{self.link_name}: {self.literal}"
        for link in self.appendix_links:
            string += f"({link})"
        return string
    
    def __repr__(self):
        return self.__str__()

transitive_reasoning_rules = {
    CO_OCCUR_LINK_NAME: {
        ACTION_LINK_NAME: ACTION_LINK_NAME,
    },
}

def is_link_name(name):
    return name in link_clue_words

def transitive_reasoning(link_1, link_2):
    pass

def part_of_reasoning(entity):
    pass

class SLN:

    def __init__(self, tuples):
        self.nodes = set()
        self.outgoing_links, self.incoming_links = {}, {}
        self.tuples = []

        self.update_tuples(tuples)

    def update_tuples(self, tuples):
        self.tuples.extend(tuples)
        
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

    @property
    def node_count(self):
        return len(self.nodes)

    def apply_reasoning(self):
        pass
    
    def __add__(self, other):
        # sln1 + sln2
        assert isinstance(other, SLN)

        merged_SLN = SLN(self.tuples + other.tuples)
        return merged_SLN
    
    def __eq__(self, other):
        return self.nodes == other.nodes and self.outgoing_links == other.outgoing_links
        # return self.tuples == other.tuples

    def __str__(self):
        strings = []
        for from_node in self.outgoing_links:
            for to_node in self.outgoing_links[from_node]:
                string = f"{from_node} -{str(self.outgoing_links[from_node][to_node])}> {to_node}"
                strings.append(string)

        return "\n".join(strings)

    def __repr__(self):
        return self.__str__()
    
    def to_neo4j_expression(self):
        relation_statements = []
        node_normalize_dict = {}
        for s in slns:
            for t in s.semantic_tuples:
                from_node_id, to_node_id = map(normalize_node_text, [t.from_node, t.to_node])
                node_normalize_dict[from_node_id] = t.from_node
                node_normalize_dict[to_node_id] = t.to_node
                link = t.link.lower()
                if link == "action":
                    link = f"action_{t.literal}"

                statement = f"CREATE ({from_node_id})-[:{link}]->({to_node_id})"
                relation_statements.append(statement)
        
        relation_statements = list(set(relation_statements))

        node_statements = []
        node_ids = set()
        for s in slns:
            for t in s.semantic_tuples:
                from_node_id, to_node_id = map(normalize_node_text, [t.from_node, t.to_node])
                node_ids.update([from_node_id, to_node_id])

        for node_id, node_text in node_normalize_dict.items():
            statement = f"CREATE ({node_id}:{node_tag} {{name: '{node_text}'}})"
            node_statements.append(statement)

        return node_statements, relation_statements

def normalize_word_pos_tag(words):
    def _normalize_word(word):
        # if word in ["am", "is", "are", "was", "were"]: return "be"
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
    return pos and pos.startswith("NN")

def is_verb(pos):
    return pos and pos.startswith("VB")

def is_passive_verb(pos):
    return pos and pos == "VBD"

def is_adj(pos):
    return pos and pos.startswith("JJ")

def is_pronoun(pos):
    return pos and pos.startswith("PRP")

def is_adv(pos):
    return pos and pos.startswith("RB")

def is_num(pos):
    return pos and pos == "CD"

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

def make_sln(words, candidate_link_names, funcs=None):
    funcs = funcs if funcs else []
    candidate_link_names += [SEQUENTIAL_LINK_NAME, PART_OF_LINK_NAME, TEMPORAL_LINK_NAME, NEGATIVE_LINK_NAME]

    word_pos_tags = normalize_word_pos_tag(words)
    word_pos_tags = merge_neighbor_clue_words(
        word_pos_tags,
        {l: link_clue_words[l] for l in ALL_LINK_NAMES if l in candidate_link_names}
    )
    word_pos_tags = merge_neighbor_identical_tag(word_pos_tags)

    print(word_pos_tags)

    candidate_tuples = []
    from_node, links, to_node, appendix = "", [], "", []
    i = 0
    while i < len(word_pos_tags):
        word, pos = word_pos_tags[i]
        next_word, next_pos = word_pos_tags[i+1] if i + 1 < len(word_pos_tags) else (None, None)
        next2_word, next2_pos = word_pos_tags[i+2] if i + 2 < len(word_pos_tags) else (None, None)
        if (is_be(word)):
            if is_passive_verb(next_pos):
                links.append(SemanticLink(PASSIVE_LINK_NAME, f"{word} {next_word}"))
                to_node = next_word
                i += 1
            elif is_adj(next_pos):
                links.append(SemanticLink(ATTRIBUTE_LINK_NAME, word))
                to_node = next_word
                i += 1
            else:
                links.append(SemanticLink(ATTRIBUTE_LINK_NAME, word))
            
        elif is_noun(pos) or is_pronoun(pos):
            if from_node and not links:
                links.append(SemanticLink(CO_OCCUR_LINK_NAME, ""))
            to_node = word
        elif pos == NEGATIVE_LINK_NAME:
            # appendix.append(SemanticLink(NEGATIVE_LINK_NAME, word))
            links.append(SemanticLink(NEGATIVE_LINK_NAME, word))
            if is_adj(next_pos):
                to_node = next_word
                i += 1
        elif is_verb(pos):
            link = SemanticLink(ACTION_LINK_NAME, word)
            links.append(link)
        elif is_adj(pos) and is_num(next_pos) and next2_pos == TEMPORAL_LINK_NAME:
            appendix.append(SemanticLink(TEMPORAL_LINK_NAME, f"{word} {next_word} {next2_word}"))
            # links.append(SemanticLink(TEMPORAL_LINK_NAME, f"{word} {next_word} {next2_word}"))
            i += 2
        elif (is_adj(pos) or is_num(pos)) and next_pos == TEMPORAL_LINK_NAME:
            appendix.append(SemanticLink(TEMPORAL_LINK_NAME, f"{word} {next_word} {next2_word}"))
            # links.append(SemanticLink(TEMPORAL_LINK_NAME, f"{word} {next_word}"))
            i += 1
            to_node = word
        elif pos == TEMPORAL_LINK_NAME:
            links.append(SemanticLink(TEMPORAL_LINK_NAME, f"{word}"))
        elif is_link_name(pos):
            links.append(SemanticLink(pos, word))
        else:
            for func in funcs:
                if ret := func(word, pos, next_word, next_pos, next2_word, next2_pos):
                    offset, _tuples, _to_node, _appendix = ret
                    i += offset
                    candidate_tuples.extend(_tuples)
                    to_node = _to_node
                    appendix.extend(_appendix)
                    break
            else:
                print("dropped:", word, pos)

        if not from_node and to_node:
            from_node = to_node
            to_node = ""

        if not from_node and not to_node and links:
            from_node = PLACEHOLDER_NODE_NAME
        
        if to_node or (i >= len(word_pos_tags) - 1 and from_node and links):
            for link in links:
                for app_link in appendix:
                    link.add_appendix(app_link)
                candidate_tuples.append((from_node, link, to_node))
            links = []
            from_node = to_node
            to_node = ""
            appendix = []
        elif (i >= len(word_pos_tags) - 1 and appendix):
            link = candidate_tuples[-1][1]
            for app_link in appendix:
                link.add_appendix(app_link)

        i += 1

    return SLN(candidate_tuples)

def make_sln_noun_verb(words):
    return make_sln(words, [])

def make_sln_noun_verb_pronoun(words, additional_links=None, additional_funcs=None):
    additional_links = additional_links if additional_links else []
    additional_funcs = additional_funcs if additional_funcs else []

    return make_sln(words, [
        CAUSE_EFFECT_LINK_NAME, EFFECT_CAUSE_LINK_NAME, PURPOSE_LINK_NAME, OWN_LINK_NAME, SITUATION_LINK_NAME,
        MEANS_LINK_NAME, CONDITION_LINK_NAME, SIMILAR_LINK_NAME
    ] + additional_links, additional_funcs)

def make_sln_noun_verb_pronoun_adj(words, additional_links=None, additional_funcs=None):
    additional_links = additional_links if additional_links else []
    additional_funcs = additional_funcs if additional_funcs else []

    def adj_noun_func(word, pos, next_word, next_pos, next2_word, next2_pos):
        if is_adj(pos) and is_noun(next_pos):
            return 1, [
                (f"{word} {next_word}", SemanticLink(PART_OF_LINK_NAME, ""), next_word),
            ], f"{word} {next_word}", []
    return make_sln_noun_verb_pronoun(words, [] + additional_links, additional_funcs=[adj_noun_func] + additional_funcs)

def make_sln_noun_verb_pronoun_adj_adv(words):
    def adv_adj_func(word, pos, next_word, next_pos, next2_word, next2_pos):
        if is_adv(pos):
            if is_adj(next_pos):
                if is_noun(next2_pos):
                    return 2, [
                        (f"{word} {next_word} {next2_word}", SemanticLink(PART_OF_LINK_NAME, ""), f"{word} {next_word}"),
                        (f"{word} {next_word} {next2_word}", SemanticLink(PART_OF_LINK_NAME, ""), f"{next_word} {next2_word}"),
                        (f"{word} {next_word}", SemanticLink(PART_OF_LINK_NAME, ""), next_word),
                        (f"{next_word} {next2_word}", SemanticLink(PART_OF_LINK_NAME, ""), next2_word),
                    ], f"{word} {next_word} {next2_word}", []
                else:
                    return 1, [
                        (f"{word} {next_word}", SemanticLink(PART_OF_LINK_NAME, ""), next_word)
                    ], f"{word} {next_word}", []
            else:
                return 0, [SemanticLink(CO_OCCUR_LINK_NAME, word)], "", []

    return make_sln_noun_verb_pronoun_adj(words, [], [adv_adj_func])