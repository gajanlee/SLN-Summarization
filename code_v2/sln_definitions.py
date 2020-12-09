

CO_OCCUR_LINK_NAME = "_CO_OCC"
ACTION_LINK_NAME = "_ACT"
PASSIVE_LINK_NAME = "_PASS"
NEGATIVE_LINK_NAME = "_NEG"
ATTRIBUTE_LINK_NAME = "_ATTR"
SEQUENTIAL_LINK_NAME = "_SEQ"
CONDITION_LINK_NAME = "_COND"
CAUSE_EFFECT_LINK_NAME = "_CE"
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
    NEGATIVE_LINK_NAME: ["not", "don't", "isn't", "is not", "not", "do not", "no", "nothing", "never", "neither", "rarely", "seldom", "scarcely"],
    ATTRIBUTE_LINK_NAME: ["am", "is", "are", "feel", "sound", "hear", "keep", "remain", "look", "smell", "become", "grow", "fall", "get", "go", "come"],
    SEQUENTIAL_LINK_NAME: ["and", "then", "or", "as well as", "beside", "besides", "except"],
    CONDITION_LINK_NAME: ["if", "only if", "when", "unless", "while"],
    CAUSE_EFFECT_LINK_NAME: ["because", "because of", "so", "therefore", "thus", "lead to", "due to", "insight to", "in view of", "in consideration of", "be responsible for", "be attributable to", "result in"],
    TEMPORAL_LINK_NAME: ["second", "minute", "hour", "day", "year", "month", "before", "after", "same time", "time"],
    PART_OF_LINK_NAME: ["kind of", "part of", "part", "kind", "such as", "subclass", "child"],
    SIMILAR_LINK_NAME: ["similar to", "like", "same as", "resemblance to", "resemble", "comprable with"],
    PURPOSE_LINK_NAME: ["for", "to", "in order to", "aim to", "so as to"],
    MEANS_LINK_NAME: ["by", "through", "via", "use", "make use of"],
    OWN_LINK_NAME: ["of", "'s", "his", "their", "own", "her", "our", "your", "have", "has", "had"],
    SITUATION_LINK_NAME: ["in", "at"],
}

ALL_LINK_NAMES = list(link_clue_words.keys())

class SemanticLink:

    def __init__(self, link_name, literal, appendix=[]):
        self.link_name = link_name
        self.literal = literal
        self.appendix_links = appendix
    
    def add_appendix(self, appendix_link):
        self.appendix_links.append(appendix_link)
    
    def __str__(self):
        string = f"{self.link_name}: {self.literal}"
        if self.appendix_links:
            string += f"{self.appendix_links}"
        return string
    
    def __repr__(self):
        return self.__str__()

def make_link(link_name, literal):
    return SemanticLink(link_name, literal)

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