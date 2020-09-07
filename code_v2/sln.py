import nltk
import numpy as np
from collections import namedtuple


SemanticElement = namedtuple("SemanticElement", ["element_type", "literal"])
SemanticTuple = namedtuple("SemanticTuple", ["from_node", "to_node", "link", "literal"])


class SLN:

    def __init__(self, words):
        self.words = words
        self.word_pos_tags = nltk.pos_tag(words)
        self.curr_index = 0

    def construct(self):
        self.elements = self.identify_semantic_elements()
        self.tuples = self.construct_tuples(self.semantic_elements)

    @property
    def semantic_elements(self):
        if not hasattr(self, "elements"):
            raise ValueError("Please call the function of construct first")
        return self.elements

    @property
    def semantic_tuples(self):
        return self.tuples
    
    def print_semantic_tuples(self):
        for t in self.semantic_tuples:
            print(f"{t.from_node} --{t.link}--> {t.to_node}")

    def is_noun(self, pos_tag):
        return pos_tag.startswith("NN")

    def noun_handler(self):
        node_words = [self.current()[0]]

        while (word_pos := self.next()):
            if self.is_noun(word_pos[1]):
                node_words.append(word_pos[0])
            else:
                break

        return SemanticElement("NODE", " ".join(node_words))

    def verb_handler(self):
        action_word = self.current()[0]
        self.next()
        return SemanticElement("ACTION", action_word)

    def identify_clue_words(self):
        link_word_mapper = {
            "Attribute": ["am", "is", "are", "feel", "sound"],
            "Sequential": ["and", "then", "or", "as well as"],
            "Condition": ["if", "onlyif", "when", "unless"],
            "Cause_effect": ["because", "because of", "so", "therefore", "thus", "lead to", "due to", "insight to", "in view of", "in consideration of", "be responsible for", "be attributable to", "result in"],
            # "Effect-cause": [],
            "Part_of": ["kind of", "part of"],
            "Similar": ["similar to", "like", "same as", "resemblance to", "resemble", "comprable with"],
            "Purpose": ["for", "to", "in order to", "aim to", "so as to"],
            "Means": ["by", "through", "via"],
            "Own": ["of"],
            "Situation": ["in", "at"],
        }

        word_link_mapper = {}
        for link, words in link_word_mapper.items():
            for word in words:
                word_link_mapper[word] = link
        
        word = self.current()[0]
        for i in range(3):
            matched_words = [key for key in word_link_mapper.keys() if key.startswith(word)]

            if len(matched_words) == 0:
                break

            if (word_pos := self.peek(i + 1)):
                peek_matched_words = [key for key in word_link_mapper.keys() if key.startswith(word + " " + word_pos[0])]
            else:
                peek_matched_words = []

            if len(matched_words) != 0 and len(peek_matched_words) == 0:
                for matched in matched_words:
                    if matched == word:
                        for _ in range(i+1):
                            self.next()

                        link = word_link_mapper[word]
                        if link == "Attribute" and (adj := self.current()):
                            if adj[1] == "JJ":
                                self.next()
                                return [
                                    SemanticElement(link, word),
                                    SemanticElement("NODE", adj[0]),
                                ]

                        return SemanticElement(link, word)
            
            if self.peek(i + 1):
                word += " " + self.peek(i+1)[0]
            else:
                break

    def identify_semantic_elements(self):
        elements = []
        while (word_pos := self.current()):
            word, pos = word_pos
            if not (element := self.identify_clue_words()):
                if self.is_noun(pos):
                    element = self.noun_handler()
                elif pos.startswith("VB"):
                    element = self.verb_handler()
                else:
                    self.next()
            if element:
                if type(element) in [list, tuple]:
                    elements.extend(element)
                else:
                    elements.append(element)
        
        return elements

    
    def current(self):
        if self.curr_index >= len(self.word_pos_tags):
            return None
        return self.word_pos_tags[self.curr_index]

    def next(self):
        self.curr_index += 1
        if self.curr_index + 1 >= len(self.word_pos_tags):
            return None

        return self.word_pos_tags[self.curr_index]

    def peek(self, offset=1):
        if self.curr_index + offset >= len(self.word_pos_tags):
            return None
        return self.word_pos_tags[self.curr_index + offset]


    def construct_tuples(self, elements):
        from_node, links = "", []
        tuples = []
        for element in elements:
            if not from_node and element.element_type != "NODE":
                continue
            
            if element.element_type == "NODE":
                if from_node and links:
                    for link in links:
                        tuples.append(SemanticTuple(from_node, element.literal, link[0], link[1]))
                elif from_node:
                    tuples.append(SemanticTuple(from_node, element.literal, "SEQUENTIAL", ""))

                from_node = element.literal
                links = []
                    
            else:
                links.append((element.element_type, element.literal))

        if from_node and links and element.element_type != "NODE":
            for link in links:
                tuples.append(SemanticTuple(from_node, from_node, link[0], link[1]))
        return tuples


def merge_sln_tuples(slns):
    merged_semantic_relations = {}

    for sln in slns:
        for t in sln.semantic_tuples:
            merged_semantic_relations[t.from_node] = merged_semantic_relations.get(t.from_node, {})

            link = t.link if t.link != "ACTION" else t.literal

            merged_semantic_relations[t.from_node][t.to_node] = merged_semantic_relations[t.from_node].get(t.to_node, []) + [link]

    return merged_semantic_relations


def extract_relations_from_sentence(sentences):
    slns = []
    for words in sentences:
        s = SLN(words)
        s.construct()
        slns.append(s)

    relations = merge_sln_tuples(slns)
    return relations


def summarize(sentences, score_dict):
    summary_slns = []
    summary_sentences = []

    slns = []
    for sentence_words in sentences:
        s = SLN(sentence_words)
        s.construct()
        slns.append(s)
    whole_slns = slns
    relations = extract_relations_from_sentence(sentences)

    for _ in range(10):
        score_list = []
        for s in slns:
            score = 0
            for element in s.semantic_elements:
                element_score = 0
                for word in element.literal.split(" "):
                    element_score += score_dict[word]
                if element.literal not in relations:
                    element_score *= 1
                else:
                    element_score *= len(relations[element.literal])
                
                score += element_score
            score /= len(s.words)
            score_list.append(score)

        max_indice = np.argmax(score_list)

        selected_sln = slns[max_indice]
        summary_slns.append(selected_sln)
        summary_sentences.append(sentences[max_indice])

        slns = slns[:max_indice] + slns[max_indice + 1:]
        sentences = sentences[:max_indice] + sentences[max_indice + 1:]

        alpha = 1.5

        for element in selected_sln.semantic_elements:
            # Increase Diversity
            for word in element.literal.split(" "):
                score_dict[word] /= alpha
            
            # Increase Coherence
            if element.literal in relations:
                for element_literal in relations[element.literal]:
                    for word in element_literal.split(" "):
                        score_dict[word] *= alpha

    return whole_slns, summary_slns, summary_sentences


def normalize_node_text(node_text):
    return "_".join(node_text.split(" "))


def slns_to_neo4j(slns, node_tag):
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


def summarization_slns_to_neo4j(slns, summary_slns, abstract_slns):
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

    summary_node_ids = set()
    for s in summary_slns:
        for t in s.semantic_tuples:
            from_node_id, to_node_id = map(normalize_node_text, [t.from_node, t.to_node])
            summary_node_ids.update([from_node_id, to_node_id])

    abstract_node_ids = set()
    for s in abstract_slns:
        for t in s.semantic_tuples:
            from_node_id, to_node_id = map(normalize_node_text, [t.from_node, t.to_node])
            abstract_node_ids.update([from_node_id, to_node_id])
    
    print(f"The overlap nodes count between summary and abstract is {len(abstract_node_ids & summary_node_ids)}")
    print(f"The overlap nodes is {abstract_node_ids & summary_node_ids}")
    # print(f"The summary nodes count is {len(summary_node_ids)}")
    # print(f"The summary nodes count is {len(summary_node_ids & set(node_normalize_dict.keys()))}")

    for node_id, node_text in node_normalize_dict.items():
        if node_id not in summary_node_ids and node_id not in abstract_node_ids:
            statement = f"CREATE ({node_id}:Node {{name: '{node_text}'}})"
        elif node_id in summary_node_ids and node_id not in abstract_node_ids:
            statement = f"CREATE ({node_id}:Summary {{name: '{node_text}'}})"
        elif node_id not in summary_node_ids and node_id in abstract_node_ids:
            statement = f"CREATE ({node_id}:Abstract {{name: '{node_text}'}})"
        else:
            statement = f"CREATE ({node_id}:Both {{name: '{node_text}'}})"

        node_statements.append(statement)

    return node_statements, relation_statements
