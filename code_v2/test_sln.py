from sln import SLN

def test_action_link():
    return ["Jack", "creates", "a", "big", "machine"]

def test_situation_link_verb():
    return ["Halen", "can", "dance", "at", "playgound"]

def test_negative_link():
    return "I'm not your father".split(" ")

def test_sequential_link():
    return "I and you".split(" ")

def test_condition_link():
    return "hike if weather is rainy".split(" ")

def test_part_of_link():
    return "It is a kind of flower".split(" ")

def test_similar_link():
    return "It is good like him".split(" ")

def test_means_link():
    return "I use a gun".split(" ")

def test_means_link2():
    return "go to scool by bus".split(" ")

def test_own_link():
    return "I have a brain".split(" ")

def test_own_link2():
    return "His brain".split(" ")

def test_situation_link_noun():
    return ["foot", "mark", "in", "the", "sand"]

def test_cause_effect_link():
    return ["foot", "mark", "because", "of", "the", "sand"]

def test_multi_relation_link():
    return ["foot", "mark", "to", "help", "the", "sand"]

def test_part_of_link2():
    return ["foot", "mark", "is", "a", "part", "of", "mark"]

def test_attribute_link():
    return ["Jack", "is", "good"]

def test_seperate_noun_link():
    return ["data", "from", "dbpedia", "to", "label", "topic"]


if __name__ == "__main__":
    for test_fn in [
        test_action_link,
        test_situation_link_verb,
        test_situation_link_noun,
        test_cause_effect_link,
        test_multi_relation_link,
        test_part_of_link,  # part 是个名词，调整next指针顺序，在外部next.
        test_attribute_link,    # 系动词后接形容词
        test_seperate_noun_link,
        test_negative_link,
        test_sequential_link,
        test_condition_link,
        test_part_of_link,
        test_similar_link,
        test_means_link,
        test_means_link2,
        test_own_link,
        test_own_link2,
    ]:
        words = test_fn()
        s = SLN(words)
        s.construct()
        print(test_fn.__name__)
        s.print_semantic_tuples()
        print( words, "=>", s.semantic_elements, "\n")