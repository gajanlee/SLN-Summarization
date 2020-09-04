from sln import SLN

def test_action_link():
    return ["Jack", "creates", "a", "big", "machine"]

def test_situation_link_verb():
    return ["Halen", "can", "dance", "at", "playgound"]

def test_situation_link_noun():
    return ["foot", "mark", "in", "the", "sand"]

def test_cause_effect_link():
    return ["foot", "mark", "because", "of", "the", "sand"]

def test_multi_relation_link():
    return ["foot", "mark", "to", "help", "the", "sand"]

def test_part_of_link():
    return ["foot", "mark", "is", "a", "part", "of", "mark"]

def test_attribute_link():
    return ["Jack", "is", "good"]


if __name__ == "__main__":
    for test_fn in [
        test_action_link,
        test_situation_link_verb,
        test_situation_link_noun,
        test_cause_effect_link,
        test_multi_relation_link,
        test_part_of_link,  # part 是个名词，调整next指针顺序，在外部next.
        test_attribute_link,    # 系动词后接形容词
    ]:
        words = test_fn()
        s = SLN(words)
        s.construct()
        s.print_semantic_tuples()
        print(test_fn.__name__, words, "=>", s.semantic_elements, "\n")