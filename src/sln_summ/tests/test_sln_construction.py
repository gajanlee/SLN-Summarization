from sln_summ.sln_construction import *

def test_make_sln_noun_verb():
    cases = [
        (
            "Jack smells the flower and a bird which danced in the window",
            SLN([
                ("Jack", SemanticLink(ACTION_LINK_NAME, "smells"), "flower"),
                ("flower", SemanticLink(SEQUENTIAL_LINK_NAME, "and"), "bird"),
                ("bird", SemanticLink(ACTION_LINK_NAME, "danced"), "window"),
            ]),
        ),
        (
            "Effel is at Paris",
            SLN([
                ("Effel", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "Paris"),
            ]),
        ),
        (
            "Jack is a good man",
            SLN([
                ("Jack", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "man"),
            ]),
        ),
        (
            "Jack is good",
            SLN([
                ("Jack", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good"),
            ]),
        ),
    ]

    for sentence, expected in cases:
        assert make_sln_noun_verb(sentence.split(" ")) == expected

def test_make_sln_noun_verb_pronoun():
    cases = [
        (
            "Jack smells the flower and a bird which danced in the window",
            SLN([
                ("Jack", SemanticLink(ACTION_LINK_NAME, "smells"), "flower"),
                ("flower", SemanticLink(SEQUENTIAL_LINK_NAME, "and"), "bird"),
                ("bird", SemanticLink(ACTION_LINK_NAME, "danced"), "window"),
                ("bird", SemanticLink(SITUATION_LINK_NAME, "in"), "window"),
            ]),
        ),
        (
            "Effel is at Paris",
            SLN([
                ("Effel", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "Paris"),
                ("Effel", SemanticLink(SITUATION_LINK_NAME, "at"), "Paris"),
            ]),
        )
    ]

    for sentence, expected in cases:
        assert make_sln_noun_verb_pronoun(sentence.split(" ")) == expected

def test_make_sln_noun_verb_pronoun_adj():
    cases = [
        (
            "Jack is a good man",
            SLN([
                ("Jack", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good man"),
                ("good man", SemanticLink(PART_OF_LINK_NAME, ""), "man"),
            ]),
        ),
    ]

    for sentence, expected in cases:
        assert make_sln_noun_verb_pronoun_adj(sentence.split(" ")) == expected

def test_make_sln_noun_verb_pronoun_adj_adv():
    cases = [
        (
            "Jack is a very good man",
            SLN([
                ("Jack", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "very good man"),
                ("very good man", SemanticLink(PART_OF_LINK_NAME, ""), "very good"),
                ("very good man", SemanticLink(PART_OF_LINK_NAME, ""), "good man"),
                ("very good", SemanticLink(PART_OF_LINK_NAME, ""), "good"),
                ("good man", SemanticLink(PART_OF_LINK_NAME, ""), "man"),
            ]),
        ),
        (
            "Jack is very good",
            SLN([
                ("Jack", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "very good"),
                ("very good", SemanticLink(PART_OF_LINK_NAME, ""), "good"),
            ]),
        ),
        (
            "Jack built the house last five years",
            SLN([
                ("Jack", SemanticLink(ACTION_LINK_NAME, "built", [SemanticLink(TEMPORAL_LINK_NAME, "last five years")]), "house"),
            ]),
        ),
        (
            "Jack and Wolf",
            SLN([
                ("Jack", SemanticLink(SEQUENTIAL_LINK_NAME, "and"), "Wolf")
            ])
        ),
        (
            "hike if the weather is rainy",
            SLN([
                ("hike", SemanticLink(CONDITION_LINK_NAME, "if"), "weather"),
                ("weather", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "rainy"),
            ]),
        ),
        (
            "it is a kind of flower",
            SLN([
                ("it", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "flower"),
                ("it", SemanticLink(PART_OF_LINK_NAME, "kind of"), "flower"),
            ])
        ),
        (
            "it is not good",
            SLN([
                ("it", SemanticLink(NEGATIVE_LINK_NAME, "is not"), "good")
            ])
        ),
        (
            "it is good like him",
            SLN([
                ("it", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good"),
                ("good", SemanticLink(SIMILAR_LINK_NAME, "like"), "him"),
            ])
        ),
        (
            "go to school by bus",
            SLN([
                (PLACEHOLDER_NODE_NAME, SemanticLink(ACTION_LINK_NAME, "go"), "school"),
                (PLACEHOLDER_NODE_NAME, SemanticLink(PURPOSE_LINK_NAME, "to"), "school"),
                ("school", SemanticLink(MEANS_LINK_NAME, "by"), "bus"),
            ])
        ),
        (
            "I have a brain",
            SLN([
                ("I", SemanticLink(OWN_LINK_NAME, "have"), "brain"),
            ]),
        ),
        (
            "foot mark in the sand",
            SLN([
                ("foot mark", SemanticLink(SITUATION_LINK_NAME, "in"), "sand"),
            ]),
        ),
        (
            "foot mark in the sand result in the accident",
            SLN([
                ("foot mark", SemanticLink(SITUATION_LINK_NAME, "in"), "sand"),
                ("sand", SemanticLink(CAUSE_EFFECT_LINK_NAME, "result in"), "accident"),
            ]),
        ),
        (
            "foot mark because of the sand",
            SLN([
                ("foot mark", SemanticLink(EFFECT_CAUSE_LINK_NAME, "because of"), "sand"),
            ]),
        ),
        (
            "foot mark to help the sand go to school",
            SLN([
                ("foot mark", SemanticLink(PURPOSE_LINK_NAME, "to"), "sand"),
                ("foot mark", SemanticLink(ACTION_LINK_NAME, "help"), "sand"),
                ("sand", SemanticLink(ACTION_LINK_NAME, "go"), "school"),
                ("sand", SemanticLink(PURPOSE_LINK_NAME, "to"), "school"),
            ])
        )
    ]

    for sentence, expected in cases:
        assert make_sln_noun_verb_pronoun_adj_adv(sentence.split(" ")) == expected

def test_sln_equal():
    cases = [
        (
            SemanticLink(ATTRIBUTE_LINK_NAME, "is"),
            SemanticLink(ATTRIBUTE_LINK_NAME, "is"),
            True,
        ),
        (
            SemanticLink(ATTRIBUTE_LINK_NAME, "is"),
            SemanticLink(ATTRIBUTE_LINK_NAME, "be"),
            False,
        ),
        (
            SemanticLink(ATTRIBUTE_LINK_NAME, "is"),
            SemanticLink(ACTION_LINK_NAME, "are"),
            False,
        )
    ]

    for link1, link2, result in cases:
        assert (link1 == link2) == result

def test_sln_merge():
    cases = [
        (
            SLN([
                ("Jack", SemanticLink(ACTION_LINK_NAME, "hates"), "Mary"),
            ]),
            SLN([
                ("Jack", SemanticLink(ACTION_LINK_NAME, "loves"), "Mary"),
                ("Hellen", SemanticLink(ACTION_LINK_NAME, "loves"), "Mary"),
            ]),
            SLN([
                ("Jack", SemanticLink(ACTION_LINK_NAME, "hates"), "Mary"),
                ("Jack", SemanticLink(ACTION_LINK_NAME, "loves"), "Mary"),
                ("Hellen", SemanticLink(ACTION_LINK_NAME, "loves"), "Mary"),
            ])
        )
    ]

    for sln_1, sln_2, merged_sln in cases:
        assert sln_1 + sln_2 == merged_sln