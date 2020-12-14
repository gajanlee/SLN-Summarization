from sln_summ.summarizers import *
from sln_summ.sln_construction import *


def test_select_sentence():
    cases = [
        (
            [
                ["jack", "have", "a", "dream"],
                ["jack", "have", "a", "big", "dream"],
            ],
            [
                SLN([("jack", SemanticLink(ACTION_LINK_NAME, "have"), "dream")]),
                SLN([("jack", SemanticLink(ACTION_LINK_NAME, "have"), "big dream")])
            ],
            {"jack": 1, "have": 1, "a": 1, "dream": 1, "do": 2, "jack": 2, "big": 3},
            True,
            1
        ),
        (
            [["jack", "like", "dog"], ["jack", "like", "handsome", "dog"]],
            [
                SLN([("jack", SemanticLink(ACTION_LINK_NAME, "like"), "dog")]),
                SLN([("jack", SemanticLink(ACTION_LINK_NAME, "like"), "dog")]),
            ],
            {"jack": 1, "like": 1, "dog": 1, "handsome": 1},
            True,
            0,
        ),
    ]

    for sentences, slns, word_score_dict, normalized, index in cases:
        candidate_index = select_sentence(sentences, slns, word_score_dict, normalized)
        assert candidate_index == index

def test_generated_summary_length_from_lib():
    pass
