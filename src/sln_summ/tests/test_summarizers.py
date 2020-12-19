from pytest import approx
from sln_summ.tokenizer import tokenize_sentences_and_words
from sln_summ.summarizers import *
from sln_summ.sln_construction import *
from sln_summ.word_scoring import tf, tf_idf

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

def test_simplify_sentence():
    cases = [
        (
            ["I", "want", "a", "gun", "and", "gun", "."],
            {"I", "want", "gun", "what"},
            ["I", "want", "gun", "gun"],
        )
    ]

    for original_sentence, kept_word_set, simplified_sentence in cases:
        assert simplify_sentence(original_sentence, kept_word_set) == simplified_sentence

def test_find_inequivalence_words():
    cases = [
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
                ("guy", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good"),
            ]),
            SLN([
                ("guy", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good"),
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
            ]),
            [],
        ),
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
                ("guy", SemanticLink(ATTRIBUTE_LINK_NAME, "are"), "good"),
            ]),
            SLN([
                ("guy", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good"),
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
                ("mary", SemanticLink(ACTION_LINK_NAME, "love"), "jack"),
            ]),
            ["are"],
        ),
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "good guy"),
            ]),
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
            ]),
            ["good"]
        ),
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
            ]),
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "good guy"),
            ]),
            []
        ),
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "guy"),
            ]),
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "hit"), "good guy"),
            ]),
            []
        ),
    ]

    for original_sln, simplified_sln, words in cases:
        found_words = find_inequivalence_words(original_sln, simplified_sln)
        assert len(set(found_words)) == len(found_words)
        assert set(found_words) == set(words)

def test_decrease_redundancy():
    cases = [
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "like"), "swimming"),
                ("swimming", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "healthy"),
                ("healthy", SemanticLink(ATTRIBUTE_LINK_NAME, "are"), "healthy"),
                ("mary", SemanticLink(ACTION_LINK_NAME, "love"), "jack"),
            ]),
            {"jack": 10., "like": 20., "swimming": 30., "is": 9., "healthy": 3., "what": 6., "mary": 6., "love": 6., "are": 8.},
            0.3,
            {"jack": 7., "like": 14., "swimming": 21., "is": 6.3, "healthy": 2.1, "what": 6., "love": 4.2, "are": 5.6, "mary": 4.2},
        )
    ]
    
    for sln, word_score_dict, decay_rate, decreased_score_dict in cases:
        assert decrease_redundancy(sln, word_score_dict, decay_rate) == approx(decreased_score_dict)

def test_increase_coherence():
    cases = [
        (
            SLN([
                ("jack", SemanticLink(ACTION_LINK_NAME, "like"), "swimming"),
                ("swimming", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "healthy"),
            ]),
            SLN([
                ("jack", SemanticLink(ATTRIBUTE_LINK_NAME, "is"), "good"),
                ("healthy", SemanticLink(PART_OF_LINK_NAME, "kind of"), "healthy fitness"),
                ("good", SemanticLink(PART_OF_LINK_NAME, "kind of"), "love"),
                ("jack", SemanticLink(ACTION_LINK_NAME, "like"), "mary"),
                ("mary", SemanticLink(ACTION_LINK_NAME, "hate"), "john")
            ]),
            {"jack": 3., "like": 4., "swimming": 5., "is": 6., "healthy": 7., "good": 8., "kind": 9., "of": 10., "fitness": 11., "love": 12., "mary": 3., "hate": 4., "john": 5.},
            0.5,
            {"jack": 3., "like": 4., "swimming": 5., "is": 6., "healthy": 7., "good": 16., "kind": 18., "of": 20., "fitness": 22., "love": 12., "mary": 6., "hate": 4., "john": 5.},
        )
    ]
    for sln, merged_sln, word_score_dict, decay_rate, expected_score_dict in cases:
        merged_sln = merged_sln + sln
        assert increase_coherence(sln, merged_sln, word_score_dict, decay_rate) == expected_score_dict

def test_sln_summarize():
    cases = [
        "We introduce a novel modification to the standard projected subgradient dual decomposition algorithm for performing MAP inference subject to hard constraints to one for performing MAP in the presence of soft constraints.     In addition, we offer an easy-to-implement procedure for learning the penalties on soft constraints. This method drives many penalties to zero, which allows users to automatically discover discriminative constraints from     large families of candidates.We show via experiments on a recent substantial dataset that using soft constraints, and selecting which constraints to use with our penalty-learning procedure, can lead to significant gains in accuracy. We achieve a 17 gain in accuracy over a chain-structured CRF model, while only needing to run MAP in the CRF an average of less than 2 times per example. This minor incremental cost over Viterbi, plus the fact that we obtain certificates of optimality on 100 of our test examples in practice, suggests the usefulness of our algorithm for large-scale applications. We encourage further use of our Soft-DD procedure for other structured prediction problems."
    ]

    for passage in cases:
        sentences = tokenize_sentences_and_words(passage, remove_stop=False, lower=True)
        score_dict = tf(itertools.chain(*sentences))
        summary = sln_summarize(sentences, score_dict, strategies=["simplification", "diversity", "coherence"], desired_length=20, verbose=True)
        assert summary and type(summary) == str

def test_sln_simplification():
    cases = [
        (
            "I want an egg.",
            {"I": 1, "want": 0.5, "an": 0.1, "egg": 0.5, "place": 0},
            0.3,
            "I want egg.",
        ),
        (
            "I want a big egg.",
            {"I": 1, "want": 0.5, "a": 0.1, "big": 0.1, "egg": 0.5, "place": 0},
            0.3,
            "I want a big egg.",
        ),
        (
            "I want an egg.",
            {"I": 1, "want": 0.5, "an": 0.5, "egg": 0.5, "place": 0},
            0.3,
            "I want an egg.",
        ),
    ]

    for input_sentence, word_score_dict, threshold, expected_summary in cases:
        sentences = tokenize_sentences_and_words(input_sentence, remove_stop=False, lower=False)
        summary = sln_summarize(sentences, word_score_dict, ["simplification"], score_threshold=threshold, verbose=True)
        assert expected_summary == summary

def test_generated_summary_length_from_lib():
    cases = [
        "luhn",
        "lsa",
        "kl",
        "lexrank",
        "textrank",
        "textteaser",
    ]

    text = "We introduce a novel modification to the standard projected subgradient dual decomposition algorithm for performing MAP inference subject to hard constraints to one for performing MAP in the presence of soft constraints.     In addition, we offer an easy-to-implement procedure for learning the penalties on soft constraints. This method drives many penalties to zero, which allows users to automatically discover discriminative constraints from     large families of candidates.We show via experiments on a recent substantial dataset that using soft constraints, and selecting which constraints to use with our penalty-learning procedure, can lead to significant gains in accuracy. We achieve a 17 gain in accuracy over a chain-structured CRF model, while only needing to run MAP in the CRF an average of less than 2 times per example. This minor incremental cost over Viterbi, plus the fact that we obtain certificates of optimality on 100 of our test examples in practice, suggests the usefulness of our algorithm for large-scale applications. We encourage further use of our Soft-DD procedure for other structured prediction problems."

    for method in cases:
        summary = summarize_from_library(text, method)
        assert summary and type(summary) is str
