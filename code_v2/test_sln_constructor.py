import nltk
from sln_constructor import *

def test_normalize_word_pos_tag():
    words = ["Jack", "like", "the", "flower", "and", "a", "bird", "which", "danced", "in", "the", "window"]
    words = ["Jack", "is", "shit"]
    print(nltk.pos_tag(words))
    print(normalize_word_pos_tag(words))

def test_merge_neighbor_identical_tag():
    word_pos_tags = [("word1", "pos1"), ("word2", "pos1"), ("word3", "pos2"), ("word4", "pos3")]
    expected = [("word1 word2", "pos1"), ("word3", "pos2"), ("word4", "pos3")]

    if expected != merge_neighbor_identical_tag(word_pos_tags):
        print(merge_neighbor_identical_tag(word_pos_tags))

def test_make_sln_noun_verb():
    words = ["Jack", "smells", "the", "flower", "and", "a", "bird", "which", "danced", "in", "the", "window"]

    sln = make_sln_noun_verb(words)
    print(sln)

def test_merge_neighbor_clue_words():
    words = ["Jack", "like", "the", "flower", "and", "a", "bird", "which", "is", "the", "same", "as", "you"]

    word_pos_tags = nltk.pos_tag(words)

    x = merge_neighbor_clue_words(word_pos_tags, {
        "_SIM": [
            "same",
            "be the same as",
            "the same as",
            "same as"
        ],
    })
    print(x)

def test_make_sln_noun_verb_prep():
    words = "Jack is a kind of flower because of the kindle.".split(" ")
    sln = make_sln_noun_verb_prep(words)
    print(sln)


if __name__ == "__main__":
    # test_make_sln_noun_verb()
    test_merge_neighbor_clue_words()