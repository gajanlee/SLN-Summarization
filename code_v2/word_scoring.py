import math
from collections import Counter

log = lambda x: 0 if x == 0 else math.log(x)


def smi(informative_tokens, detailed_tokens, lamda=0.3, normalize=True, debug=False):
    all_tokens = informative_tokens + detailed_tokens
    all_counter = Counter(all_tokens)
    informative_counter = Counter(informative_tokens)
    detailed_counter = Counter(detailed_tokens)

    def _smi(token):
        p_x_yi = informative_counter.get(token, 0) / len(all_tokens)
        p_x_yd = detailed_counter.get(token, 0) / len(all_tokens)

        p_x = all_counter.get(token) / len(all_tokens)
        p_yi = len(informative_tokens) / len(all_tokens)
        p_yd = len(detailed_tokens) / len(all_tokens)

        return p_x_yi * log(p_x_yi / (p_x * p_yi)) - lamda * p_x_yd * log(p_x_yd / (p_x * p_yd))

    score_dict = {
        token: _smi(token) for token in all_counter
    }
    if normalize:
        max_score, min_score = max(score_dict.values()), min(score_dict.values())
        score_dict = {
            token: (value - min_score) / (max_score - min_score)
            for token, value in score_dict.items()
        }

    if debug:
        for word, _ in sorted(score_dict.items(), key=lambda x: x[1]):
            print(f"{word}\t{informative_counter.get(word, 0)}\t{detailed_counter.get(word, 0)}\t{all_counter.get(word, 0)}\t{score_dict[word]}")

        print(len(informative_tokens), len(detailed_tokens), len(all_tokens))

    return score_dict


def test_smi():
    informative_tokens = [
        "georgia", "mayor", "assails", "governor's", "move", "to", "reopen", "beaches"
    ]
    detailed_tokens = "the tybee island city council voted to close the beaches on march 20 but georgia gov brian kemp issued a statewide shelter-in-place executive order which supersedes all local orders relating to coronavirus and also opened up the state's beaches".split(" ")
    smi(informative_tokens, detailed_tokens, normalize=False)


if __name__ == "__main__":
    test_smi()