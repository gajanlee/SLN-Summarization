import nltk
import re
import string
from itertools import chain
from lxml import etree
from pathlib import Path
from sln import extract_relations_from_sentence, summarize, slns_to_neo4j, SLN
from nltk.stem import WordNetLemmatizer
from word_scoring import smi
lemmatizer = WordNetLemmatizer()


symbols = r"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
symbols += string.punctuation
symbols += "".join([chr(i) for i in range(945, 970)])
digits = "1234567890"

def segment_sentence(text):
    return nltk.sent_tokenize(text)


def paper_corpus(file_name="147.P14-1087.xhtml.txt"):
    base_path = Path("/media/lee/辽东铁骑/数据集/acl2014/RST_summary/data/acl2014/")
    abstract = (base_path / "abstract" / file_name).read_text()
    introduction, *sections, conclusion = (base_path / "content" / file_name).read_text().split("\n")

    return abstract, introduction, " ".join(sections), conclusion


def tokenize_sentences_and_words(text):
    sentences = []
    for sentence_ in nltk.sent_tokenize(text.lower().replace(".", ". ")):
        # sentence = sentence.lower().replace(".", ". ")
        sentence = re.sub(f"[{symbols+digits}]", " ", sentence_)
        sentence = "".join([char for char in sentence if 0 < ord(char) < 128])
        words = nltk.word_tokenize(sentence)
        words = list(map(lemmatizer.lemmatize, words))
        words = [word for word in words if len(word) > 1]
        if len(words) <= 3:
            continue

        sentences.append(words)
    return sentences


def run(file_name):
    abstract, introduction, section, conclusion = paper_corpus(file_name)

    abstract_sentences_tokens = tokenize_sentences_and_words(abstract)
    introduction_sentences_tokens = tokenize_sentences_and_words(introduction)
    section_sentences_tokens = tokenize_sentences_and_words(section)
    conclusion_sentences_tokens = tokenize_sentences_and_words(conclusion)

    score_dict = smi(
        list(chain(*(introduction_sentences_tokens + conclusion_sentences_tokens))),
        list(chain(*section_sentences_tokens)),
    )

    whole_slns, summary_slns = summarize(
        introduction_sentences_tokens + section_sentences_tokens + conclusion_sentences_tokens,
        score_dict,
    )
    abstract_slns = []
    for sentence_tokens in abstract_sentences_tokens:
        s = SLN(sentence_tokens)
        s.construct()
        abstract_slns.append(s)

    node_statements, relation_statements = slns_to_neo4j(whole_slns, summary_slns, abstract_slns)

    Path("./neo4j.txt").write_text(
        "\n".join([
            "MATCH (n)-[r]-() DELETE n,r",
            "\n".join(node_statements),
            "\n".join(relation_statements),
        ])
    )

    # print(sorted(score_dict.items(), key=lambda x: x[1]))


    # sentences = []
    # for sentence in nltk.sent_tokenize(introduction + " ".join(sections) + conclusion):
    #     sentence = sentence.lower().replace(".", ". ")
    #     words = nltk.word_tokenize(sentence)
    #     words = list(map(lemmatizer.lemmatize, words))
    #     sentences.append(words)

    # relations = extract_relations_from_sentence(sentences)
    # print(len(relations))


if __name__ == "__main__":
    run("100.P14-2103.xhtml.txt")
    # for file_path in Path("/media/lee/辽东铁骑/数据集/acl2014/RST_summary/data/acl2014/abstract").glob("*"):
    #     name = file_path.name
    #     print(name)
    #     run(name)