from lxml import etree
from pathlib import Path
from tqdm import tqdm

PAPER_BASE_PATH = Path("F:/数据集/acl2014/RST_summary/data/acl2014/")
CNN_BASE_PATH = Path("F:/codes/summarization/data/cnn")
LEGAL_BASE_PATH = Path("F:/codes/summarization/data/corpus")

def split_sentences(sentences, points):
    if len(points) <= 0: return sentences
    points.extend([0, 1])

    points = sorted(set(points))

    sentence_group = []
    sentence_len = len(sentences)
    for pre_point, after_point in zip(points[:-1], points[1:]):
        pre_point, after_point = map(int, 
            [sentence_len * pre_point, sentence_len * after_point])
        sentence_group.append(sentences[pre_point:after_point])

    return map(lambda lines: "".join(lines), sentence_group)

def read_paper_corpus():
    base_path = PAPER_BASE_PATH
    
    return [
        read_paper(base_path, file_path.name) for file_path in tqdm((base_path / "abstract").glob("*.txt"), desc="paper-corpus")
    ]

def read_paper(base_path, file_name="147.P14-1087.xhtml.txt"):
    abstract = (base_path / "abstract" / file_name).read_text(encoding="utf-8")
    introduction, *sections, conclusion = (base_path / "content" / file_name).read_text(encoding="utf-8").split("\n")

    return abstract, introduction, " ".join(sections), conclusion

def read_cnn_corpus(limit=10000):
    base_path = CNN_BASE_PATH
    items = []

    for file_path in tqdm(list((base_path / "story").glob("*.story"))[:limit], desc="cnn-corpus"):
        item = read_cnn(file_path)
        items.append(item)
    
    return items

def read_cnn(file_path):
    content = file_path.read_text(encoding="utf-8")
    highlight_index = content.find("@highlight")
    story, highlights = content[:highlight_index], content[highlight_index:].split("@highlight")

    story_lines = [line for line in story.split("\n") if line]
    highlight_lines = [line for line in highlights if line]

    story_lines = [line[9:] if line.startswith("(CNN) -- ") else line for line in story_lines]
    highlight_lines = [line[9:] if line.startswith("(CNN) -- ") else line for line in highlight_lines]

    return " ".join(highlight_lines), *split_sentences(story_lines, [0.5, 0.99])

def read_legal_corpus(limit=3000):
    base_path = LEGAL_BASE_PATH
    items = []

    for file_path in tqdm(list((base_path / "citations_summ").glob("*.xml"))[:limit], desc="legal-corpus"):
        items.append(read_legal(
            base_path / "citations_summ" / file_path.name,
            base_path / "fulltext" / file_path.name,
        ))
    
    return items

def read_legal(citation_path, fulltext_path):
    summ_tree = etree.HTML(citation_path.read_text(encoding="utf-8", errors="ignore"))
    full_tree = etree.HTML(fulltext_path.read_text(encoding="utf-8", errors="ignore"))

    summ_phrases = ". ".join(summ_tree.xpath("//citphrase//text()"))
    cite_phrases = ". ".join(full_tree.xpath("//catchphrase//text()"))
    sentences = full_tree.xpath("//sentence//text()")[:-2]  # Irrelevant sentence to text

    return f"{summ_phrases} {cite_phrases}", *split_sentences(sentences, [0.4, 0.8])

if __name__ == "__main__":
    read_paper_corpus()
    read_cnn_corpus()
    read_legal_corpus()