from lxml import etree
from pathlib import Path
from tqdm import tqdm
import json

config = json.load(open((Path(__file__)).parent / "../required_data/config.json"))["corpus"]

PAPER_BASE_PATH = Path(config["paper_corpus"])
CNN_BASE_PATH = Path(config["cnn_corpus"])
LEGAL_BASE_PATH = Path(config["legal_corpus"])

def read_bbc_file_paths():
    return list(Path(config["bbc_corpus"]).glob("*.txt"))

def read_paper_corpus(limit=50):
    base_path = PAPER_BASE_PATH
    
    return [
        read_paper(base_path, file_path.name) for file_path in tqdm(list((base_path / "abstract").glob("*.txt"))[:limit], desc="paper-corpus")
    ]

def read_paper(base_path, file_name="147.P14-1087.xhtml.txt"):
    abstract = (base_path / "abstract" / file_name).read_text(encoding="utf-8")
    introduction, *sections, conclusion = (base_path / "content" / file_name).read_text(encoding="utf-8").split("\n")
    middle_text = ". ".join(sections)

    return abstract, introduction + middle_text + conclusion, introduction + ". " + conclusion, middle_text 

def read_cnn_corpus(limit=10000):
    base_path = CNN_BASE_PATH
    items = []

    for file_path in tqdm(list(base_path.glob("*.story"))[:limit], desc="cnn-corpus"):
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

    informative_cutoff = int(0.2 * len(story_lines))

    return ". ".join(highlight_lines), ". ".join(story_lines), ". ".join(story_lines[:informative_cutoff]), ". ".join(story_lines[informative_cutoff+1:])

def read_legal_corpus(limit=3000, *args, **kwargs):
    base_path = LEGAL_BASE_PATH
    items = []

    for file_path in tqdm(list((base_path / "citations_summ").glob("*.xml"))[:limit], desc="legal-corpus"):
        items.append(read_legal(
            base_path / "citations_summ" / file_path.name,
            base_path / "fulltext" / file_path.name,
            *args, **kwargs
        ))
    
    return items

def read_legal(citation_path, fulltext_path, truth_selection="all"):
    summ_tree = etree.HTML(citation_path.read_text(encoding="utf-8", errors="ignore"))
    full_tree = etree.HTML(fulltext_path.read_text(encoding="utf-8", errors="ignore"))

    summ_phrases = ". ".join(summ_tree.xpath("//citphrase//text()"))
    cite_phrases = ". ".join(full_tree.xpath("//catchphrase//text()"))
    sentences = full_tree.xpath("//sentence//text()")[:-2]  # Irrelevant sentence to text
    
    informative_cutoff = int(len(sentences) * 0.1)

    abstract = {
        "summ": summ_phrases,
        "cite": cite_phrases,
        "all": summ_phrases + ". " + cite_phrases,
    }[truth_selection]
    return (
        abstract,
        " ".join(sentences),
        " ".join(sentences[:informative_cutoff]),
        " ".join(sentences[informative_cutoff + 1:]),
    ) 

if __name__ == "__main__":
    read_paper_corpus()
    read_cnn_corpus()
    read_legal_corpus()
