import nltk
import re
from collections import Counter
import string
from itertools import chain
from lxml import etree
from pathlib import Path
from copy import deepcopy
from nltk.stem import WordNetLemmatizer
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
stopwords = (Path(__file__).parent / "../required_data/stopwords.txt").read_text().split("\n")
lemmatizer = WordNetLemmatizer()

symbols = r"！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."
symbols += string.punctuation + "\\"
symbols += "".join([chr(i) for i in range(945, 970)])
digits = "1234567890"

def segment_sentence(text):
    return nltk.sent_tokenize(text)

def tokenize_sentences_and_words(text, remove_stop=True, lower=False, cut_words=True):
    if lower: text = text.lower()
    sentences = []
    for sentence_ in nltk.sent_tokenize(text.replace(".", ". ")):
        # sentence = sentence.lower().replace(".", ". ")
        sentence = re.sub(f"[{symbols+digits}]", " ", sentence_)
        sentence = "".join([char for char in sentence if 0 < ord(char) < 128])
        words = nltk.word_tokenize(sentence)
        words = list(map(lemmatizer.lemmatize, words))
        # words = [word for word in words if len(word) > 1]
        
        if remove_stop:
            words = [word for word in words if word not in stopwords]

        if len(words) <= 3:
            continue

        if cut_words:
            sentences.append(words)
        else:
            sentences.append(" ".join(words) + ".")
    return sentences
