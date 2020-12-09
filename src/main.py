from sln_summ.tokenizer import tokenize_sentences_and_words
from sln_summ.sln_construction import make_sln_noun_verb_pronoun_adj_adv

text = "Our neighbour, Captain Charles Alison, will sail from Portsmouth tomorrow. We'll meet him at the harbour early in the morning. He will be in his small boat, Topsail. Topsail is a famous little boat. It has sailed across the Atlantic many times. Captain Alison will set out at eight o'clock, so we'll have plenty of time. We'll see his boat and then we'll say goodbye to him. He will be away for two months. We are very proud of him. He will take part in an important race across the Atlantic. ".lower()

x = tokenize_sentences_and_words(text, remove_stop=False)
print(x)

for words in x:
    sln = make_sln_noun_verb_pronoun_adj_adv(words)
    print(sln)
    exit()