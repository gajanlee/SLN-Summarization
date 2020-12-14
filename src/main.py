from sln_summ.tokenizer import tokenize_sentences_and_words
from sln_summ.sln_construction import make_sln_noun_verb_pronoun, SLN, make_sln_noun_verb_pronoun_prep_adj_adv


text = "Existing researches of Semantic Link Network (SLN) in text space neglect the literal relations, reasoning and genre structure. This paper proposes a new text summarization approach based on the key word distribution within texts of different genre structures and an extraction of SLN from the literal text to reflect the basic semantics. The links are represented in logic to match the reasoning rules and derive out hidden links. The summarization algorithm involves the structure of SLN and generate summary under the interpretable criteria of sentence selection, sentence simplification and word score adjustment. Experimental summarizations on scientific papers, legal cases and news outperform baselines.".lower()

text = "Automatically constructing text Resource Space Model (RSM) is the foundational problem of the RSM theory. This paper proposed a method based on the hierarchical structure of the scientific literature to auto-matically construct the scientific RSM and demonstrated the feasibility of the hieratical structure in the text resources as the dimensions. The method has four steps: 1) extract hypernym-of relationships from the sci-entific literature resources; 2) construct the hieratical structure consists of a set of trees based on extracted hypernym-of relationships; 3) extract trees that meet the conditions of forming dimensions from the hierat-ical structure to make up resource space; 4) located scientific literature resources to the resource space and build the index. Besides, this paper analyzed the normal forms of the automatically constructed resource space. Experiments on three scientific literature data sets show that the extracted dimensions satisfy the definition of the dimensions in the RSM and the automatically constructed RSM is conducive to human understand-ing and research, and can support basic retrieval applications.".lower()
text = "Our neighbour, Captain Charles Alison, will sail from Portsmouth tomorrow. We'll meet him at the harbour early in the morning. He will be in his small boat, Topsail. Topsail is a famous little boat. It has sailed across the Atlantic many times. Captain Alison will set out at eight o'clock, so we'll have plenty of time. We'll see his boat and then we'll say goodbye to him. He will be away for two months. We are very proud of him. He will take part in an important race across the Atlantic. ".lower()

text = "Last week I went to the theatre. I had a very good seat. The play was very interesting. I did not enjoy it. A young man and a young woman were sitting behind me. They were talking loudly. I got very angry. I could not hear the actors. I turned round. I looked at the man and the woman angrily. They did not pay any attention. In the end, I could not bear it. I turned round again. ‘I can't hear a word!’ I said angrily."

text = "In this work, we revisit Shared Task 1 from the 2012 * SEM Conference: the automated analysis of negation. Unlike the vast majority of participating systems in 2012, our approach works over explicit and formal representations of propositional semantics, i.e., derives the notion of negation scope assumed in this task from the structure of logical-form meaning representations. We relate the task-specific interpretation of (negation) scope to the concept of (quantifier and operator) scope in mainstream underspecified semantics. With reference to an explicit encoding of semantic predicate-argument structure, we can operationalize the annotation decisions made for the 2012 * SEM task, and demonstrate how a comparatively simple system for negation scope resolution can be built from an off-the-shelf deep parsing system. In a system combination setting, our approach improves over the best published results on this task to date."

text = "The task organizers designed and documented an annotation scheme  and applied it to a little more than 100,000 tokens of running text by the novelist Sir Arthur Conan Doyle. Our system implements these findings through a notion of functor-argument ‘crawling’, using a sour starting point the under specified logical-form meaning representations provided by a general-purpose deep parser. They investigated two approaches for scope resolution, both of which were based on syntactic constituents. This system operates over the normalized semantic representations provided by the LinGO English Resource Grammar. MRS makes explicit predicate-argument relations, as well as partial information about scope (see below)."

sents = tokenize_sentences_and_words(text, remove_stop=False)
# slns = [make_sln_noun_verb_pronoun_adj_adv(words, verbose=False) for words in sents]

# slns = [make_sln_noun_verb_pronoun_adj(words, verbose=False) for words in sents]
# merged_sln1 = sum(slns, start=SLN([]))

# slns = [make_sln_noun_verb_pronoun(words, verbose=False) for words in sents]
# merged_sln2 = sum(slns, start=SLN([]))

# print(merged_sln1 - merged_sln2)
# print(merged_sln2 - merged_sln1)
# exit()

for func in (
    # make_sln_noun_verb, make_sln_noun_verb_pronoun, make_sln_noun_verb_pronoun_adj, make_sln_noun_verb_pronoun_adj_adv
    # make_sln_noun_verb_pronoun_adj_adv,
    make_sln_noun_verb_pronoun,
    make_sln_noun_verb_pronoun_prep_adj_adv,
):
    slns = [func(words, verbose=True) for words in sents]
    merged_sln = sum(slns, start=SLN([]))
    print(merged_sln)

    statements = merged_sln.to_neo4j_expression()
    with open(f"{func.__name__}.txt", "w") as f:
        f.write("\n".join(statements))