# SLN-Summarization
A summarization algorithm based Semantic Link Network.

### Method and Files

* code/corpus.py
    - all corpus file
    - `config.ini` is used to config the dataset path

* code/eval_metric.py
    - ROUGE-1.5.5 toolkit
    - `rouge_perl` method

* code/psumm.py
    - Summarizer().summarize_item() is compared approaches

* notes/Structure_Analysis
    - Show the distribution of words in different corpus

* notes/SLN-new
    - Our approach and evaluation
    - `summ_item` method

### Evaluation

| System      | ACL2014                  | Legal                | CNN                       |
|    -        |                        - | -                    | -                         |
|First        | 28.37/4.98/17.99         | 23.86/**6.66**/12.84 | 24.3/10.09/21.43          |
|TextRank     | 32.55/**9.27**/**22.53** | 21.95/5.66/**13.47** | 28.26/9.4/24.38           |
|LexRank      | 33.03/8.66/16.17         | 22.72/5.79/13.1      | 26.5/9.1/21.76            |
|LSA          | 32.35/5.87/15.94         | 12.49/2.31/7.37      | 26.36/6.74/20.97          |
|KL           | 30.95/5.55/17.09         | 14.72/3.75/9.31      | 25.82/7.71/22.04          |
|Reinforce    | 32.81/-/-                | -                    | -                         |
|**Ours**     | **34.42**/8.33/21.73     | **25.67**/6.39/**14.49** | **30.57**/**10.35**/**25.06** |


### Graph

* From `P14-2103.xhtml`

* Full SLN
![avatar](./res/graph-final.png)

* Summary SLN
![avatar](./res/generated-final.png)


### Analysis

原文1616个词，生成摘要有10句/108词，标准摘要有82词。
原文包含：197节点、510个语义链。
摘要的SLN包含20个节点、42个语义链。

#### 1. 词分析

- 重合的词都是比较重要的词，按分布看是比较重要的词或是度数较高的词

| 词 | 引言次数 | 中间次数 | 结论次数 | 备注 |
| -- | -- | -- | -- | -- |
| approach | 5 | 6 | 1 | |
| topic | 16 | 17 | 2 | 度数很高，在图中看是核心词汇 |
| method | 5 | 4 | 3 | 评分0.88，很重要 |
| label | 9 | 25 | 2 | 评分0.6，但是度数高，容易被其他句子连带选中。一部分label作为动词在link中 |
| topic | 16 | 17 | 2 | |

- 未重合词：

| 词 | 引言次数 | 中间次数 | 结论次数 | 备注 |
| -- | -- | -- | -- | -- |
| performance | 1 | 5 | 0 | 显著性不够好，频率低 |
| state | 0 | 0 | 1 | 只在Conclusion出现一次 |
| engine | 1 | 1 | 1 | |
| search engine | 1 | 1 | 1 | search及短语search result被选中，engine重要度不足|
| topic keywords | 1 | 1 | 1 | 两个词都分别出现在摘要中，但是合起来的名词短语组合度较低。 |


#### 2. 生成SLN分析

108词包含20个节点、42个关系，说明多样性好，重复性冗余少。
生成的SLN是一个完整的图，说明句子之间主题连贯。


#### 3. 词选中的原因

词是否被选中与所在句子的整体质量也有关系，出现频率越高的词，即使不重要，但是整句被选中后，也会被加入到摘要中。