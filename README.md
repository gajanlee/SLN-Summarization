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
|**Ours**     | **33.82**/6.33/16.73     | **24.67**/5.39/11.49 | **30.57**/**10.35**/**25.06** |

