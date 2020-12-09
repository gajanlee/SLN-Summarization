import pyrouge
import os
import shutil
import codecs
import logging


def rouge_perl(ref, hyp, log_name):
    """Compute the Rouge score based ROUGE-Toolkits 1.5.5
    
    params:
        :ref [[sent1, sent2], [sent1, sent2]], 每个doc分句子输入
        :hyp [[sent1, sent2], [sent1, sent2]]
        :log_name 模型名称，在result/{log_name}目录下生成
    """

    assert len(ref) == len(hyp)
    log_dir = f"result/{log_name}/"

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)

    for i in range(len(ref)):
        with codecs.open(log_dir+"%06d_reference.txt" % i, 'w', 'utf-8') as f:
            #f.write(" ".join(ref[i]).replace(' ', ' ') + '\n')
            f.write("\n".join(ref[i]))
        with codecs.open(log_dir+"%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            #f.write(" ".join(hyp[i]).replace(' ', ' ').replace('<unk>', 'UNK') + '\n')
            f.write("\n".join(hyp[i]))

    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_candidate.txt'
    r.model_dir = log_dir   # reference dir
    r.system_dir = log_dir  # hypothesis dir
    logging.getLogger('global').setLevel(logging.WARNING)
    rouge_results = r.convert_and_evaluate(
    #    rouge_args='-a -2 -1 -c 95 -U -n 2 -w 1.2 -b 75'
    )
    scores = r.output_to_dict(rouge_results)

    return {
        "rouge-1": {
            "f": round(scores["rouge_1_f_score"] * 100, 2),
            "r": round(scores["rouge_1_recall"] * 100, 2),
            "p": round(scores["rouge_1_precision"] * 100, 2),
        },
        "rouge-2": {
            "f": round(scores["rouge_2_f_score"] * 100, 2),
            "r": round(scores["rouge_2_recall"] * 100, 2),
            "p": round(scores["rouge_2_precision"] * 100, 2),
        },
        "rouge-l": {
            "f": round(scores["rouge_l_f_score"] * 100, 2),
            "r": round(scores["rouge_l_recall"] * 100, 2),
            "p": round(scores["rouge_l_precision"] * 100, 2),
        },
    }

    recall = [round(scores["rouge_1_recall"] * 100, 2),
              round(scores["rouge_2_recall"] * 100, 2),
              round(scores["rouge_l_recall"] * 100, 2)]
    precision = [round(scores["rouge_1_precision"] * 100, 2),
                 round(scores["rouge_2_precision"] * 100, 2),
                 round(scores["rouge_l_precision"] * 100, 2)]
    f_score = [round(scores["rouge_1_f_score"] * 100, 2),
               round(scores["rouge_2_f_score"] * 100, 2),
               round(scores["rouge_l_f_score"] * 100, 2)]
    print("F_measure: %s Recall: %s Precision: %s\n"
              % (str(f_score), str(recall), str(precision)))
    
    with codecs.open(log_dir+"rougeScore", 'w+', 'utf-8') as f:
        f.write("F_measure: %s Recall: %s Precision: %s\n"
              % (str(f_score), str(recall), str(precision)))
    return f_score[:], recall[:], precision[:]