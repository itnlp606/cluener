import json
from typing import List, Dict, Tuple, Set

from tqdm import tqdm


def decode_pred_seqs(batch_tag_paths: List[List[str]], sample_infos: List) -> List[Dict]:
    """
    从预测的tag序列解码
    """
    assert len(batch_tag_paths) == len(sample_infos)
    entitys = []
    for tags, info in zip(batch_tag_paths, sample_infos):
        text = info['text']

        cur_set = {}
        i = 0
        while i < len(tags):
            if tags[i].startswith('B-'):
                tag_type = tags[i][2:]
                tag_start = i
                tag_end = i
                i += 1
                while i < len(tags) and tags[i] == f'I-{tag_type}':
                    tag_end = i
                    i += 1
                tag_end += 1
                tag_start, tag_end = tag_start-1, tag_end-1  # bert cls标签占一个位置
                tag_text = text[tag_start: tag_end]
                cur_set[tag_type] = cur_set.get(tag_type, {})
                cur_set[tag_type][tag_text] = cur_set[tag_type].get(tag_text, [])
                cur_set[tag_type][tag_text].append([tag_start, tag_end-1])
            else:
                i += 1
        entitys.append(cur_set)
    return entitys

def get_f1_score_label(pre_lines, gold_lines, label="organization"):
    """
    打分函数
    """
    # pre_lines = [json.loads(line.strip()) for line in open(pre_file) if line.strip()]
    # gold_lines = [json.loads(line.strip()) for line in open(gold_file) if line.strip()]
    TP = 0
    FP = 0
    FN = 0
    for pre, gold in zip(pre_lines, gold_lines):
        pre = pre["label"].get(label, {}).keys()
        gold = gold["label"].get(label, {}).keys()
        for i in pre:
            if i in gold:
                TP += 1
            else:
                FP += 1
        for i in gold:
            if i not in pre:
                FN += 1
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    f = 2 * p * r / (p + r)
    return f


def get_f1_score(pre_file="ner_predict.json", gold_file="data/thuctc_valid.json"):
    pre_lines = [json.loads(line.strip()) for line in open(pre_file, encoding='utf8') if line.strip()]
    gold_lines = [json.loads(line.strip()) for line in open(gold_file, encoding='utf8') if line.strip()]
    f_score = {}
    labels = ['address', 'book', 'company', 'game', 'government', 'movie', 'name', 'organization', 'position', 'scene']
    sum = 0
    for label in labels:
        f = get_f1_score_label(pre_lines, gold_lines, label=label)
        f_score[label] = f
        sum += f
    avg = sum / len(labels)
    return f_score, avg


if __name__ == "__main__":
    f_score, avg = get_f1_score(pre_file="ner_predict_large.json", gold_file="data/thuctc_valid.json")

    print(f_score, avg)