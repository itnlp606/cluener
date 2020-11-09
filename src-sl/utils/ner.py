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

