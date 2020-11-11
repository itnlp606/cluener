import json
import os
import re
from os.path import join

import torch

from reader.tags import *
from torch.utils.data import Dataset, DataLoader
from thulac import thulac
import jieba

from utils.utils import get_tokenizer_cls, print_execute_time


class NERSet(Dataset):
    @print_execute_time
    def __init__(self, args, version_cfg, mode):
        assert mode in ['train', 'dev', 'test']

        self.args = args
        self.mode = mode
        self.samples = self._load_label_data()
        self.cfg = version_cfg

        pretrained_cache = join(args.pretrained_cache_dir, version_cfg.encoder_model)

        TokenizerCLS = get_tokenizer_cls(version_cfg.encoder_model)
        self.tokenizer = TokenizerCLS.from_pretrained(version_cfg.encoder_model,
                                                      cache_dir=pretrained_cache)

    def _load_label_data(self):
        data_path = join(self.args.data_dir, f'{self.mode}.json')
        samples = []
        with open(data_path, encoding='utf8') as f:
            for line in f.readlines():
                line = json.loads(line)
                char_list = list(line['text'])

                if self.mode != 'test':
                    label_list = ['O'] * len(char_list)

                    for label_type, entities in line['label'].items():
                        for tag_text, locs in entities.items():
                            for tag_begin, tag_end in locs:
                                tag_end += 1

                                assert char_list[tag_begin: tag_end] == list(tag_text)
                                label_list[tag_begin] = f'B-{label_type}'
                                for i in range(tag_begin + 1, tag_end):
                                    label_list[i] = f'I-{label_type}'
                    samples.append({'text': line['text'], 'input_chars': char_list,
                                    'labels_list': label_list, 'gold': line['label']})
                else:
                    samples.append({'id': line['id'], 'text': line['text'], 'input_chars': char_list})
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]

        text_b_word_head, text_b_word_tail = ['#'], ['#']  # avoid empty exception

        for w in jieba.cut_for_search(sample['text'].replace(' ', '')):
            if len(w) > 1 and not re.match(r'.*[0-9].*', w):
                text_b_word_head.append(w[0])
                text_b_word_tail.append(w[-1])

        encoded0 = self.tokenizer.encode_plus(text=sample['input_chars'],
                                              truncation=True,
                                              max_length=self.cfg.max_seq_length,
                                              is_split_into_words=True)

        loss_mask = encoded0['attention_mask']
        max_b_len = self.cfg.max_seq_length - sum(loss_mask) - 1
        encoded1 = self.tokenizer.encode_plus(text=sample['input_chars'],
                                              text_pair=text_b_word_head[:max_b_len],
                                              truncation=True,
                                              max_length=self.cfg.max_seq_length,
                                              is_split_into_words=True)
        encoded2 = self.tokenizer.encode_plus(text=sample['input_chars'],
                                              text_pair=text_b_word_tail[:max_b_len],
                                              truncation=True,
                                              max_length=self.cfg.max_seq_length,
                                              is_split_into_words=True)

        attention_mask = encoded1['attention_mask']
        len_a, len_b = sum(loss_mask), sum(attention_mask) - sum(loss_mask)
        token_type_ids = encoded1['token_type_ids']
        position_ids = [i for i in range(len_a)] + [0]*len_b

        assert len(encoded1['input_ids']) == len(encoded2['input_ids']) == len(attention_mask) == len(token_type_ids) == len(position_ids)

        sample_info = {
            'text': sample['text']
        }

        model_inputs = {
            'input_ids_1': encoded1['input_ids'],
            'input_ids_2': encoded2['input_ids'],
            'attention_mask': attention_mask,
            'loss_mask': loss_mask,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids
        }

        if self.mode != 'test':
            tag_ids = [TAG2ID['[CLS]']] + [TAG2ID[t] for t in sample['labels_list']] + [TAG2ID['[END]']]
            assert len(tag_ids) == sum(loss_mask)
            sample_info['gold'] = sample['gold']
            return model_inputs, sample_info, tag_ids
        else:
            sample_info['id'] = sample['id']
            return model_inputs, sample_info

    @staticmethod
    def collate_fn(batch: list):
        max_len_in_batch = max([len(s[0]['input_ids_1']) for s in batch])
        pad_lens = [max_len_in_batch - len(s[0]['input_ids_1']) for s in batch]
        pad_loss_lens = [max_len_in_batch - len(s[0]['loss_mask']) for s in batch]
        has_label = bool(len(batch[0]) == 3)

        _model_inputs = [s[0] for s in batch]
        sample_infos = [s[1] for s in batch]
        if has_label:
            tag_ids = [s[2] for s in batch]

        for i in range(len(batch)):
            for k in _model_inputs[i].keys():
                if k != 'loss_mask':
                    _model_inputs[i][k] += [0] * pad_lens[i]
                    _model_inputs[i][k] = torch.tensor(_model_inputs[i][k])
            if has_label:
                tag_ids[i] += [TAG2ID['[END]']] * pad_loss_lens[i]

        model_inputs = {}
        for k in _model_inputs[0].keys():
            if k == 'loss_mask':
                model_inputs[k] = [inputs[k] for inputs in _model_inputs]
            else:
                model_inputs[k] = torch.stack([inputs[k] for inputs in _model_inputs])

        if has_label:
            tag_ids = torch.tensor(tag_ids)
            return model_inputs, sample_infos, tag_ids
        else:
            return model_inputs, sample_infos


if __name__ == '__main__':
    class args:
        pretrained_cache_dir = 'pretrained'
        data_dir = './data'


    class cfg:
        encoder_model = 'clue/roberta_chinese_clue_tiny'
        max_seq_length = 64


    dataset = NERSet(args, cfg, 'dev')
    dataloader = DataLoader(dataset, batch_size=3, collate_fn=NERSet.collate_fn)
    for b in dataloader:
        print(b)
