import json
import os
from os.path import join

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from reader.myReader import NERSet
from utils import ner
from utils.args import get_parser, VersionConfig
from utils.utils import clear_dir

args = get_parser()
VERSION_CONFIG = VersionConfig()
VERSION_CONFIG.load(args.model_dir)
GPU_IDS = [args.gpu_id]

if args.no_cuda or not torch.cuda.is_available():
    DEVICE = torch.device('cpu')
    logger.info('use cpu!')
    USE_CUDA = False
else:
    USE_CUDA = True
    DEVICE = torch.device('cuda', GPU_IDS[0])
    torch.cuda.set_device(DEVICE)
    logger.info('use gpu!')


def main(mode='dev'):
    assert mode in ['dev', 'test1', 'test2']
    logger.info(f"load model from {args.model_dir}")
    model = torch.load(join(args.model_dir, 'model.pth'), map_location=DEVICE)
    model.eval()

    # TODO 调试使用
    # from train import evaluate
    # devset = NERSet(args, VERSION_CONFIG, 'dev', True)
    # devloader = DataLoader(devset, batch_size=args.batch_size, collate_fn=NERSet.collate)
    # print(evaluate(model, devloader, debug=True))

    def predict():
        testset = NERSet(args, VERSION_CONFIG, mode)
        testloader = DataLoader(testset, batch_size=args.batch_size, collate_fn=NERSet.collate_fn)
        result = []
        with tqdm(total=len(testloader), ncols=50) as t:
            t.set_description(f'EVAL')
            for batch_data in testloader:
                if mode != 'test':
                    model_inputs, sample_infos, _ = batch_data
                else:
                    model_inputs, sample_infos = batch_data

                if USE_CUDA:
                    for k, v in model_inputs.items():
                        if isinstance(v, torch.Tensor):
                            model_inputs[k] = v.cuda(DEVICE)

                pred_tag_seq = model.predict(model_inputs)
                batch_decode_labels = ner.decode_pred_seqs(pred_tag_seq, sample_infos)

                for decode_labels, sample_info in zip(batch_decode_labels, sample_infos):
                    sample = {}
                    if mode=='test':
                        sample['id'] = sample_info['id']
                    sample['text'] = sample_info['text']
                    sample['label'] = decode_labels
                    result.append(sample)
                t.update(1)

        save_dir = join(args.model_dir, f'predict_{mode}')
        if os.path.exists(save_dir):
            clear_dir(save_dir)
        else:
            os.mkdir(save_dir)

        with open(join(save_dir, 'cluener_predict.json'), 'w', encoding='utf8') as f:
            for sample in result:
                f.write(json.dumps(sample, ensure_ascii=False)+'\n')

    predict()


if __name__ == '__main__':
    main('dev')