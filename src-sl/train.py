import os
from os.path import join

import torch
from torch.optim import Adam, swa_utils, AdamW
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from transformers import set_seed

from reader.tags import *
from utils.args import get_parser, VersionConfig
from utils.optim import get_linear_schedule_with_warmup, get_cycle_schedule
from utils.utils import strftime, CountSmooth
from model.myModel import BertNER
from reader.myReader import NERSet
from utils import ner

args = get_parser()
assert not args.use_crf and not args.k_folds
VERSION_CONFIG = VersionConfig(
    max_seq_length=args.max_seq_length,
    encoder_model=args.model_name_or_path,
    use_crf=args.use_crf,
    k_folds=args.k_folds
)

GPU_IDS = args.gpu_id
if args.k_folds:
    OUTPUT_DIR = join(args.output_dir, args.k_folds.split('/')[0])
else:
    OUTPUT_DIR = join(args.output_dir, strftime())

if args.no_cuda or not torch.cuda.is_available():
    USE_CUDA = False
    DEVICE = torch.device('cpu')
else:
    USE_CUDA = True
    DEVICE = torch.device('cuda', GPU_IDS)
print('using', str(DEVICE))


def evaluate(model, devloader, debug=False):
    model.eval()
    score = {}
    for label_type in LABEL_TYPES:
        score[label_type] = {'tp': 0, 'fp': 0, 'fn': 0}
    print()
    with tqdm(total=len(devloader), ncols=100) as t:
        t.set_description(f'EVAL')
        for model_inputs, sample_infos, _ in devloader:
            for k, v in model_inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(DEVICE, non_blocking=True)
            pred_tag_seq = model.predict(model_inputs)

            batch_decode_labels = ner.decode_pred_seqs(pred_tag_seq, sample_infos)
            for decode_labels, sample_info in zip(batch_decode_labels, sample_infos):
                for label_type in LABEL_TYPES:
                    for lb in sample_info['gold'].get(label_type, {}).keys():
                        if not lb in decode_labels.get(label_type, {}).keys():
                            if debug:
                                print('FN:', lb)
                            score[label_type]['fn'] += 1
                        else:
                            score[label_type]['tp'] += 1
                    for lb in decode_labels.get(label_type, {}).keys():
                        if not lb in sample_info['gold'].get(label_type, {}).keys():
                            if debug:
                                print('FP:', lb)
                            score[label_type]['fp'] += 1
                    if debug:
                        print(pred_tag_seq[0])
        t.update(len(devloader))
    model.train()
    for label_type in LABEL_TYPES:
        tp, fp, fn = score[label_type]['tp'], score[label_type]['fp'], score[label_type]['fn']
        score[label_type]['P'] = tp / (tp + fp + 1e-7)
        score[label_type]['R'] = tp / (tp + fn + 1e-7)
        score[label_type]["F1"] = 2 * (score[label_type]['P'] * score[label_type]['R']) / (score[label_type]['P'] + score[label_type]['R'] + 1e-7)

    avg_P = sum([s['P'] for s in score.values()])/len(score)
    avg_R = sum([s['R'] for s in score.values()])/len(score)
    avg_F1 = sum([s['F1'] for s in score.values()])/len(score)
    return avg_P, avg_R, avg_F1


def main():
    logger.info(f"output dir is: {OUTPUT_DIR}")
    set_seed(args.random_seed)
    model = BertNER(args, VERSION_CONFIG)

    if USE_CUDA:
        model = model.cuda(DEVICE)

    trainset = NERSet(args, VERSION_CONFIG, 'train')
    devset = NERSet(args, VERSION_CONFIG, 'dev')

    devloader = DataLoader(devset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False,
                           collate_fn=NERSet.collate_fn, pin_memory=True)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers,
                             shuffle=True, collate_fn=NERSet.collate_fn, pin_memory=True)

    T = 4
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_cycle_schedule(optimizer, T)

    global_step = 0
    loss_ = CountSmooth(100)

    swa_model = swa_utils.AveragedModel(model, device=DEVICE)

    for epoch in range(args.max_epoches):
        print()
        with tqdm(total=len(trainloader), ncols=100) as t:
            t.set_description(f'Epoch {epoch}')
            model.train()
            for model_inputs, sample_infos, label_ids in trainloader:
                for k, v in model_inputs.items():
                    if isinstance(v, torch.Tensor):
                        model_inputs[k] = v.to(DEVICE, non_blocking=True)
                label_ids = label_ids.to(DEVICE, non_blocking=True)
                global_step += 1
                optimizer.zero_grad()
                loss = model(model_inputs, label_ids)
                loss_.add(loss.item())

                loss.backward()
                optimizer.step()

                t.set_postfix(loss=loss_.get())
                t.update(1)

                scheduler.step()

            # eval and save model every epoch
            p, r, f1 = evaluate(model, devloader)
            logger.info(f"after {epoch + 1} epoches,  percision={p}, recall={r}, f1={f1}\n")

        if epoch % T == T-1:
            swa_model.update_parameters(model)
            p,r,f1 = evaluate(swa_model.module, devloader)
            logger.info(f"swa: {epoch + 1}epoches,  percision={p}, recall={r}, f1={f1}\n")

            if epoch > T:
                save_dir = join(OUTPUT_DIR, f'epoch_{epoch}')
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                with open(join(save_dir, 'evaluate.txt'), 'w') as f:
                    f.write(f'precision={p}, recall={r}, f1={f1} dev_size={len(devset)}\n')
                    f.write(f'batch_size={args.batch_size}, epoch={epoch}')
                torch.save(swa_model.module, join(save_dir, 'model.pth'))
                VERSION_CONFIG.dump(save_dir)
                with open(f'{OUTPUT_DIR}/args.txt', 'w') as f:
                    f.write(str(args))

if __name__ == '__main__':
    main()
