import argparse
import json
import os
from os.path import join
from loguru import logger
from typing import List


def get_parser():
    parser = argparse.ArgumentParser()

    ## path manager
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--pretrained_cache_dir", type=str, default=None)

    ## training args
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--max_epoches", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=200000)

    ## predict
    parser.add_argument("--model_dir", type=str, default=None)

    ## batch size and device
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=512)

    ## device
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--gpu_id", type=int, default=0)

    ## optimizer
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-5, dest="learning_rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_decay_rate", type=float, default=0.2)
    parser.add_argument("--lr_decay_steps", type=float, default=5000)


    ## logging and save
    parser.add_argument("--save_steps", type=int, default=500)


    ## develop and debug
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--load_limit", type=int, default=0)

    ## task specified
    parser.add_argument("--select_methods", type=str, default='mean')
    parser.add_argument("--eval_num", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=2020)
    parser.add_argument("--use_crf", action='store_true')
    parser.add_argument("--k_folds", type=str, default=None,
                        help="""specified as 'dev_id/num_folds', where dev_id and num_folds are both Int, like 0/10、 
                        4/5.""")
    parser.add_argument("--topk", action="store_true")
    parser.add_argument("mode", type=str, default=None)

    return parser.parse_args()


class VersionConfig:
    '''
    本配置用于保存影响预测的项目参数
    1. 属性方式读取参数
    2. 保存模型参数时：json方式存储到本地  dump(path)
    3. 预测阶段：从本地加载json配置  load(path)
    6. 训练阶段：代码内部显示初始化配置 __init(kargs)__
    7. 属性可以扩展，并兼容旧版不完整属性
    '''

    def __init__(self,
                 encoder_model='voidful/albert_chinese_small',
                 max_seq_length=256,
                 use_crf=False,
                 k_folds=None
                 ):
        self.encoder_model = encoder_model
        self.max_seq_length = max_seq_length
        self.use_crf = use_crf
        self.k_folds = k_folds

    def load(self, cfg_dir):
        cfg_path = join(cfg_dir, 'version_config.json')
        if not os.path.exists(cfg_path):
            logger.warning("there is no version_config file, make sure being loading old version!")
        else:
            params = json.load(open(cfg_path, encoding='utf8'))
            self.__init__(**params)

    def dump(self, cfg_dir):
        params = {}
        for var in self.__dict__:
            if not var.startswith('_'):
                params[var] = getattr(self, var)
        json.dump(params, open(join(cfg_dir, 'version_config.json'), 'w', encoding='utf8'), ensure_ascii=False)


if __name__ == '__main__':
    # cfg = VersionConfig('a', 'b', 'c')
    # cfg.dump('./')
    # n_cfg = VersionConfig.load('./')
    args = get_parser()
    print(args.k_folds)
