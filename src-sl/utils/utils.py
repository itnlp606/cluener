from numpy import mean
from datetime import datetime
import os

class CountSmooth:
    def __init__(self, max_steps):
        self.q = []
        self.max_steps = max_steps

    def get(self):
        return mean(self.q)

    def add(self, value):
        if len(self.q) > self.max_steps:
            self.q.pop(0)
        self.q.append(value)


def strftime():
    return datetime.now().strftime('%m-%d_%H-%M-%S')


def print_execute_time(func):
    '''函数执行时间装饰器'''
    from time import time

    def wrapper(*args, **kwargs):
        start = time()
        func_return = func(*args, **kwargs)
        end = time()
        print(f'{func.__name__}() execute time: {end - start}s')
        return func_return

    return wrapper


def clear_dir(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    for f in files:
        os.remove(f)

def get_tokenizer_cls(name):
    from transformers import BertTokenizer, AutoTokenizer
    bt_mapping = [
        'voidful/albert_chinese_tiny',
        'clue/roberta_chinese_clue_tiny'
    ]
    if name in bt_mapping:
        return BertTokenizer
    else:
        return AutoTokenizer

if __name__ == '__main__':
    print(strftime())
