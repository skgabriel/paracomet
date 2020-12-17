import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
import itertools
from typing import List

def convert_example(y, encoder):
    text = [[encoder.encoder[y.split('|>')[0].replace(' ','') + '|>']] + encoder.convert_tokens_to_ids(encoder.tokenize(x)) for x in y.split('<|r|>')[1:-1]]
    scores = [float(x) for x in y.split('<|s|>')[1:]]
    return text, scores

def convert_list(list_, encoder):
    vals = [convert_example(y,encoder) for y in list_]
    return list(itertools.chain.from_iterable([v[0] for v in vals])),list(itertools.chain.from_iterable([v[1] for v in vals]))

def encode_dataset2(*splits, encoder):
    encoded_splits = []
    for split in splits:
        fields = []
        field_t = 0
        for field in split:
            if isinstance(field[0], str):
                if field_t == 0:
                   special = [[encoder.encoder['<|' + x.split('<|')[1].replace(' ','')],encoder.encoder['<|' + x.split('<|')[2].replace(' ','')]] for x in field]
                   field = [encoder.convert_tokens_to_ids(encoder.tokenize(field[i].split('<|')[0])) + special[i] for i in range(len(field))]
                else:
                   field = [encoder.convert_tokens_to_ids(encoder.tokenize(field[i])) for i in range(len(field))]
            if field_t == 2:
                field = [[convert_list(x[key],encoder) for key in x.keys()] for x in field]
                field_vals = [list(itertools.chain.from_iterable([s[0] for s in x])) for x in field]
                field_scores = [list(itertools.chain.from_iterable([s[1] for s in x])) for x in field]
                field = [(field_vals[i],field_scores[i]) for i in range(len(field))]
            fields.append(field)
            field_t += 1
        encoded_splits.append(fields)
    return encoded_splits

def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f

class ResultLogger(object):
    def __init__(self, path, *args, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log = open(make_path(path), 'w')
        self.f_log.write(json.dumps(kwargs)+'\n')

    def log(self, **kwargs):
        if 'time' not in kwargs:
            kwargs['time'] = time.time()
        self.f_log.write(json.dumps(kwargs)+'\n')
        self.f_log.flush()

    def close(self):
        self.f_log.close()

def iter_data(*datas, n_batch=128, truncate=False, verbose=False, max_batches=float("inf")):
    n = len(datas[0])
    if truncate:
        n = (n//n_batch)*n_batch
    n = min(n, max_batches*n_batch)
    n_batches = 0
    if verbose:
        f = sys.stderr
    else:
        f = open(os.devnull, 'w')
    for i in tqdm(range(0, n, n_batch), total=n//n_batch, file=f, ncols=80, leave=False):
        if n_batches >= max_batches: raise StopIteration
        if len(datas) == 1:
            yield datas[0][i:i+n_batch]
        else:
            yield (d[i:i+n_batch] for d in datas)
        n_batches += 1

def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]

def write_items(items: List[str], output_file):
    with open(output_file, 'w') as f:
        for item in items:
            f.write(str(item) + "\n")
    f.close()
