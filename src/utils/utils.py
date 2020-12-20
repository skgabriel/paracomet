import os
import sys
import json
import time
import numpy as np
from tqdm import tqdm
import itertools
from typing import List
from distutils.dir_util import mkpath
import contextlib
import copy
import torch
import ftfy
import spacy
import re

def update_classification_losses(losses, nums, name, bs, loss):
    if not isinstance(loss, float):
        print(type(loss))
        raise

    nums[name] += bs

    losses[name] += loss * bs


def update_generation_losses(losses, nums, micro, macro, bs, length, loss):
    # Update Losses
    nums[macro] += bs

    if isinstance(length, int):
        update_indiv_generation_losses(
            losses, nums, micro, macro, bs, length, loss)
    else:
        update_tensor_generation_losses(
            losses, nums, micro, macro, bs, length, loss)


def update_indiv_generation_losses(losses, nums, micro,
                                   macro, bs, length, loss):
    nums[micro] += bs * length

    batch_loss = loss * bs

    losses[micro] += batch_loss
    losses[macro] += batch_loss / length


def update_tensor_generation_losses(losses, nums, micro,
                                    macro, bs, length, loss):
    nums[micro] += length.sum().item()

    losses[micro] += loss.sum().item()
    losses[macro] += (loss / length.float()).sum().item()

def load_existing_data_loader(data_loader, path):
    old_data_loader = torch.load(path)
    for attr in data_loader.__dict__.keys():
        if attr not in old_data_loader.__dict__.keys():
            continue
        setattr(data_loader, attr, getattr(old_data_loader, attr))


################################################################################
#
# Code Below taken from HuggingFace pytorch-openai-lm repository
#
################################################################################


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load(
            'en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v: k for k, v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(
                pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if (word[i] == first and i < len(word) - 1 and
                        word[i+1] == second):
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                text = self.nlp(text_standardize(ftfy.fix_text(text)))
                text_tokens = []
                for token in text:
                    text_tokens.extend(
                        [self.encoder.get(t, 0) for t in
                         self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens

def make_new_tensor_from_list(items, device_num, dtype=torch.float32):
    if device_num is not None:
        device = torch.device("cuda:{}".format(device_num))
    else:
        device = torch.device("cpu")
    return torch.tensor(items, dtype=dtype, device=device)


# is_dir look ast at whether the name we make
# should be a directory or a filename
def make_name(opt, prefix="", eval_=False, is_dir=True, set_epoch=None,
              do_epoch=True):
    string = prefix
    string += "{}-{}".format(opt.dataset, opt.exp)
    string += "/"
    string += "{}-{}-{}".format(opt.trainer, opt.cycle, opt.iters)
    string += "/"
    string += opt.model
    if opt.mle:
        string += "-{}".format(opt.mle)
    string += "/"
    string += make_name_string(opt.data) + "/"

    string += make_name_string(opt.net) + "/"
    string += make_name_string(opt.train.static) + "/"

    if eval_:
        string += make_name_string(opt.eval) + "/"
    # mkpath caches whether a directory has been created
    # In IPython, this can be a problem if the kernel is
    # not reset after a dir is deleted. Trying to recreate
    # that dir will be a problem because mkpath will think
    # the directory already exists
    if not is_dir:
        mkpath(string)
    string += make_name_string(
        opt.train.dynamic, True, do_epoch, set_epoch)
    if is_dir:
        mkpath(string)

    return string


def make_name_string(dict_, final=False, do_epoch=False, set_epoch=None):
    if final:
        if not do_epoch:
            string = "{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs)
        elif set_epoch is not None:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, set_epoch)
        else:
            string = "{}_{}_{}_{}".format(
                dict_.lr, dict_.optim, dict_.bs, dict_.epoch)

        return string

    string = ""

    for k, v in dict_.items():
        if type(v) == DD:
            continue
        if isinstance(v, list):
            val = "#".join(is_bool(str(vv)) for vv in v)
        else:
            val = is_bool(v)
        if string:
            string += "-"
        string += "{}_{}".format(k, val)

    return string


def is_bool(v):
    if str(v) == "False":
        return "F"
    elif str(v) == "True":
        return "T"
    return v


def generate_config_files(type_, key, name="base", eval_mode=False):
    with open("config/default.json".format(type_), "r") as f:
        base_config = json.load(f)
    with open("config/{}/default.json".format(type_), "r") as f:
        base_config_2 = json.load(f)
    if eval_mode:
        with open("config/{}/eval_changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)
    else:
        with open("config/{}/changes.json".format(type_), "r") as f:
            changes_by_machine = json.load(f)

    base_config.update(base_config_2)

    if name in changes_by_machine:
        changes = changes_by_machine[name]
    else:
        changes = changes_by_machine["base"]

    # for param in changes[key]:
    #     base_config[param] = changes[key][param]

    replace_params(base_config, changes[key])

    mkpath("config/{}".format(type_))

    with open("config/{}/config_{}.json".format(type_, key), "w") as f:
        json.dump(base_config, f, indent=4)


def replace_params(base_config, changes):
    for param, value in changes.items():
        if isinstance(value, dict) and param in base_config:
            replace_params(base_config[param], changes[param])
        else:
            base_config[param] = value


def initialize_progress_bar(data_loader_list):
    num_examples = sum([len(tensor) for tensor in
                        data_loader_list.values()])
    return set_progress_bar(num_examples)


def set_progress_bar(num_examples):
    bar = tqdm(total=num_examples)
    bar.update(0)
    return bar


def merge_list_of_dicts(L):
    result = {}
    for d in L:
        result.update(d)
    return result


def return_iterator_by_type(data_type):
    if isinstance(data_type, dict):
        iterator = data_type.items()
    else:
        iterator = enumerate(data_type)
    return iterator


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def flatten(outer):
    return [el for inner in outer for el in inner]


def zipped_flatten(outer):
    return [(key, fill, el) for key, fill, inner in outer for el in inner]


def remove_none(l):
    return [e for e in l if e is not None]


# Taken from Jobman 0.1
class DD(dict):
    def __getattr__(self, attr):
        if attr == '__getstate__':
            return super(DD, self).__getstate__
        elif attr == '__setstate__':
            return super(DD, self).__setstate__
        elif attr == '__slots__':
            return super(DD, self).__slots__
        return self[attr]

    def __setattr__(self, attr, value):
        # Safety check to ensure consistent behavior with __getattr__.
        assert attr not in ('__getstate__', '__setstate__', '__slots__')
#         if attr.startswith('__'):
#             return super(DD, self).__setattr__(attr, value)
        self[attr] = value

    def __str__(self):
        return 'DD%s' % dict(self)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memo):
        z = DD()
        for k, kv in self.items():
            z[k] = copy.deepcopy(kv, memo)
        return z


def convert_nested_dict_to_DD(obj):
    if type(obj) == dict:
        new_obj = DD(obj)
        for k, v in obj.items():
            new_DD = convert_nested_dict_to_DD(v)
            new_obj[k] = new_DD
        return new_obj
    else:
        return obj


def convert_DD_to_nested_dict(obj):
    if type(obj) == DD:
        x = {}
        for k, v in obj.items():
            x[k] = dictify(v)
        return x
    else:
        return obj

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
