import json
import nltk
import os
import math
import numpy as np
from tqdm import tqdm 
import torch
import argparse
import random
import ast 
import itertools 
import csv
from collections import Counter
from Levenshtein import ratio

def convert_l(l):
    if type(l) == list:
       return l
    else:
       return ast.literal_eval(l)

def check_dist(a,set_):
    ratios = [ratio(a,b) for b in set_]
    return max(ratios)


def intersection(lst1, lst2): 
     return [t for t in lst1 if check_dist(t,lst2) >= .95]

def prepare_results(p, r, f):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def add_template(rel, dim, kg_type='atomic'):
    if len(rel) == 0:
       rel = 'none.'
    if rel[-1] != '.':
       rel += '.'

    if 'xEffect' in dim: #yes
       return 'PersonX is likely: ' + rel #return 'PersonX ' + rel

    if 'oEffect' in dim: #yes
       return 'PersonY is likely: ' + rel #return 'PersonY ' + rel

    if 'xWant' in dim: #yes
       return 'PersonX wants: ' + rel #'PersonX will want to ' + rel
    if 'oWant' in dim: #yes
       return 'PersonY wants: ' + rel

    if 'xIntent' in dim: #yes
       return 'PersonX wanted: ' + rel #'The intent was ' + rel

    if 'oIntent' in dim:
       return 'PersonY wanted: ' + rel

    if 'xAttr' in dim: #yes
       return 'PersonX is seen as: ' + rel

    if 'xNeed' in dim: #yes
       return 'PersonX needed: ' + rel #'PersonX needs to ' + rel

    if 'xReact' in dim: #yes
       return 'PersonX then feels: ' + rel #'PersonX feels ' + rel

    if 'oReact' in dim: #yes
       return 'Others then feel: ' + rel #'PersonY feels ' + rel
    return rel

def reverse_template(rel):
    prefix = rel.split(':')[0]
    if 'PersonY/Others want' in prefix:
       return 'oWant'
    if 'PersonX wants' in prefix:
       return 'xWant'
    if 'PersonY/Others are likely' in prefix:
       return 'oEffect'
    if 'PersonY/Others then feel' in prefix:
       return 'oReact'
    if 'PersonX then feels' in prefix:
       return 'xReact'
    if 'PersonX is likely' in prefix:
       return 'xEffect'
    if 'PersonX is seen as' in prefix:
       return 'xAttr'
    if 'PersonX needed' in prefix:
       return 'xNeed'
    if 'PersonX wanted' in prefix:
       return 'xIntent'
random.seed(0)

parser = argparse.ArgumentParser(description='Evaluate bleu')
parser.add_argument('--model_type',type=str,default='mem')
parser.add_argument('--decoding',type=str,default='beam')
parser.add_argument('--input_dir',type=str,default='./gen_data/gpt2')
parser.add_argument('--data_dir',type=str,default='./data')
parser.add_argument('--kg_type',type=str,default='atomic')
parser.add_argument('--print_iter',type=int,default=50)
parser.add_argument('--ref',type=str,default='rel')
args = parser.parse_args()

dir_ = args.input_dir 
model_type = args.model_type
if model_type == 'comet_mem':
   model_name = 'mem_comet_beam_comet_outputs.jsonl'
elif model_type == 'comet_baseline':
   model_name = 'baseline_beam_comet_outputs.jsonl'
elif model_type != 'val' and model_type != 'bval':
   model_name = model_type + '_' + args.decoding + '_outputs.jsonl'
elif model_type == 'bval':
   model_name = 'comet-b1.jsonl'
else:
   model_name = 'comet-b10.jsonl'

original_data = open('./gold_set.jsonl')
original_data = [json.loads(l) for l in original_data.readlines()] 
data = [json.loads(l) for l in open(os.path.join(dir_, model_name)).readlines()]
dims = ["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]

training_rels_all = [l for l in csv.reader(open('./data/v4_atomic_all.csv'))][1:]
training_rels_all = [list(itertools.chain.from_iterable([convert_l(o) for o in l[1:-2]])) for l in training_rels_all]
training_rels_all = set(list(itertools.chain.from_iterable(training_rels_all)))


hyps = []
refs = []
hyps_tokenized = []
refs_tokenized = []
stories = []
dim_rels = []
nodim_rels = []
for l in original_data:
    stories.append(l['story'])
    if model_type == 'val' or model_type == 'bval' or model_type == 'gold_retrieval':
       d_ = [entry for entry in data if ' '.join([entry['sentence1'],entry['sentence2'],entry['sentence3'],entry['sentence4'],entry['sentence5']]) == l['story']][0]
    else:
       try:
          d_ = [entry for entry in data if entry['story'] == l['story']][0]
       except:
          import pdb; pdb.set_trace()
    dim = reverse_template(l['prefix'])
    dim_rels.append(dim)
    gold_rel = l['prefix'] + ' ' + l['rel']
    if model_type == 'gold_retrieval':
       gen_rel = d_["distance_supervision_relations"][str(l['sentID'])][dim]['relations'][-1]
    elif model_type == 'val': 
       gen_rel = [ast.literal_eval(r) for r in d_['distance_supervision_relations'][int(l['sentID'])][1][0][0][1:-1]][dims.index(dim)] 
    elif model_type == 'bval':
       gen_rel = [ast.literal_eval(r) for r in d_['distance_supervision_relations'][int(l['sentID'])][1][0][0][1:-1]][dims.index(dim)]
    else:
       gen_rel = d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][dims.index(dim)]
    gen_rel = [g for g in gen_rel if g.lower() != 'none' and g.lower() != 'none.']
    nodim_rels.extend(gen_rel)

print('num unique stories: ' + str(len(set(stories))))
distr = Counter(dim_rels)
print('distr: ' + str(distr))


nodim_rels = [l.lower() for l in nodim_rels]
training_rels_all = [l.lower() for l in training_rels_all]
training_rels_all = [l for l in training_rels_all if l != 'none' and l != 'none.']
novelty1 = intersection(nodim_rels,training_rels_all) 
novelty = float(len(set(nodim_rels))-len(set(novelty1)))/len(set(nodim_rels))
print('novelty w/ all: ' + str(novelty))
