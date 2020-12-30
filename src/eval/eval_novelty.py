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

    if 'xEffect' in dim: 
       return 'PersonX is likely: ' + rel #return 'PersonX ' + rel

    if 'oEffect' in dim: 
       return 'PersonY is likely: ' + rel #return 'PersonY ' + rel

    if 'xWant' in dim: 
       return 'PersonX wants: ' + rel #'PersonX will want to ' + rel
    if 'oWant' in dim: 
       return 'PersonY wants: ' + rel

    if 'xIntent' in dim: 
       return 'PersonX wanted: ' + rel #'The intent was ' + rel

    if 'oIntent' in dim:
       return 'PersonY wanted: ' + rel

    if 'xAttr' in dim: 
       return 'PersonX is seen as: ' + rel

    if 'xNeed' in dim: 
       return 'PersonX needed: ' + rel #'PersonX needs to ' + rel

    if 'xReact' in dim: 
       return 'PersonX then feels: ' + rel #'PersonX feels ' + rel

    if 'oReact' in dim: 
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

parser = argparse.ArgumentParser(description='Evaluate novelty')
parser.add_argument('--decoded_file',type=str,default='../../data/gen_data/beam_outputs.jsonl')
parser.add_argument('--gold_file',type=str,default='../../data/gold_set.jsonl')
args = parser.parse_args()

original_data = open(args.gold_file)
original_data = [json.loads(l) for l in original_data.readlines()] 
data = [json.loads(l) for l in open(args.decoded_file).readlines()]
dims = ["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]

training_rels_all = [l for l in csv.reader(open('../../data/v4_atomic_all.csv'))][1:]
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
    d_ = [entry for entry in data if entry['story'] == l['story']]
    if len(d_) == 0:
       continue 
    d_ = d_[0]
    dim = reverse_template(l['prefix'])
    dim_rels.append(dim)
    gold_rel = l['prefix'] + ' ' + l['rel']
    gen_rel = d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][dims.index(dim)]
    gen_rel = [g for g in gen_rel if g.lower() != 'none' and g.lower() != 'none.']
    nodim_rels.extend(gen_rel)

print('num unique stories: ' + str(len(set(stories))))

nodim_rels = [l.lower() for l in nodim_rels]
training_rels_all = [l.lower() for l in training_rels_all]
training_rels_all = [l for l in training_rels_all if l != 'none' and l != 'none.']
novelty1 = intersection(nodim_rels,training_rels_all) 
novelty = float(len(set(nodim_rels))-len(set(novelty1)))/len(set(nodim_rels))
print('novelty w/ all: ' + str(novelty))
