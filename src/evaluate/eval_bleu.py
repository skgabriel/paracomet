import json
import os
import argparse
import random
import ast 
from collections import Counter
from nltk.translate.bleu_score import SmoothingFunction
from nltk import bleu
import numpy as np

def add_template(rel, dim, kg_type='atomic'):
    if len(rel) == 0:
       rel = 'none.'
    if rel[-1] != '.':
       rel += '.'

    if 'xEffect' in dim: 
       return 'PersonX is likely: ' + rel 

    if 'oEffect' in dim: 
       return 'PersonY is likely: ' + rel 

    if 'xWant' in dim: 
       return 'PersonX wants: ' + rel 

    if 'oWant' in dim: 
       return 'PersonY wants: ' + rel

    if 'xIntent' in dim: 
       return 'PersonX wanted: ' + rel 

    if 'oIntent' in dim:
       return 'PersonY wanted: ' + rel

    if 'xAttr' in dim: 
       return 'PersonX is seen as: ' + rel

    if 'xNeed' in dim:
       return 'PersonX needed: ' + rel 

    if 'xReact' in dim: 
       return 'PersonX then feels: ' + rel

    if 'oReact' in dim:
       return 'Others then feel: ' + rel
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
parser.add_argument('--model_type',type=str,default='baseline')
parser.add_argument('--decoding',type=str,default='beam')
parser.add_argument('--input_dir',type=str,default='./gen_data')
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
elif model_type == 'knn':
   model_name = 'knn.jsonl'
else:
   model_name = 'comet-b10.jsonl'
original_data = open('./gold_set.jsonl')
original_data = [json.loads(l) for l in original_data.readlines()] 
data = [json.loads(l) for l in open(os.path.join(dir_, model_name)).readlines()]
dims = ["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]


hyps = []
refs = []
stories = []
dim_rels = []
for l in original_data:
    stories.append(l['story'])
    if model_type == 'val' or model_type == 'bval':
       d_ = [entry for entry in data if ' '.join([entry['sentence1'],entry['sentence2'],entry['sentence3'],entry['sentence4'],entry['sentence5']]) == l['story']][0]
    else:
       d_ = [entry for entry in data if entry['story'] == l['story']][0]
    dim = reverse_template(l['prefix'])
    dim_rels.append(dim)
    gold_rel = add_template(l['rel'],dim)
    if model_type == 'val': 
       gen_rel = [ast.literal_eval(r) for r in d_['distance_supervision_relations'][int(l['sentID'])][1][0][0][1:-1]][dims.index(dim)] 
    elif model_type == 'bval':
       gen_rel = [ast.literal_eval(r) for r in d_['distance_supervision_relations'][int(l['sentID'])][1][0][0][1:-1]][dims.index(dim)]
    else:
       if model_type == 'knn':
          gen_rel = ast.literal_eval(d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][str(dims.index(dim))]) 
       else:
          gen_rel = d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][dims.index(dim)]
    gen_rel = [add_template(g, dim) for g in gen_rel]
    hyps.extend(gen_rel)
    refs.extend([gold_rel] * len(gen_rel))

print('total relations: ' + str(len(refs)))
print('num unique stories: ' + str(len(set(stories))))
distr = Counter(dim_rels)
print('distr: ' + str(distr))
import pdb; pdb.set_trace()
hyps = [tuple(h.split()) for h in hyps]
refs = [tuple(r.split()) for r in refs]
smoothing = SmoothingFunction().method1
weights = [0.5] * 2

bleu_scores1 = [bleu(refs, pred, weights=[1.0], smoothing_function=smoothing) for pred in hyps]
print(f"bleu1={100.0 * np.mean(bleu_scores1):.3f}")
bleu_scores2 = [bleu(refs, pred, weights=weights, smoothing_function=smoothing) for pred in hyps]
print(f"bleu2={100.0 * np.mean(bleu_scores2):.3f}")
