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
import csv
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn

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

# Just to make it a bit more readable
WN_NOUN = 'n'
WN_VERB = 'v'
WN_ADJECTIVE = 'a'
WN_ADJECTIVE_SATELLITE = 's'
WN_ADVERB = 'r'
 
def convert(word, from_pos, to_pos):    
    """ Transform words given from/to POS tags """
 
    synsets = wn.synsets(word, pos=from_pos)
 
    # Word not found
    if not synsets:
        return word
 
    # Get all lemmas of the word (consider 'a'and 's' equivalent)
    lemmas = [l for s in synsets
                for l in s.lemmas 
                if s.name.split('.')[1] == from_pos
                    or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                        and s.name.split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
 
    # Get related forms
    derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]
 
    # filter only the desired pos (consider 'a' and 's' equivalent)
    related_noun_lemmas = [l for drf in derivationally_related_forms
                             for l in drf[1] 
                             if l.synset.name.split('.')[1] == to_pos
                                or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
                                    and l.synset.name.split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
 
    # Extract the words from the lemmas
    words = [l.name for l in related_noun_lemmas]
    len_words = len(words)
 
    # Build the result in the form of a list containing tuples (word, probability)
    result = [(w, float(words.count(w))/len_words) for w in set(words)]
    result.sort(key=lambda w: -w[1])
 
    # return all the possibilities sorted by probability
    return result[0][0]

def normalize_rel(rel, dim, kg_type='atomic',pronoun='They'):
    pronouns = ['i','me','you','she','her','he','him','it','we','us','they','them']
    rel_ = rel.lower()
    if rel_.split(' ')[0] in pronouns:
       rel_ = ' '.join(rel_.split(' ')[1:])
    if len(rel_) == 0:
       rel_ = 'none.'
    if not rel_.endswith('.'):
       rel_ += '.'
    #subj = [token for token in text if token.dep == 'nsubj'][0]
    if 'xEffect' in dim: #yes
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' are likely ' #'The first person is likely ' #return 'PersonX ' + rel
       else: 
          root = pronoun + ' is likely ' 
#       root = pronoun + ' are likely ' #'The first person is likely ' #return 'PersonX ' + rel

    if 'oEffect' in dim: #yes
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' are likely ' #'The second person is likely ' #return 'PersonY ' + rel
       else: 
          root = pronoun + ' is likely '  

    if 'xWant' in dim: #yes
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' want ' #'The first person wants ' #'PersonX will want to ' + rel
       else: 
          root = pronoun + ' wants ' 

    if 'oWant' in dim: #yes
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       root = pronoun + ' want ' #'The second person wants '

    if 'xIntent' in dim: #yes
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' wanted ' #"The first person wanted " #'The intent was ' + rel
       else: 
          root = pronoun + ' wanted ' 

    if 'xAttr' in dim: #yes
       if rel_.startswith('to'):
          rel_ = convert(rel_, WN_VERB, WN_ADJECTIVE)
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' are seen as ' #'The first person is seen as '
       else: 
          root = pronoun + ' is seen as ' 

    if 'xNeed' in dim: #yes
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' needed ' #'The second person needs ' #'PersonX needs to ' + rel
       else: 
          root = pronoun + ' needed ' 

    if 'xReact' in dim: #yes
       if rel_.startswith('to'):
          rel_ = convert(rel_, WN_VERB, WN_ADJECTIVE)
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' then feel ' #'The first person then feels ' #'PersonX feels ' + rel
       else: 
          root = pronoun + ' then feels ' 

    if 'oReact' in dim: #yes
       if rel_.startswith('to'):
          rel_ = convert(rel_, WN_VERB, WN_ADJECTIVE)
       root = 'The others then feel ' #'The others then feel ' #'PersonY feels ' + rel

    return root + rel_


random.seed(0)

parser = argparse.ArgumentParser(description='make tsv')
parser.add_argument('--model_type',type=str,default='mem_comet')
parser.add_argument('--decoding',type=str,default="beam")
parser.add_argument('--input_dir',type=str,default='../../testing_code/modeling/gen_data/gpt2')
parser.add_argument('--output_file',type=str, default='val_gold')
parser.add_argument('--data_dir',type=str,default='../data')
parser.add_argument('--split',type=int,default=0) #0,1,2
args = parser.parse_args()

output_file = 'dev_examples/' + args.output_file + "_" + args.model_type + "_" + args.decoding + "_" + str(args.split) + ".tsv"

dir_ = args.input_dir 
model_type = args.model_type
kg_type = 'atomic'
if model_type == 'mem_comet':
   model_name = 'mem_comet_beam_outputs.jsonl'
elif model_type == 'baseline_comet':
   model_name = 'baseline_comet_beam_outputs.jsonl'
elif model_type == 'gold_retrieval':
   model_name = '/net/nfs2.corp/ca-data/data_comet/h_atomic_val.jsonl'
elif model_type != 'val' and model_type != 'bval':
   model_name = model_type + "_" + args.decoding + '_outputs.jsonl'
elif model_type == 'bval':
   model_name = '/net/nfs2.corp/ca-data/updated_gen_data/comet-b1.jsonl'
elif model_type == 'knn':
   model_name = './knn_beam_outputs.jsonl'
else:
   model_name = '/net/nfs2.corp/ca-data/updated_gen_data/comet-b10.jsonl'
original_data = open('../../testing_code/modeling/gold_set.jsonl')
original_data = [json.loads(l) for l in original_data.readlines()]
data = [json.loads(l) for l in open(os.path.join(dir_, model_name)).readlines()]
dims = ["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]

tsv_writer = csv.writer(open(output_file,'w'),delimiter='\t')
hyps = []
sents = []
stories = []
dim_rels = []
tsv_writer.writerow(['story','relation','sent_id'])
stories = []
for l in original_data:
    if model_type == 'val' or model_type == 'bval' or model_type == 'gold_retrieval':
       d_ = [entry for entry in data if ' '.join([entry['sentence1'],entry['sentence2'],entry['sentence3'],entry['sentence4'],entry['sentence5']]) == l['story']][0]
    else:
       d_ = [entry for entry in data if entry['story'] == l['story']][0]
    story = nltk.tokenize.sent_tokenize(l['story'])
    if len(story) < 5:
       if 'Super Smash Bros' in story[0]:
           story = [story[0].split('.')[0] + '.',story[0].split('.')[1]] + story[1:]
       else:
           story1 = story[:2]
           story2 = [' '.join(story[2].split(' ')[:7]) + '.']
           story3 = [' '.join(story[2].split(' ')[7:])] + story[3:]
           story = story1 + story2 + story3
    if story in stories:
       continue 
    else:
       stories.append(story)
    dim = reverse_template(l['prefix'])
    if model_type == 'gold_retrieval':
       gen_rel = d_["distance_supervision_relations"][str(l['sentID'])][dim]['relations'][-1]
    elif model_type == 'val':
       gen_rel = [ast.literal_eval(r)[0] for r in d_['distance_supervision_relations'][int(l['sentID'])][1][0][0][1:-1]][dims.index(dim)] #d_["distance_supervision_relations"][str(l['sentID'])][dim]['relations'][-1]
    elif model_type == 'bval':
       gen_rel = [ast.literal_eval(r)[0] for r in d_['distance_supervision_relations'][int(l['sentID'])][1][0][0][1:-1]][dims.index(dim)]
    else:
       if model_type == 'knn':
          gen_rel = d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][str(dims.index(dim))][0]
       else:
          gen_rel = d_['<|sent' + str(l['sentID']) + '|>_generated_relations'][dims.index(dim)][0]
    gen_rel = normalize_rel(gen_rel, dim)
    for sent in range(5):
        tsv_writer.writerow([story[sent],gen_rel, str(sent)])
          
