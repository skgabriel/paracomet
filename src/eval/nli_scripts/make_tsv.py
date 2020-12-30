import json
import nltk
import os
import math
import numpy as np
import argparse
import random
import ast 
import csv
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
##format SEMBERT input 

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
    if 'xEffect' in dim: 
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' are likely ' #'The first person is likely ' #return 'PersonX ' + rel
       else: 
          root = pronoun + ' is likely ' 
#       root = pronoun + ' are likely ' #'The first person is likely ' #return 'PersonX ' + rel

    if 'oEffect' in dim: 
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' are likely ' #'The second person is likely ' #return 'PersonY ' + rel
       else: 
          root = pronoun + ' is likely '  

    if 'xWant' in dim: 
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' want ' #'The first person wants ' #'PersonX will want to ' + rel
       else: 
          root = pronoun + ' wants ' 

    if 'oWant' in dim: 
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       root = pronoun + ' want ' #'The second person wants '

    if 'xIntent' in dim: 
       if not rel_.startswith('to'):
          rel_ = 'to ' + ' '.join([WordNetLemmatizer().lemmatize(w,'v') for w in rel_.split(' ')])
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' wanted ' #"The first person wanted " #'The intent was ' + rel
       else: 
          root = pronoun + ' wanted ' 

    if 'xAttr' in dim: 
       if rel_.startswith('to'):
          rel_ = convert(rel_, WN_VERB, WN_ADJECTIVE)
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' are seen as ' #'The first person is seen as '
       else: 
          root = pronoun + ' is seen as ' 

    if 'xNeed' in dim: 
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' needed ' #'The second person needs ' #'PersonX needs to ' + rel
       else: 
          root = pronoun + ' needed ' 

    if 'xReact' in dim: 
       if rel_.startswith('to'):
          rel_ = convert(rel_, WN_VERB, WN_ADJECTIVE)
       if pronoun.lower() == 'they' or pronoun.lower() == 'we':
          root = pronoun + ' then feel ' #'The first person then feels ' #'PersonX feels ' + rel
       else: 
          root = pronoun + ' then feels ' 

    if 'oReact' in dim: 
       if rel_.startswith('to'):
          rel_ = convert(rel_, WN_VERB, WN_ADJECTIVE)
       root = 'The others then feel ' #'The others then feel ' #'PersonY feels ' + rel

    return root + rel_


random.seed(0)

parser = argparse.ArgumentParser(description='make tsv')
parser.add_argument('--input_file',type=str,default='examples.jsonl')
parser.add_argument('--output_file',type=str, default='beam_outputs')
parser.add_argument('--data_dir',type=str,default='../../data')
args = parser.parse_args()

output_file = args.output_file + ".tsv"
original_data = open(args.input_file)
original_data = [json.loads(l) for l in original_data.readlines()]
dims = ["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]

tsv_writer = csv.writer(open(output_file,'w'),delimiter='\t')
hyps = []
sents = []
stories = []
dim_rels = []
tsv_writer.writerow(['story','relation','sent_id'])
stories = []
for l_ in original_data:
    story = nltk.tokenize.sent_tokenize(l_['story'])
    for sent in range(len(story))[:5]:
        for d in dims:
            gen_rel = l_['<|sent' + str(sent) + '|>_generated_relations'][dims.index(d)][0]
            gen_rel = normalize_rel(gen_rel, d)
            tsv_writer.writerow([story[sent],gen_rel, str(sent)])
          
