import argparse
import csv
import numpy as np
parser = argparse.ArgumentParser(description='make tsv')
parser.add_argument('--input_file',type=str,default='examples.tsv')
parser.add_argument('--decoding',type=str,default="beam")
parser.add_argument('--data_dir',type=str,default='./')
parser.add_argument('--split',type=int,default=0)
args = parser.parse_args()

preds = csv.reader(open(args.input_file),delimiter='\t')
preds = [l for l in preds][1:]
story_id = 0
story_arcs = {}
for l in preds:
    if story_id not in story_arcs.keys():
       story_arcs[story_id] = []
    if not l[2].endswith('none.'):
       story_arcs[story_id].append(l[1])
    if (int(l[-1]) + 1) % 5 == 0:
       story_id += 1

contradictions = 0
entailments = 0
neutrals = 0
total = 0
for s in story_arcs.keys():
    if len(story_arcs[s]) == 0:
       continue
    if 'contradiction' in story_arcs[s]:
       contradictions += 1
    elif 'entailment' in story_arcs[s]:
       entailments += 1
    else:
       neutrals += 1
    total += 1
print('% stories - contradiction: ' + str(contradictions/float(len(total))))
print('% stories - entailment or neutral: ' + str((entailments + neutrals)/float(len(total))))

