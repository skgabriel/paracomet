import argparse
import csv
import numpy as np
parser = argparse.ArgumentParser(description='make tsv')
parser.add_argument('--model_type',type=str,default='mem')
parser.add_argument('--decoding',type=str,default="beam")
parser.add_argument('--data_dir',type=str,default='./')
parser.add_argument('--split',type=int,default=0)
args = parser.parse_args()

preds = csv.reader(open('gpt_gold_' + args.model_type + '_' + args.decoding + '_' + str(args.split) + '.tsv'),delimiter='\t')
preds = [l for l in preds][1:]
import pdb; pdb.set_trace()
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
flips = 0
story_level_cs = []
story_level_es = []
for s in story_arcs.keys():
    if len(story_arcs[s]) == 0:
       continue
    if 'contradiction' in story_arcs[s]:
       contradictions += 1
    elif 'entailment' in story_arcs[s]:
       entailments += 1
    else:
       neutrals += 1
    if 'contradiction' in story_arcs[s] and 'entailment' in story_arcs[s]:
       flips += 1
    story_level_c = float(len([l for l in story_arcs[s] if l == 'contradiction']))/len(story_arcs[s])
    story_level_e = float(len([l for l in story_arcs[s] if l == 'entailment' or l == 'neutral']))/len(story_arcs[s])
    story_level_cs.append(story_level_c)
    story_level_es.append(story_level_e)
import pdb; pdb.set_trace()
print('% stories - contradiction: ' + str(contradictions/float(len(story_level_cs))))
print('% stories - entailment or neutral: ' + str((entailments + neutrals)/float(len(story_level_cs))))
print('% stories - flip: ' + str(flips))
print('story-level % of contradiction: ' + str(np.mean(story_level_cs)))
print('story-level % of entailment or neutral: ' + str(np.mean(story_level_es)))

