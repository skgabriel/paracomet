import json 
import sys
sys.path.insert(1, '../utils')
from utils import write_items

comet = True
data = [json.loads(l) for l in open('../../data/all_data.jsonl').readlines()]

train_f = '../../data/' + 'c_' * comet + 'h_' * (not comet) + 'atomic_train.jsonl'
val_f = '../../data/' + 'c_' * comet + 'h_' * (not comet) + 'atomic_val.jsonl'
test_f = '../../data/' + 'c_' * comet + 'h_' * (not comet) + 'atomic_test.jsonl'

train_ids = [l.strip() for l in open('../../data/train_splits.txt').readlines()]
val_ids = [l.strip() for l in open('../../data/val_splits.txt').readlines()]
test_ids = [l.strip() for l in open('../../data/test_splits.txt').readlines()]

train_list = []
test_list = []
val_list = []
for d in data:
    if d["storyid"] in train_ids:
       train_list.append(json.dumps(d))   
    if d["storyid"] in test_ids:
       test_list.append(json.dumps(d))
    if d["storyid"] in val_ids:
       val_list.append(json.dumps(d))
write_items(train_list, train_f)
write_items(test_list, test_f)
write_items(val_list, val_f)
