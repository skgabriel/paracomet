import os
import json
import sys
sys.path.insert(1, '../utils')
from utils import write_items

dir = '../../data/atomic'
save_dir = '../../data'
files = os.listdir(dir)

data = []
for f in files:
    if f.endswith('.jsonl') and 'train' not in f:
       data.append(json.dumps(json.load(open(dir + '/' + f))))
new_file = os.path.join(save_dir, 'all_data.jsonl')
write_items(data, new_file)
