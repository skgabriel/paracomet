
import json
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from transformers import GPT2Tokenizer
from src.text_utils import TextEncoder, fix_malformed, set_up_special_tokens
from src.transformer_models import GPT2Model, GPT2LMHeadModel, GPT2MemModel
from src.decoding import beam_search, topk
import random
import nltk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model_type',type=str,default='nomem')
    parser.add_argument('--model_dir',type=str,default='/home/saadiag/narrative_inference_demo')
    parser.add_argument('--source',type=str,default='example2.jsonl')
    parser.add_argument('--save_filename',type=str,default='outputs.jsonl')
    parser.add_argument('--decoding',type=str,default='beam')
    parser.add_argument('--beam',type=int,default=10)
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.model_type == 'mem':
   use_mem = True
else:
   use_mem = False

model_path = os.path.join(args.model_dir, args.model_type)
device = torch.device(device)
text_encoder = GPT2Tokenizer.from_pretrained('gpt2')
encoder,decoder = set_up_special_tokens(text_encoder.encoder,text_encoder.decoder)
text_encoder.encoder = encoder
text_encoder.decoder = decoder

n_vocab = len(text_encoder.encoder)

if args.model_type == 'mem':
    model = GPT2MemModel.from_pretrained('gpt2')
else:
    model = GPT2LMHeadModel.from_pretrained('gpt2')

model.resize_token_embeddings(n_vocab)
model = nn.DataParallel(model).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

n_gpu = 1
print("device", device, "n_gpu", n_gpu)

def get_token(next_idx):
    try:
       return text_encoder.decoder[next_idx]
    except:
       return next_idx

gen_file = open(os.path.join(args.model_type + '_' + args.decoding + '_' + args.save_filename),'w')

dims_ = ["<|xNeed|>","<|xIntent|>","<|xWant|>","<|oEffect|>","<|xReact|>","<|oWant|>","<|oReact|>","<|xEffect|>","<|xAttr|>"]
dims = [encoder[d] for d in dims_]

def clean_gen(gen):
    gen = [w for w in gen.tolist() if w != encoder['<|PAD|>']]
    gen = [get_token(idx) for idx in gen]
    if '<unk>' in gen:
       gen = [t for t in gen if t != '<unk>']
    gen = "".join([word.replace("Ġ", " ") for word in gen])
    gen = gen.replace("<|endoftext|>","")
    gen = fix_malformed(gen)
    return gen

def pad_rels(relation, pad_len=100):
    return relation[:100] + [encoder['<|PAD|>']] * (100-len(relation[:100]))

if use_mem:
   external_mem = {}

n_updates = 0 

teX = [json.loads(l) for l in open(args.source).readlines()]
for line in teX:
    id = 0 #line['id']
    if use_mem:
       if id not in external_mem.keys():
          external_mem[id] = []
          size_mem = 0
    original = nltk.tokenize.sent_tokenize(line['story'])
    save_output = {}
    save_output["storyid"] = id
    save_output["story"] = ' '.join(original)
    for i in range(len(original)):
        sent_id = "<|sent" + str(i) + "|>"
        with torch.no_grad():
             for d in range(len(dims)):
                 XMB = [encoder['<|PAD|>']] * 600
                 if ('xIntent' in dims_[d] or 'xNeed' in dims_[d] or 'xAttr' in dims_[d]):
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original[:i+1])))
                 else:
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original)))
                 context += [encoder[sent_id], dims[d]]
                 i_1 = len(context)-1
                 XMB[:len(context)] = context 
                 XMB = torch.tensor(XMB,dtype=torch.long).to(device)
                 XMB = XMB.unsqueeze(0)
                 if use_mem and size_mem != 0:
                    mem = external_mem[id]
                    mem = torch.LongTensor([pad_rels(r) for r in mem]).unsqueeze(0)
                    if args.decoding == 'topk':
                       gen = topk(model, encoder, XMB,i_1,mem=mem,size_mem=size_mem)
                    else:
                       gen = beam_search(model, encoder, XMB,i_1,mem=mem,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)
                 else:
                    if args.decoding == 'topk':
                       if use_mem:
                          gen = topk(model, encoder, XMB, i_1,size_mem=size_mem)
                       else:
                          gen = topk(model, encoder, XMB, i_1)
                    else:
                       if use_mem:
                          gen = beam_search(model, encoder, XMB, i_1,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)
                       else:
                          gen = beam_search(model, encoder, XMB, i_1,num_beams=args.beam)
                 gen = [clean_gen(g) for g in gen]
                 if use_mem:
                    mem_gen = gen[0]
                    external_mem[id].append(text_encoder.convert_tokens_to_ids(text_encoder.tokenize(mem_gen)))
                    size_mem += 1
                 if sent_id + '_' + "generated_relations" in save_output.keys(): 
                    save_output[sent_id + '_' + "generated_relations"].append(gen)
                    save_output[sent_id + '_' + "generated_dims"].append([decoder[dims[d]]] * len(gen))
                 else:
                    save_output[sent_id + '_' + "generated_relations"] = [gen]
                    save_output[sent_id + '_' + "generated_dims"] = [[decoder[dims[d]]] * len(gen)]
    gen_file.write(json.dumps(save_output) + '\n')
    n_updates += 1
