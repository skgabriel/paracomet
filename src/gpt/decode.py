import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from text_utils import TextEncoder
import pickle
import numpy as np
from datasets import roc_stories
from utils import (encode_dataset2, iter_data,
                   ResultLogger, make_path)
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, GPT2LMHeadModel, GPT2Tokenizer
from modeling_openai import OpenAIGPTMemModel
from transformer_models import GPT2MemModel
import random
from sklearn.utils import shuffle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_epoch',type=str,default='3')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kg_type',type=str,default='atomic')
    parser.add_argument('--use_gpt2',type=bool,default=False)   
    parser.add_argument('--use_filter',type=bool,default=True)
    parser.add_argument('--use_mem',type=bool,default=True)
    parser.add_argument('--comet',type=bool, default=True)
    parser.add_argument('--model_type',type=str,default='mem_comet') #specify model name
    parser.add_argument('--load_data',type=str,default='../updated_gen_data')
    parser.add_argument('--save_filename',type=str,default='outputs.jsonl')
    parser.add_argument('--original_file',type=str, default='distant_atomic_masked.jsonl')
    parser.add_argument('--data_dir',type=str,default='../data')
    parser.add_argument('--n_batch',type=int,default=1)
    parser.add_argument('--beam',type=int,default=10)
    parser.add_argument('--split',type=str,default='val') #test or val?
    parser.add_argument('--decoding',type=str,default='beam')
    parser.add_argument('--s_split',type=int,default=0)
    parser.add_argument('--e_split',type=int,default=5000)
    parser.add_argument('--filter_decode',type=bool,default=True)
    parser.add_argument('--mem_k',type=int,default=1)
args = parser.parse_args()
print(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#splits = [(0,5000),(5000,10000),(10000,15000),(15000,20000),(20000,25000)]

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_mem = args.use_mem
use_gpt2 = args.use_gpt2

def transform_story(X1, X2):
    n_batch = len(X1)
    n_ctx = 602
    end_token = [encoder['<|endoftext|>']]
    xmb_kg = np.tile(np.array([encoder['<|PAD|>']] * n_ctx,dtype=np.int32),(n_batch,1)) 
    xmb_sy = np.tile(np.array([encoder['<|PAD|>']] * n_ctx,dtype=np.int32),(n_batch,1)) 
    #xmb_kg = np.zeros((n_batch, n_ctx), dtype=np.int32)
    #xmb_sy = np.zeros((n_batch, n_ctx), dtype=np.int32)
    for i, (x1,x2), in enumerate(zip(X1,X2)):
        new_x1 = x1
        new_x2 = x2
        x12 = new_x1
        x13 = new_x2 + end_token
        x14 = new_x2 
        x15 = new_x1[-2:] + new_x1[:-2] + end_token 
        xmb_kg[i,:len(x12)] = x12
        xmb_sy[i,:len(x14)] = x14
        xmb_kg[i,len(x12):len(x12)+len(x13)] = x13
        xmb_sy[i,len(x14):len(x14)+len(x15)] = x15
    return xmb_kg, xmb_sy

device = torch.device(device)
if use_gpt2:
   text_encoder = GPT2Tokenizer.from_pretrained('gpt2') 
else:
   text_encoder = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
encoder = text_encoder.encoder
decoder = text_encoder.decoder

#sentence-level special tokens
encoder['<|sent0|>'] = len(encoder)
decoder[len(decoder)] = '<|sent0|>'

encoder['<|sent1|>'] = len(encoder)
decoder[len(decoder)] = '<|sent1|>'

encoder['<|sent2|>'] = len(encoder)
decoder[len(decoder)] = '<|sent2|>'

encoder['<|sent3|>'] = len(encoder)
decoder[len(decoder)] = '<|sent3|>'

encoder['<|sent4|>'] = len(encoder)
decoder[len(decoder)] = '<|sent4|>'

sent_ids = [encoder['<|sent0|>'],encoder['<|sent1|>'],encoder['<|sent2|>'],encoder['<|sent3|>'],encoder['<|sent4|>']]

#ATOMIC special tokens
encoder['<|xNeed|>'] = len(encoder)
decoder[len(decoder)] = '<|xNeed|>'

encoder['<|xIntent|>'] = len(encoder)
decoder[len(decoder)] = '<|xIntent|>'

encoder['<|xWant|>'] = len(encoder)
decoder[len(decoder)] = '<|xWant|>'

encoder['<|oEffect|>'] = len(encoder)
decoder[len(decoder)] = '<|oEffect|>'

encoder['<|xReact|>'] = len(encoder)
decoder[len(decoder)] = '<|xReact|>'

encoder['<|oWant|>'] = len(encoder)
decoder[len(decoder)] = '<|oWant|>'

encoder['<|oReact|>'] = len(encoder)
decoder[len(decoder)] = '<|oReact|>'

encoder['<|xEffect|>'] = len(encoder)
decoder[len(decoder)] = '<|xEffect|>'

encoder['<|xAttr|>'] = len(encoder)
decoder[len(decoder)] = '<|xAttr|>'

encoder['<|PAD|>'] = len(encoder)
decoder[len(decoder)] = '<|PAD|>'

if not use_gpt2:
   encoder['<|endoftext|>'] = len(encoder) 
   decoder[len(decoder)] = '<|endoftext|>'
text_encoder.encoder = encoder
text_encoder.decoder = decoder
n_vocab = len(text_encoder.encoder)

best_model = 'best_params_' + args.load_epoch


if args.model_type == 'baseline_comet':
   model_path = './models/model/'
if args.model_type == 'baseline':
   model_path = '/home/saadiag/testing_code/modeling/heuristic/models/model/'
if args.model_type == 'mem':
   model_path = './heuristic/mem_models/model/'
if args.model_type == 'baseline_nopre':
   model_path = './h_models_nopre/model/'
if args.model_type == 'mem_nopre':
   model_path = '/net/nfs2.corp/ca-data/masked_baseline_mem_nopre/model/'
if args.model_type == 'rec':
   model_path = '/home/saadiag/commonArc/rec_models/model/'
if args.model_type == 'mem_comet':
   model_path = './mem_models/model/'
if args.model_type == 'mem3':
   model_path = '/home/saadiag/commonArc/masked_mem3_models4/model/'
if args.model_type == 'mem5':
   model_path = '/home/saadiag/commonArc/masked_mem5_models4/model/'
if args.model_type == 'mem10':
   model_path = '/home/saadiag/commonArc/masked_mem10_models4/model/'
if args.model_type == 'mem20':
   model_path = '/home/saadiag/commonArc/masked_mem20_models4/model/'
if args.model_type == 'max-mem':
   model_path = '/home/saadiag/commonArc/max_mem_models/model/'
if args.model_type == 'pg-mem':
   model_path = '/home/saadiag/commonArc/pg_mem_models/model/'

if args.model_type == 'ensemble':
   model_path1 = '/home/saadiag/commonArc/masked_baseline/model/'
   model_path1 = model_path1 + best_model
   model_path2 = '/home/saadiag/commonArc/masked_mem_models/model/'
   model_path2 = model_path2 + best_model
else:
   model_path = model_path + best_model

if use_mem or 'mem' in args.model_type:
    if use_gpt2:
       model = GPT2MemModel.from_pretrained('gpt2')
    else:
       model = OpenAIGPTMemModel.from_pretrained('openai-gpt')
else:
    if use_gpt2:
       model = GPT2LMHeadModel.from_pretrained('gpt2')
    else:
       model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
if args.model_type != 'ensemble':
   model.resize_token_embeddings(n_vocab)
   ##debugging
   model = nn.DataParallel(model).to(device)
   model.load_state_dict(torch.load(model_path))
#   model = torch.load(model_path)
#   model.load_state_dict(model['state_dict'])
   model.eval()

print('loading transformed data')



try:
   teX, teMem, te_ids =  pickle.load(open(os.path.join(args.load_data, 'c' + '_' + 'h' * (1-args.comet) + '_' + args.kg_type + '_' + 'data.pkl'),'rb'))
except:
   try:
        data_dump = pickle.load(open(args.data_dir + '/' + 'c' + '_' + 'h' * (1-args.comet) + '_' + args.kg_type + '_' + 'data.pkl','rb'))
        if args.split == 'test':
           teX1, teX2, teMem, te_ids = data_dump[2]
        else:
           teX1, teX2, teMem, te_ids = data_dump[1]
   except:
        ((trX1, trX2, trMem, trIds),
         (vaX1, vaX2, vaMem, vaIds),
         (teX1, teX2, teMem, te_ids)) = encode_dataset2(*roc_stories(args.data_dir, args.comet, args.kg_type,use_filter=False),encoder=text_encoder)
        pickle.dump([(trX1,trX2, trMem, trIds), (vaX1, vaX2, vaMem, vaIds), (teX1, teX2, teMem, te_ids)], open(args.data_dir + '/' + 'c' + '_'  + 'h' * (1-args.comet) + '_' +  args.kg_type + '_' + 'data.pkl','wb'))
   teX, teM = transform_story(teX1, teX2)
   pickle.dump((teX,teMem, te_ids), open(os.path.join(args.load_data, 'c' + '_' + 'h' * (1-args.comet) + '_' + args.kg_type + '_' + 'data.pkl'),'wb'))



n_gpu = 1
print("device", device, "n_gpu", n_gpu)
n_updates = 0
n_batch_test = args.n_batch

gen_len = 50

def topk(model, XMB,i, n=1,num_beams=args.beam, mem=None,use_pointer=None,use_scores=None,size_mem=0):
    import copy
    gen = torch.Tensor([encoder['<|PAD|>']] * gen_len).long().to(device) #torch.zeros((gen_len)).long().to(device)
    prob = 0
    for step in range(gen_len):
        if encoder['<|endoftext|>'] in gen:
           break
        if step == 0:
           clear_mem = True
        else:
           clear_mem = False
        XMB[:,i+1:i+gen_len+1] = gen
        if mem == None:
           logits, _ =  model(XMB[:,:i+1+step])
        else:
           logits, _ =  model(XMB[:,:i+1+step],update_mem=mem,clear_mem=clear_mem,use_pointer=use_pointer, use_scores=use_scores,mem_k=args.mem_k,use_mem=use_mem,size_mem=size_mem)
        logits = torch.nn.functional.softmax(logits, dim=-1)
        logits = logits[:,i+step].squeeze(1)
        values, indices  = logits.sort(descending=True)
        next_indices = indices[:, :num_beams].gather(-1, torch.multinomial(values[:, :num_beams], 1))
        gen[step] = next_indices.view(-1).long() 
        prob += float(values[:,int(gen[step])])
    return [gen]

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

def beam_search(model, XMB, start_id, num_beams=1, max_length=gen_len, temperature=1, pad_token_id=encoder['<|PAD|>'], eos_token_ids=[encoder['<|endoftext|>']], length_penalty=1,mem=None,use_mem=False,size_mem=0):
    """ Generate sequences for each example with beam search.
    """

    # generated hypotheses
    vocab_size = len(encoder)
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(1)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((1, num_beams), dtype=torch.float, device=XMB.device)
    # Greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = None

    # done sentences
    done = [False for _ in range(1)]

    step = 0 
    XMB = XMB[:,:start_id+1]
    if mem != None:
       mem = mem.expand_as(torch.zeros(num_beams, mem.size(1), mem.size(2)))
    XMB = XMB.expand_as(torch.zeros(num_beams,XMB.size(1)))
    while step < max_length:
        if step == 0 and use_mem:
           clear_mem = True
        else:
           clear_mem = False
        if XMB == None:
           import pdb; pdb.set_trace()
        if mem == None:
           if not use_mem:
              outputs  = model(input_ids=XMB)  # (batch_size * num_beams, cur_len, vocab_size)
           else:
              outputs = model(input_ids=XMB,update_mem=mem,clear_mem=clear_mem,mem_k=args.mem_k,size_mem=size_mem)
        else:
           if step != 0:
              mem = None
           outputs = model(input_ids=XMB,update_mem=mem,clear_mem=clear_mem,mem_k=args.mem_k,size_mem=size_mem)
        next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

        # do greedy beam search
        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
        assert scores.size() == (num_beams, vocab_size)
        # Add the log prob of the new beams to the log prob of the beginning of the sequence (sum of logs == log of the product)
        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(1, num_beams * vocab_size)  # (batch_size, num_beams * vocab_size)
        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        assert next_scores.size() == next_tokens.size() == (1, 2 * num_beams)

        # next batch beam content
        # list of (batch_size * num_beams) tuple(next hypothesis score, next word, current position in the batch)
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(1):

            # if we are done with this sentence
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item()
            )
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_ids is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for idx, score in zip(next_tokens[batch_idx], next_scores[batch_idx]):

                # get beam and word IDs
                beam_id = idx // vocab_size
                token_id = idx % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if eos_token_ids is not None and token_id.item() in eos_token_ids:
                    generated_hyps[batch_idx].add(
                        XMB[effective_beam_id].clone(), score.item(),
                    )
                else:
                    # add next predicted word if it is not eos_token
                    next_sent_beam.append((score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # sanity check / prepare next batch
        assert len(next_batch_beam) == 1 * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = XMB.new([x[1] for x in next_batch_beam])
        beam_idx = XMB.new([x[2] for x in next_batch_beam])

        # re-order batch
        XMB = XMB[beam_idx, :]
        XMB = torch.cat([XMB, beam_tokens.unsqueeze(1)], dim=-1)

        # re-order internal states
        if past:
            past = self._reorder_cache(past, beam_idx)

        # stop when we are done with each sentence
        if all(done):
            break

        # update current length
        step = step + 1

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(1):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_ids is not None and all(
            (token_id % vocab_size).item() not in eos_token_ids for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(1, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(1, num_beams)[batch_idx]
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = XMB[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = 1 
    output_num_return_sequences_per_batch = 1

    # select the best hypotheses
    sent_lengths = XMB.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are filled with pad_token
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = XMB.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_ids[0]
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(model.parameters()).device)
    gens = [decoded[0][start_id+1:]] + [s[1][start_id+1:] for s in sorted_hyps]
    return gens

def get_greedy(XMB,i,encoder):
    gen = []
    for k in range(gen_len):
        logits = model(XMB[:,:i+1+k])
        next_idx = torch.multinomial(logits[:,i+k].data.exp(),1) 
        try:
           next_token = text_encoder.decoder[next_idx.item()]
        except:
           next_token = '<' + str(next_idx.item()) + '>'
        gen.append(next_token)
        XMB[:,i+k+1] = next_idx
    return gen
 
def get_token(next_idx, tokenizer=text_encoder):
    try:
       return tokenizer.decoder[next_idx]
    except:
       return next_idx

gen_file = open(os.path.join('./gen_data/gpt', args.model_type + '_' + args.decoding + '_' + args.save_filename),'w')
#gen_file = open(os.path.join(args.load_data, args.model_type + '_' + args.decoding + '_' + args.save_filename),'w')
original_data = open(os.path.join(args.data_dir, args.original_file))
original_data = [json.loads(l) for l in original_data.readlines()]    

dims_ = {'atomic':["<|xNeed|>","<|xIntent|>","<|xWant|>","<|oEffect|>","<|xReact|>","<|oWant|>","<|oReact|>","<|xEffect|>","<|xAttr|>"],'concept':['<|UsedFor|>','<|AtLocation|>','<|HasSubevent|>','<|CapableOf|>','<|HasPrerequisite|>','<|Causes|>','<|MotivatedByGoal|>','<|ReceivesAction|>','<|CausesDesire|>','<|HasFirstSubevent|>','<|Desires|>','<|NotDesires|>','<|HasLastSubevent|>']}[args.kg_type]
dims = [encoder[d] for d in dims_]

def clean_gen(gen):
    gen = [w for w in gen.tolist() if w != encoder['<|PAD|>']]
    gen = [get_token(idx) for idx in gen]
    if '<unk>' in gen:
       gen = [t for t in gen if t != '<unk>']
    if use_gpt2:
       gen = "".join([word.replace("Ġ", " ") for word in gen])
    else:
       gen = "".join([word.replace("</w>", " ") for word in gen])
    gen = gen.replace("<|endoftext|>","")
    if gen[-1] == ' ':
       gen = gen[:-1]
    return gen

def pad_rels(relation, pad_len=100):
    return relation[:100] + [encoder['<|PAD|>']] * (100-len(relation[:100]))

def handle_empty(list_of_rels):
    if len(list_of_rels) == 0:
       return [[] for i in range(max_mem_size)]
    if len(list_of_rels) < max_mem_size:
       list_of_rels.extend([[] for i in range(max_mem_size - len(list_of_rels))])
    return list_of_rels 

if use_mem:
   external_mem = {}

gold_set = [json.loads(l) for l in open('gold_set.jsonl').readlines()]
gold_stories = [l['story'] for l in gold_set]
#mem_file = open('retrieve_mem.txt','w')

n_updates = 0 
for line_ in original_data:
    id = line_["storyid"]
    if use_mem:
       if id not in external_mem.keys():
          external_mem[id] = []
          size_mem = 0
       else:
          continue 
    original = [line_['sentence1'],line_['sentence2'],line_['sentence3'],line_['sentence4'],line_['sentence5']]
    if ' '.join(original) not in gold_stories:
       n_updates += 1
       continue
    save_output = {}
    save_output["storyid"] = id 
    save_output["story"] = " ".join(original)
    save_output["gold_relations"] = line_["distance_supervision_relations"]
#    text_mem = []
    for sent_id in sent_ids: #= [idx for idx in XMB[0].tolist() if idx >= encoder['<|sent0|>'] and idx <= encoder['<|sent4|>']][0]    #relations = []
#        sent = original[int(decoder[sent_id].split('<|sent')[1].replace('|>',''))]
#        story = original[:int(decoder[sent_id].split('<|sent')[1].replace('|>',''))]
#        story += [decoder[sent_id].replace('sent','')]
#        story += original[int(decoder[sent_id].split('<|sent')[1].replace('|>','')) + 1 :]
        with torch.no_grad():
             for d in range(len(dims)):
                 #new_XMB = XMB.clone()
                 XMB = [encoder['<|PAD|>']] * 600
                 if args.filter_decode and ('xIntent' in dims_[d] or 'xNeed' in dims_[d] or 'xAttr' in dims_[d]):
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original[:int(decoder[sent_id].split('<|sent')[1].replace('|>',''))+1])))
                 else:
                    context = text_encoder.convert_tokens_to_ids(text_encoder.tokenize(' '.join(original)))
                 context += [sent_id, dims[d]]
                 i_1 = len(context)-1
                 XMB[:len(context)] = context 
                 XMB = torch.tensor(XMB,dtype=torch.long).to(device)
                 XMB = XMB.unsqueeze(0)
                 if use_mem and size_mem != 0:
                    mem = external_mem[id]
                    mem = torch.LongTensor([pad_rels(r) for r in mem]).unsqueeze(0)
                    if args.decoding == 'topk':
                       gen = topk(model, XMB,i_1,mem=mem,size_mem=size_mem,num_beams=args.beam)
                    else:
                       gen = beam_search(model, XMB,i_1,mem=mem,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)
                 else:
                    if args.decoding == 'topk':
                       if use_mem:
                          gen = topk(model, XMB, i_1,size_mem=size_mem,num_beams=args.beam)
                       else:
                          gen = topk(model, XMB, i_1,num_beams=args.beam)
                    else:
                       if use_mem:
                          gen = beam_search(model, XMB,i_1,num_beams=args.beam,size_mem=size_mem,use_mem=use_mem)                          
                       else:
                          gen = beam_search(model, XMB, i_1,num_beams=args.beam)
                 gen = [clean_gen(g) for g in gen]
                 #r = 0
                 #while gen.replace(' ','') == 'none' and r < max_resample:
                 #      gen = topk(model, new_XMB,i_1+len(d))
                 #      gen = clean_gen(gen)
                 #      r += 1
                 #      print(r)
                 #score_prob([story],[gen],[decoder[d]],eval_sents=[sent],kg_type=args.kg_type)[0]
                 if use_mem:
                    mem_gen = gen[0]
                    size_mem += 1
                    #text_mem.append(mem_gen)
                    external_mem[id].append(text_encoder.convert_tokens_to_ids(text_encoder.tokenize(mem_gen)))
                    #external_mem[id][0][0].append(text_encoder.convert_tokens_to_ids(text_encoder.tokenize(gen)))  #relations.append(gen)
                    #external_mem[id][0][1].append(score)
                    #external_mem[id][0][2].append(gen)
                    #external_mem[id][0][3].append(decoder[d])
                 if decoder[sent_id] + '_' + "generated_relations" in save_output.keys(): 
                    save_output[decoder[sent_id] + '_' + "generated_relations"].append(gen)
                    save_output[decoder[sent_id] + '_' + "generated_dims"].append([decoder[dims[d]]] * len(gen))
                 else:
                    save_output[decoder[sent_id] + '_' + "generated_relations"] = [gen]
                    save_output[decoder[sent_id] + '_' + "generated_dims"] = [[decoder[dims[d]]] * len(gen)]
                    
    gen_file.write(json.dumps(save_output) + '\n')
#    mem_file.write(json.dumps({mem:text_mem,retrieved}) + '\n')
    n_updates += 1
    print(n_updates)
    print('write')
