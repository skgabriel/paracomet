import time
from collections import defaultdict
import torch
import math
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer

loss_fct = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=0)
model = GPT2LMHeadModel.from_pretrained('gpt2').cuda()
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def add_template(rel, dim, kg_type='atomic'):
    if len(rel) == 0:
       rel = 'none.'
    if rel[-1] != '.':
       rel += '.'

    if 'xEffect' in dim: #yes
       return 'PersonX is likely: ' + rel #return 'PersonX ' + rel

    if 'oEffect' in dim: #yes
       return 'PersonY is likely: ' + rel #return 'PersonY ' + rel

    if 'xWant' in dim: #yes
       return 'PersonX wants: ' + rel #'PersonX will want to ' + rel
    
    if 'oWant' in dim: #yes
       return 'PersonY wants: ' + rel

    if 'xIntent' in dim: #yes
       return 'PersonX wanted: ' + rel #'The intent was ' + rel

    if 'oIntent' in dim:
       return 'PersonY wanted: ' + rel

    if 'xAttr' in dim: #yes
       return 'PersonX is seen as: ' + rel

    if 'xNeed' in dim: #yes
       return 'PersonX needed: ' + rel #'PersonX needs to ' + rel

    if 'xReact' in dim: #yes
       return 'PersonX then feels: ' + rel #'PersonX feels ' + rel

    if 'oReact' in dim: #yes
       return 'Others then feel: ' + rel #'PersonY feels ' + rel

    return rel 

def adjust_cands1(c):
    if len(c) > 0:
       c = c[:-1]
    return c

def adjust_tensor2(t):
    if len(t) > 0:
       t[0] = 'Ġ' + t[0]
    return t

def adjust_tensor3(t):
    if len(t) > 0:
       t[0] = 'Ġ' + t[0]
    return t

def score_prob(cands, refs, types, eval_sents=None, mask_rel=True, kg_type='atomic'):
    inputs = torch.zeros((len(cands),100))
    mask = torch.ones((len(cands),100)) 
    cands1 = [c.split('<|')[0] for c in cands]
    cands1 = [adjust_cands1(cands1[i]) for i in range(len(cands))] 
    cands2 = [c.split('<|')[1].split('|>')[1] for c in cands]
    tensor_input1 = [tokenizer.tokenize(cands1[i] + ' ' + eval_sents[i]) for i in range(len(cands))]
    tensor_input2 = [tokenizer.tokenize(add_template(refs[i], types[i], kg_type)) for i in range(len(cands))]
    tensor_input2 = [adjust_tensor2(tensor_input2[i]) for i in range(len(cands))]
    tensor_input3 = [tokenizer.tokenize(cands2[i]) for i in range(len(cands))]
    tensor_input3 = [adjust_tensor3(tensor_input3[i]) for i in range(len(cands))]

    tensor_input1 = [tokenizer.convert_tokens_to_ids(tensor_input1[i]) for i in range(len(cands))]
    tensor_input2 = [tokenizer.convert_tokens_to_ids(tensor_input2[i]) for i in range(len(cands))]
    tensor_input3 = [tokenizer.convert_tokens_to_ids(tensor_input3[i]) for i in range(len(cands))]
    tensor_input = [(tensor_input1[i] + tensor_input2[i] + tensor_input3[i])[:99] + [tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])] for i in range(len(cands))]

    lengths = []
    for i in range(inputs.size(0)):
        mask[i,len(tensor_input[i]):] = 0
        if mask_rel:
           mask[i,len(tensor_input1[i]):len(tensor_input1[i])+len(tensor_input2[i])] = 0
        lengths.append(mask[i].nonzero().size(0))
        inputs[i,:len(tensor_input[i])] = torch.Tensor(tensor_input[i])
    losses = []
    steps = 0
    batch_size = 130 #batch_size = 108
    inputs = inputs.long().cuda()
    input_mask = mask.cuda()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    while steps < inputs.size(0):
          lm_logits = model(inputs[steps:steps+batch_size,:])[0]
          shift_input_mask = input_mask[steps:steps+batch_size, 1:].contiguous().view(-1)
          shift_labels = inputs[steps:steps+batch_size, 1:].contiguous().view(-1)
          shift_logits = lm_logits[..., :-1, :].contiguous()
          shift_logits = shift_logits.view(-1, shift_logits.size(-1)) 
          loss = loss_fct(shift_logits, shift_labels)
          loss_mask = torch.mul(shift_input_mask, (shift_labels > 0).long())
          loss = torch.mul(loss_mask, loss)          
          loss = loss.view(inputs[steps:steps+batch_size,:].size(0),-1).sum(dim=1)
          loss = loss.cpu().tolist()
          per_token_loss = [-(loss[i]/lengths[steps+i]) for i in range(len(loss))]
          losses.extend(per_token_loss) 
          steps += inputs[steps:steps+batch_size,:].size(0)
    for i in range(len(refs)):
        if refs[i].lower() == 'none':
           losses[i] = -math.inf
    return losses

        

#        # Shift so that tokens < n predict n
#        shift_logits = output[..., :-1, :].contiguous()
#        shift_labels = inputs[steps:steps+batch_size,:][..., 1:].contiguous()
#        # Flatten the tokens
#        #for i in range(batch_size):
#        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#        loss = loss.view(inputs[steps:steps+batch_size,:].size(0), -1).sum(axis=1).tolist()
#        loss = [float(-loss[l])/lengths[steps+l] for l in range(len(loss))]
#        losses.extend(loss)
#        steps += batch_size
#    #for i in range(len(refs)):
#        #if refs[i] == 'none':
#        #   losses[i] = -math.inf
#    return losses

##function for evaluating stories with knowledge appended
def score_prob3(cands, refs, types, eval_sents=None, mask_rel=True, kg_type='atomic', story_only=True):
    inputs = torch.zeros((1,100)) 
    mask = torch.ones((1,100))
    if story_only:
       tensor_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(cands[0])))
    else:
       cands1 = [c.split('<|')[0] for c in cands]
       if len(cands1[0]) > 0:
          cands1[0] = cands1[0][:-1]
       cands2 = [c.split('<|')[1].split('|>')[1][1:] for c in cands]
       tensor_input1 = tokenizer.tokenize(cands1[0] + ' ' + eval_sents[0])
       tensor_input2 = tokenizer.tokenize(add_template(refs[0],types[0],kg_type))
       if len(tensor_input1) > 0:
          tensor_input2[0] = 'Ġ' + tensor_input2[0]
       tensor_input3 = tokenizer.tokenize(cands2[0]) 
       if len(tensor_input3) > 0:
          tensor_input3[0] = 'Ġ' + tensor_input3[0]
       tensor_input1 = tokenizer.convert_tokens_to_ids(tensor_input1)
       tensor_input2 = tokenizer.convert_tokens_to_ids(tensor_input2)
       tensor_input3 = tokenizer.convert_tokens_to_ids(tensor_input3)
       if mask_rel:
          mask[:,len(tensor_input1):len(tensor_input1)+len(tensor_input2)] = 0
#       if pre:
#          tensor_input1 = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cands1[0]))
#       else:
#          if len(cands1[0]) > 0:
#             if cands1[0][-1] == ' ':
#                cands1[0] = cands1[0][:-1]
#             tensor_input1 = tokenizer.tokenize(cands1[0] + ' ' + eval_sents[0])
#          else:
#             tensor_input1 = tokenizer.tokenize(eval_sents[0])
#          tensor_input1[0] = 'Ġ' + tensor_input1[0] 
#          tensor_input1 = tokenizer.convert_tokens_to_ids(tensor_input1)
#       if type(tensor_input1) != list:
#          tensor_input1 = [tensor_input1]
#       tensor_input2 = tokenizer.tokenize(add_template(refs[0],types[0],kg_type))
#       if len(tensor_input1) > 0:
#          tensor_input2[0] = 'Ġ' + tensor_input2[0]
#       tensor_input2 = tokenizer.convert_tokens_to_ids(tensor_input2)
#       if pre:
#          tensor_input2 = tensor_input2 + [220]
#       if type(tensor_input2) != list:
#          tensor_input2 = [tensor_input2]
#       if pre:
#          tensor_input3 = tokenizer.tokenize(eval_sents[0] + ' ' + cands2[0])
#          tensor_input3 = tokenizer.convert_tokens_to_ids(tensor_input3)
#       else:
#          tensor_input3 = tokenizer.tokenize(cands2[0])
#          if len(tensor_input3) > 0:
#             tensor_input3[0] = 'Ġ' + tensor_input3[0]
#       tensor_input3 = tokenizer.convert_tokens_to_ids(tensor_input3)
#       if type(tensor_input3) != list:
#          tensor_input3 = [tensor_input3]
#       if mask_rel:
#          if pre and len(tensor_input1) > 0:
#             mask[:,len(tensor_input1):len(tensor_input1)+len(tensor_input2)-1] = 0   
#          else:
#             mask[:,len(tensor_input1):len(tensor_input1)+len(tensor_input2)] = 0
       tensor_input = tensor_input1 + tensor_input2 + tensor_input3
    tensor_input = tensor_input[:99] + [tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])] 
    sanity_check1 = ''.join([t.replace('Ġ',' ') for t in tokenizer.convert_ids_to_tokens(tensor_input)])
    mask[:,len(tensor_input):] = 0
    inputs[:,:len(tensor_input)] = torch.Tensor(tensor_input)
    sanity_check2 = ''.join([t.replace('Ġ',' ') for t in tokenizer.convert_ids_to_tokens([int(i) for i in (mask * inputs).tolist()[0]])])
    inputs = inputs.long().cuda()
    input_mask = mask.cuda()
    shift_input_mask = input_mask[..., 1:].contiguous().view(-1)
    shift_labels = inputs[..., 1:].contiguous().view(-1)
    lm_logits = model(inputs)[0]
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1)) 
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels)
    loss_mask = torch.mul(shift_input_mask, (shift_labels > 0).long())
    loss = torch.mul(loss_mask, loss)
    per_token_loss = loss.sum() / loss_mask.nonzero().size(0)
    return per_token_loss.mean().item(), loss_mask.nonzero().size(0)

##function for evaluating stories without knowledge appended
def score_prob2(cands, refs, types, full_story=True, eval_sents=None, kg_type='atomic'):
    inputs = torch.zeros((1,100))
    mask = torch.zeros((1,100))
    lengths = [] 
    for i in range(inputs.size(0)):
        tensor_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cands[i]))[:99] + [tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])]
        mask[i,:len(tensor_input)+1] = 1 
        lengths.append(len(tensor_input))
        inputs[i,:len(tensor_input)] = torch.Tensor(tensor_input)
    inputs = inputs.long().cuda()
    input_mask = mask.cuda()
    shift_input_mask = input_mask[..., 1:].contiguous().view(-1)
    shift_labels = inputs[..., 1:].contiguous().view(-1)
    lm_logits = model(inputs)[0]
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1)) 
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits, shift_labels)
    loss_mask = torch.mul(shift_input_mask, (shift_labels > 0).long())
    loss = torch.mul(loss_mask, loss)
    per_token_loss = loss.sum() / loss_mask.nonzero().size(0)
    return per_token_loss.mean().item(), lengths[0]

def score_prob4(cands, refs, types, full_story=True, eval_sents=None, kg_type='atomic'):
    inputs = torch.zeros((1,100))
    lengths = [] 
    for i in range(inputs.size(0)):
        tensor_input = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cands[i]))[:99]  + [tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])]
        lengths.append(len(tensor_input))
        inputs[i,:len(tensor_input)] = torch.Tensor(tensor_input)
    inputs = inputs.long().cuda()
    output = model(inputs)
    output = output[0]
    # Shift so that tokens < n predict n
    shift_logits = output[..., :-1, :].contiguous()
    shift_labels = inputs[:,:][..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(inputs[:,:].size(0), -1).sum(axis=1).tolist()
    return float(loss[0]), lengths[0]
