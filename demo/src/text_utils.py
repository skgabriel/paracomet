# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import ftfy
import json
import spacy

from tqdm import tqdm

def set_up_special_tokens(encoder,decoder):
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
    
    return encoder, decoder
    

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def text_standardize(text):
    """
    fixes some issues the spacy tokenizer had on books corpus
    also does some whitespace standardization
    """
    text = text.replace('—', '-')
    text = text.replace('–', '-')
    text = text.replace('―', '-')
    text = text.replace('…', '...')
    text = text.replace('´', "'")
    text = re.sub(r'''(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)''', r' \1 ', text)
    text = re.sub(r'\s*\n\s*', ' \n ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path, bpe_path):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.encoder = json.load(open(encoder_path))
        self.decoder = {v:k for k,v in self.encoder.items()}
        merges = open(bpe_path, encoding='utf-8').read().split('\n')[1:-1]
        merges = [tuple(merge.split()) for merge in merges]
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {}

    def bpe(self, token):
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        if word == '\n  </w>':
            word = '\n</w>'
        self.cache[token] = word
        return word

    def encode(self, texts, verbose=True):
        texts_tokens = []
        if verbose:
            for text in tqdm(texts, ncols=80, leave=False):
                try:
                   text = self.nlp(text_standardize(ftfy.fix_text(text)))
                except:
                   text = self.nlp(text_standardize(ftfy.fix_text(text[0])))
                text_tokens = []
                #print(text)
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                #print(text_tokens)
                texts_tokens.append(text_tokens)
        else:
            for text in texts:
                try:
                    text = self.nlp(text_standardize(ftfy.fix_text(text)))
                except:
                    text = self.nlp(text_standardize(ftfy.fix_text(text[0])))


                text_tokens = []
                for token in text:
                    text_tokens.extend([self.encoder.get(t, 0) for t in self.bpe(token.text.lower()).split(' ')])
                texts_tokens.append(text_tokens)
        return texts_tokens
    
def fix_malformed(rel):
    if '<|' in rel:
       return 'no effect'
    return rel
