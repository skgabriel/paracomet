## Paragraph-Level Commonsense Transformers with Recurrent Memory 

This repository contains the code used in the paper:

Paragraph-Level Commonsense Transformers with Recurrent Memory. *Saadia Gabriel, Chandra Bhagavatula, Vered Shwartz, Ronan Le Bras, Maxwell Forbes, Yejin Choi*. AAAI 2021. [link] (https://arxiv.org/abs/2010.01486)

This is a general purpose framework for aligning commonsense knowledge with narrative text. The repo contains 

1) A framework for distantly supervised paragraph-level commonsense knowledge alignment; and 
2) Modeling code for finetuning pretrained transformers to generate paragraph-level commonsense inferences. 

### Instructions 

python >= 3.6  (I suggest using a virtual environment) 

#### Setup

1. pip install -r requirements.txt 
2. python -m spacy download en
3. cd data, wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz 
4. tar -xvzf atomic_data.tgz 

#### Prep for distant supervision 

Data for distant supervision should be a file "train-processed.jsonl" in the data folder. The file should contain the following keys:

dict_keys(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5', 'sentence1_tokens', 'sentence1_noun_phrases', 'sentence1_verb_phrases', 'sentence2_tokens', 'sentence2_noun_phrases', 'sentence2_verb_phrases', 'sentence3_tokens', 'sentence3_noun_phrases', 'sentence3_verb_phrases', 'sentence4_tokens', 'sentence4_noun_phrases', 'sentence4_verb_phrases', 'sentence5_tokens', 'sentence5_noun_phrases', 'sentence5_verb_phrases'])

For now, distant supervision code assumes that stories have exactly five sentences 

#### Distant Supervision (Heuristic) 

1. python distant_supervision.py --target_dir ../../data/atomic 

#### Distant Supervision (COMeT) 

1. Get pretrained models from [link] https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB and place in data folder 
2. python distant_supervision.py --comet --comet-location ../../data --target_dir ../../data/atomic 

#### Train (w/o memory)

#### Train (memory)

#### Decode (w/o memory) 

#### Decode (memory)

#### Evaluation data 

./data/gold_set.jsonl 

### References 

Please cite this repository using the following reference:

```
@inproceedings{Gabriel2021ParagraphLevelCT,
title={Paragraph-Level Commonsense Transformers with Recurrent Memory},
author={Gabriel, Saadia and Bhagavatula, Chandra and Shwartz, Vered and Le Bras, Ronan and Forbes, Maxwell and Choi, Yejin},
booktitle={AAAI},
year={2021},
}
```
