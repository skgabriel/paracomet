## Paragraph-level Commonsense Transformers with Recurrent Memory 

This repository contains the code used in the paper:

Paragraph-level Commonsense Transformers with Recurrent Memory. *Saadia Gabriel, Chandra Bhagavatula, Vered Shwartz, Ronan Le Bras, Maxwell Forbes, Yejin Choi*. AAAI 2021. [link] (https://arxiv.org/abs/2010.01486)

This is a general purpose framework for aligning commonsense knowledge with narrative text. The repo contains 

1) A framework for distantly supervised paragraph-level commonsense knowledge alignment; and 
2) Modeling code for finetuning pretrained transformers to generate paragraph-level commonsense inferences. 

### Instructions 

python >= 3.6  (I suggest using a virtual environment) 

Note: For now, the code assumes that stories contain at most 5 sentences and models generate inferences for up to 5 sentence stories. 

#### Setup

```
pip install -r requirements.txt 
cd data
wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz 
tar -xvzf atomic_data.tgz 
```

#### Prep for Distant Supervision 

Data for distant supervision should be a file "train-processed.jsonl" in the data folder. The file should contain the following keys:

```
dict_keys(['storyid', 'storytitle', 'sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5', 'sentence1_tokens', 'sentence1_noun_phrases', 'sentence1_verb_phrases', 'sentence2_tokens', 'sentence2_noun_phrases', 'sentence2_verb_phrases', 'sentence3_tokens', 'sentence3_noun_phrases', 'sentence3_verb_phrases', 'sentence4_tokens', 'sentence4_noun_phrases', 'sentence4_verb_phrases', 'sentence5_tokens', 'sentence5_noun_phrases', 'sentence5_verb_phrases'])
```

#### Distant Supervision (Heuristic) 

```
cd src/ds
python distant_supervision.py --target_dir ../../data/atomic 
```

#### Distant Supervision (COMET) 

Get pretrained comet models from [link] (https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB) and place in data folder

```
cd src/ds
python distant_supervision.py --comet --comet-location ../../data --target_dir ../../data/atomic 
```

#### Processing Data for Training Models 

Combine distantly supervised data into a single file "all_data.jsonl" by running combine_files.py in ds folder  

Split data using split.py in ds folder 

Change format between comet and heuristic data by setting comet = True or comet = False in split.py file 

For comet data, files are in the format "c_h_atomic_{split}.txt"

For heuristic data, files are in the format "h_atomic_{split}.txt"

#### Train (w/o Memory)

```
cd src/gpt (or src/gpt2) 
python finetune_model.py --log_dir ./log --model_dir ./models --data_dir ../../data --use_mem False --comet True 
```
#### Train (Memory)

```
cd src/gpt (or src/gpt2) 
python finetune_model.py --log_dir ./mem_log --model_dir ./mem_models --data_dir ../../data --use_mem True --comet True
```

#### Decode (w/o Memory) 

```
cd src/gpt (or src/gpt2) 
python decode.py --model_type ./models/model --original_file '../../data/c_atomic_test.jsonl' --data_dir ../../data --save_dir ../../data/gen_data --save_filename 'outputs.jsonl' --load_epoch 8 
```
#### Decode (Memory)

```
cd src/gpt (or src/gpt2) 
python decode.py --model_type ./mem_models/model --original_file '../../data/c_atomic_test.jsonl' --data_dir ../../data --save_dir ../../data/gen_data --save_filename 'outputs.jsonl' --load_epoch 9 --use_mem True
```

#### Evaluation Data 

./data/gold_set.jsonl 

#### Pretrained Models 

|Model Type          | Link |   
|--------------------|------|
|Para-M (w Memory)   | link |
|Para-M (w/o Memory) | link |

### References 

Please cite this repository using the following reference:

```
@inproceedings{Gabriel2021ParagraphLevelCT,
title={Paragraph-level Commonsense Transformers with Recurrent Memory},
author={Gabriel, Saadia and Bhagavatula, Chandra and Shwartz, Vered and Le Bras, Ronan and Forbes, Maxwell and Choi, Yejin},
booktitle={AAAI},
year={2021},
}
```
