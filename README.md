## Paragraph-Level Commonsense Transformers with Recurrent Memory 

This repository contains the code used in the paper:

Paragraph-Level Commonsense Transformers with Recurrent Memory. *Saadia Gabriel, Chandra Bhagavatula, Vered Shwartz, Ronan Le Bras, Maxwell Forbes, Yejin Choi*. AAAI 2021. [link] (https://arxiv.org/abs/2010.01486)

This is a general purpose framework for aligning commonsense knowledge in the ATOMIC knowledge graph with narrative text. The repo contains 

1) A framework for distantly supervised paragraph-level commonsense knowledge alignment; and 
2) Modeling code for finetuning pretrained transformers to generate paragraph-level commonsense inferences. 

### Instructions 

[COMING SOON] 

python 3.6, pytorch >= 1.0 (I suggest using a virtual environment) 

1. pip install -r requirements.txt 
2. python -m spacy download en
3. wget https://homes.cs.washington.edu/~msap/atomic/data/atomic_data.tgz ./data/atomic_data.tgz 

#### Distant Supervision (Heuristic) 

#### Distant Supervision (COMeT) 

1. git clone https://github.com/allenai/comet-public.git
2. wget https://storage.googleapis.com/ai2-mosaic/public/comet/models.zip
3. unzip models.zip 

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
