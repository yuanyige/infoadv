# Towards Generalizable Graph Contrastive Learning: An Information Theory Perspective

This repository is the official implementation of paper, Towards Generalizable Graph Contrastive Learning: An Information Theory Perspective. 

![InfoAdv](./pic/InfoAdv.png)


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training & Evaluating

To train the model(s) in the paper, run this command:

```train
python -u train.py --dataset Cora  --if_save True  --save_path ./log
```
