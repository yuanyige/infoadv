# Towards Generalizable Graph Contrastive Learning: An Information Theory Perspective

> Yige Yuan, Bingbing Xu, Huawei Shen, Qi Cao, Keting Cen, Wen Zheng, Xueqi Cheng
> 
> Neural Networks (NN), Volume 172, April 2024

This is an official PyTorch implementation of paper [Towards Generalizable Graph Contrastive Learning: An Information Theory Perspective](https://arxiv.org/abs/2211.10929).

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

## Reference
If you find our work useful, please consider citing our paper:
```
@article{yuan2022towards,
  title={Towards Generalizable Graph Contrastive Learning: An Information Theory Perspective},
  author={Yuan, Yige and Xu, Bingbing and Shen, Huawei and Cao, Qi and Cen, Keting and Zheng, Wen and Cheng, Xueqi},
  journal={arXiv preprint arXiv:2211.10929},
  year={2022}
}
```
