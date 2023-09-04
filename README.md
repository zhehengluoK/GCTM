# Contrastive Learning for Neural Topic Model
This repository contains the implementation of the paper [Contrastive Learning for Neural Topic Model](https://arxiv.org/abs/2110.12764).

[Thong Nguyen](https://nguyentthong.github.io/), [Luu Anh Tuan](https://tuanluu.github.io/) (NeurIPS 2021)


In this work, we target the problem of capturing meaningful representations through modeling the relations among samples from a mathematical perspective and propose a novel contrastive objective to train the neural topic model, along with the optimization of the variational lower bound. In our contrastive learning framework, we introduce a novel sampling strategy that is motivated by human behavior when comparing numerous documents. Our results show that capturing mutual information between the prototype and its positive sample provides a strong foundation for constructing coherent topics, while differentiating the prototype from the negative samples plays a less fundamental role.

```
@inproceedings{
nguyen2021contrastive,
title={Contrastive Learning for Neural Topic Model},
author={Thong Thanh Nguyen and Anh Tuan Luu},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=NEgqO9yB7e}
}
```

## Requirements
- python3
- pandas
- gensim
- numpy
- torchvision
- pytorch
- scipy

more in requirements.txt 


## How to Run
1. Download and put the dataset in the ```data``` folder: https://www.dropbox.com/scl/fi/mskgptr4zuk8igqdr3ieb/data.zip?rlkey=rt8ybdev3bkrg87wbjwd3yczu&dl=0
2. process nips dataset: ```python preprocess_data.py data/nips/train.jsonlist data/nips/processed/ --vocab-size 10000 --test data/nips/test.jsonlist``` \
  process 20ng ```python preprocess_data.py data/20ng/20ng_all/train.jsonlist data/20ng/processed/ --vocab-size 2000 --label group --test data/20ng/20ng_all/test.jsonlist``` \
  process IMDB dataset：```python preprocess_data.py data/imdb/train.jsonlist data/imdb/processed/ --vocab-size 5000 --label sentiment --test data/imdb/test.jsonlist```
3. to use word2vec download GoogleNews-vectors-negative300.bin.gz from https://code.google.com/archive/p/word2vec/ 
4. run ```python build_graph_new.py DATASET WINDOW_SIZE THRESHOLD```(For example: ```python build_graph_new.py imdb 20 0.0```) and ```python get_bert_embeddings.py```
5. change permissions if be denied: \
```chmod u+x scripts/train_models/run_20ng_gcn.sh``` \
```chmod u+x scripts/evaluate/run_20ng_npmi_gcn.sh```
6. Train the model by running ```scripts/train_models/run_20ng_gcn.sh```
7. Evaluate the model via executing ```scripts/evaluate/run_20ng_npmi_gcn.sh```

### recommended parameters

| dataset |      LR       |        epochs         | label | word2vec |
|:-------:|:-------------:|:---------------------:|:-----:|:--------:|
|  20NG   | 0.001 / 0.002 | 300 400 500 / 200 300 |  no   |    no    |
|  IMDB   |     0.001     |        300 400        |       |          |
|  NIPS   | 0.004 / 0.005 |        400 500        |       |          |


## Acknowledgement
Our implementation is based on the official code of [SCHOLAR](https://github.com/dallascard/scholar).
