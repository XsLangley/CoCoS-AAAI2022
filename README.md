# CoCoS: Contrastive Context Sharing

This is the code repository of **AAAI 2022** paper 'CoCoS: Enhancing Semi-Supervised Learning on Graphs with Unlabeled Data via Contrastive Context Sharing'.
CoCos is an enhancement technique for GNN-based models.
It aims to improve GNN's representation capacity and performance when labeled nodes are scarce in a given dataset.
The information of the corresponding paper is as follows:

> Authors: Siyue Xie, Da Sun Handason Tam and Wing Cheong Lau

> Affiliation: The Chinese University of Hong Kong

> Abstract:  Graph Neural Networks (GNNs) have recently become a popular framework for semi-supervised learning on graph-structured data. 
>However, typical GNN models heavily rely on labeled data in the learning process, while ignoring or paying little attention to the data that are unlabeled but available. 
>To make full use of available data, we propose a generic framework, **Co**ntrastive **Co**ntext **S**haring (CoCoS), to enhance the learning capacity of GNNs for semi-supervised tasks. 
>By sharing the contextual information among nodes estimated to be in the same class, different nodes can be correlated even if they are unlabeled and remote from each other in the graph. 
>Models can therefore learn different combinations of contextual patterns, which improves the robustness of node representations. 
>Additionally, motivated by recent advances in self-supervised learning, we augment the context sharing strategy by integrating with contrastive learning, which naturally correlates intra-class and inter-class data. 
>Such operations utilize all available data for training and effectively improve a model's learning capacity. 
>CoCoS can be easily extended to a wide range of GNN-based models with little computational overheads. 
>Extensive experiments show that CoCoS considerably enhances typical GNN models, especially when labeled data are sparse in a graph, and achieves state-of-the-art or competitive results in real-world public datasets. 

The paper will be available soon on AAAI library.

## Progress
- [x] the entrance run script for results reproduction
- [x] the main program for one experiment (only for small datasets, such as Cora, Citeseer and Pubmed)
- [x] data preparation script/ data preprocessing program
- [x] trainers for each model (only for small datasets, such as Cora, Citeseer and Pubmed)
- [x] other utilizations
- [x] Clean up
- [ ] Instructions (the readme file) for this repo

## Requirements

- DGL >= 0.6 (or newer versions)
- PyTorch >= 1.6 (Except v1.10. There are some issues on v1.10 when loading the graph dataset.)
- numpy
- sklearn
- [OGB](https://ogb.stanford.edu/docs/home/)

## Instructions

### Directory Tree
```
CoCoS   
│   main.py  
│   models_ogb.py  
│   models_small.py    
│   README.md
│   run_experiments.py
|   trainers.py
│   utils.py
└───data_prepare
│   └───__init__.py
│   └───data_preparation.py
└───dataset
│   └───Cora
│   └───Citeseer
│   └───...
└───exp
│   └───GCN_ori
│   └───GCNCoCoS_ori
│   └───...
```

### Descriptions of Each File/ Directory

Will be added soon.