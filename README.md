# CoCoS: Contrastive Context Sharing

<p align="center"><img alt='A CoCoS exmaple' width="60%" src="assets/CoCoS_node_classification_example.png" /></p>

This is the code repository of **AAAI 2022** paper 'CoCoS: Enhancing Semi-Supervised Learning on Graphs with Unlabeled Data via Contrastive Context Sharing'.
CoCos is an enhancement technique for GNN-based models.
It aims to improve GNN's representation capacity and performance when labeled nodes are scarce in a given dataset.
The main task is for node classifications.
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

The paper will be available soon on the AAAI library.

## A Brief Introduction for CoCoS

### Motivation

The idea is inspired by some observations on experiments.
For current GNN models under semi-supervised settings, models are usually trained by the supervised classification loss
that is based only on the information of labeled nodes (as well as its neighborhoods).
However, when the labeled nodes are very sparse in the graph (which can be true in many real-world scenarios), the 
nodes that are incorporated in the training can be very limited.
In other words, the model can only learn knowledge from a small part of available data, while leaving all other data
idle.
Such drawbacks limit the learning capacity of GNN models. 
Some observations, quantitative analyses and detailed explanations refer to our paper.
To make use of all available data and improve the representation power of current GNN models on 
sparse-labeled datasets, we propose to enhance the learning capacity by learning from unlabeled nodes.


### Methodology

<p align="center"><img alt='CoCoS framework' width="80%" src="assets/CoCoS_framework.png" /></p>


To learn from all available data, our idea is to find a way to correlate the labeled and unlabeled nodes.
Note that GNN is able to classify a node by looking at both the target node itself as well as the context
(neighborhood information) of the target node.
From such a starting point, we argue that GNN is still able to classify a node even if perturbations occur in its 
neighborhood, as long as the context is still consistent with that before perturbations.
Here, 'context' refers to an instance's neighborhood information, including node/ edge attributes and the graph topology.
To introduce perturbations but keep the context consistent, our idea is to shuffle the nodes that within the same 
class, i.e., intra-class node shuffling.
(An illustrative example for such an idea can refer to the Figure 2 in our paper.)
However, for graph learning under semi-supervised settings, only a small proportion of nodes have labels.
An workaround is to estimate the labels for those unlabeled nodes.
Therefore, in our method, we will first use a pretrain GNN backbone model to generate the label estimation for each node.
With these estimated labels, we can apply intra-class shuffling.
We argue that the graph after such an intra-class shuffling (we call it context-consistent graph) shares the same context with the original graph.
This helps correlate the labeled and unlabeled nodes, even if they are remote from each other in the graph.
Also, more nodes can be incorporated into the learning process, which enlarge the domain that the model can learn from.
In addition, we can draw analogy of such a context sharing scheme to the data augmentation in contrastive learning.
Therefore, we borrow the ideas from advanced graph contrastive works to further enhance the framework.
Concretely, we can construct some positive pairs between the context consistent graph and the original graph, and also 
the nodes within the same class.
Negative pairs can be constructed across nodes from different class (intra-class shuffling).
With these positive and negative pairs, we can derive a contrastive loss and include it together with the supervised 
classification loss for training, so as to enhance the learning capacity of the GNN backbone model.

The steps (refer to the above figure) to apply CoCos can be summarized as follows:
1. Pretraining stage: select a GNN backbone model (which is going to enhanced by CoCoS) and train it in the target 
dataset with limited labeled data.
After training, the GNN backbone model can provide an estimated label for each node in the graph.
2. Context sharing: with the estimated labels, we apply intra-class shuffling on the given graph, which will generate
a context-consistent graph.
3. Contrast: the context-consistent graph can be treated as a data augmentation against the original graph.
Therefore, we construct some positive paris and negative pairs between the original and the context-consistent graph.
4. Training: with the contrastive pairs, we can derive the contrastive loss.
The part of loss will be added to the overall loss together with the supervised classification loss on labeled nodes.
The model can then be trained through backward-propagation based on the overall loss function (refer to Eq 11 in our 
paper).


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

### Quick Start: Run GCN and Enhance GCN with CoCoS

CoCoS is an GNN enhancement techniques, which will be applied to a pretrained GNN model.
Let's take GCN on Cora as an example, steps and commands are as follows:
1. Pre-train GCN on Cora: `python --dataset cora --model GCN`.
2. Enhance GCN with CoCoS: `python --dataset cora --model GCNCoCoS`.

Step 1 will will create a folder `GCNCoCoS_ori` under `./exp/` to store all experimental results, 
including hyperparameters, accuracies and model parameters.
Step 2 will create a folder `GCN_ori` under `./exp/` to store all experimental results, 
including hyperparameters, accuracies and model parameters.

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

- `main.py`: the main program to run an experiment. Optional arguments:
    - --model: the name of the model for training. 
    Currently support baseline GNN models include `GCN`, `GAT`, `SAGE` (GraphSAGE), `JKNet` and `SGC`. 
    For CoCoS-enhanced models, add an suffix 'CoCoS' to each baseline model's name, e.g., `GCNCoCoS`.
    Default: `GCN`.
    - --dataset: the name of datasets.
    Options include `cora` (Cora), `citeseer` (Citeseer), `pubmed` (Pubmed), `amz-computer` (Amazon-Computer),
    `amz-photo` (Amazon-Photos), `co-cs` (Coauthor-CS), `co-phy` (Coauthor-Physics), `ogbn-arxiv` (Ogbn-arxiv).
    Default: `cora`. 
    - --pretr_state: the version of pretrained GNN model (only for CoCoS-enhanced models). 
    In the pretrain stage, we will store two versions of the pretrained model, 
    one is the version with the best validation accuracy, the other is the version we stored
    after the last training epoch.
    Use 'val' and 'fin' to specify these two version respectively.
    Default: `val`.
    - --n_epochs: the total number of training epochs.
    Default: `300`.
    - --eta: the number of epochs to override/ update the estimated labels for each unlabeled node, corresponding to 
    the hyperparameter &eta; in the paper.
    Only for CoCoS-enhanced model.
    Default: `10`.
    - --n_cls_pershuf: the number of classes for each context shuffling operation (only for experiments on 
    Ogbn-arxiv dataset).
    Only for CoCoS-enhanced model.
    Default: `4`.
    - --n_layers: the number of layers of the specified model.
    Default: `2`.
    - --cls_layers: the number of layers of MLP classifier (only for JKNet).
    Default: `1`.
    - --dis_layers: the number of layers of the discriminator for contrastive learning.
    Default: `2`.
    Only for CoCoS-enhanced model.
    - --hid_dim: the hidden dimension of each GNN layer.
    Default: `16`.
    - emb_hid_dim: the hidden dimension of each layer in the discriminator.
    Default: `32`.
    Only for CoCoS-enhanced model.
    - --dropout: dropout rate.
    Default: `0.6`.
    - --input_drop: the input dropout rate, only for models for Ogbn-arxiv.
    Default: `0.25`.
    - --attn_drop: the attention dropout rate, only for GAT.
    Default: `0.6`.
    - --edge_drop: the edge dropout rate, only for GAT in Ogbn-arxiv.
    Default: `0.3`.
    - --num_heads: the number of attention heads for GAT.
    Default: `8`.
    - --agg_type: the aggregation type of each GraphSAGE layer (`sageconv` module in DGL).
    Only for GraphSAGE
    Default: `gcn`.
    - --gpu: specify the GPU index for model training.
    Use `-1` for CPU training.
    Default: `-1`.
    - --lr: the learning rate.
    Default: `0.01`.
    - --weight_decay: the weight decay for the optimizer.
    Default: `5e-4`.
    - --seed: the random seed for results reproduction.
    Default: `0`.
    - --split: the train-validation split for the experiment.
    The format should be {tr}-{val}, e.g., 0.6-0.2, which means 60\% of data (nodes) for training, 20\% for
    validation and remaining 20\% data for testing.
    Use `None` for the standard split.
    Default: `None`.
    - --alpha: the coefficient for the contrastive loss part, corresponding to the hyperparamter &alpha; in the paper.
    Only for CoCoS-enhanced model.
    Default: `0.6`.
    - --cls_mode: the version of classification loss.
    Options include: `raw`, `shuf` and `both`.
    `raw` indicates that the classification loss is computed based on the unshuffled node features.
    `shuf` indicates that the classification loss is computed based on the shuffled node features.
    `both` combines the above two modes, which is the version used in the paper.
    Default: `both`.
    - --ctr_mode: the type of positive pairs for contrastive learning.
    Options: `F`, `T`, `M`, `S` or their combinations, e.g., `MS`.
    Default: `FS`.
    - --bn: a flag to indicate whether to use batch normalization for training.

- `run_experiments.py`: a script to repeatedly run experiments using the same set of hyperparameters but with different 
random seed, which is for experiments reproduction.
Optional arguments are the same as `main.py`, but remove `--seed` argument and add an additional argument:
    - --round: specify how many rounds you would like to repeat the experiments.
    Each round will use the same set of hyperparameters but with different random seed.
    Default: `10`.

- `trainers.py`: a script includes all required trainers for experiments on different datasets using different mdoels.

- `models_small.py`: a script includes all required models for training on small datasets, i.e., Cora, Citeseer, Pubmed,
Amazon-computers, Amazon-photos, Coauthor-CS and Coauthor-Physics.

- `models_ogb.py`: a script includes all required models for training on Ogbn-arxiv dataset.
Some modules are adapted from [here](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv). 
    
- `utils.py`: other utilizations.

- `./data_prepare/data_preparation.py`: script for data pre-processing and loading.

- `./dataset`: the folder that stores the downloaded dataset.
Will be created in the first time to run the experiment.

- `./exp`: the directory to hold all experimental results.

### Datasets


