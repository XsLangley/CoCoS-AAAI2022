import numpy as np
import dgl
from dgl.data import AmazonCoBuyComputerDataset
from dgl.data import AmazonCoBuyPhotoDataset
from dgl.data import CoauthorCSDataset
from dgl.data import CoauthorPhysicsDataset
from dgl.data import citegrh
from dgl.data import RedditDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import torch
import random


def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g


def load_CoAuthor(db, db_dir):
    if db == 'co-cs':
        dataset = CoauthorCSDataset(raw_dir=db_dir)
    elif db == 'co-phy':
        dataset = CoauthorPhysicsDataset(raw_dir=db_dir)
    else:
        raise ValueError('Unknown dataset: {}'.format(db))

    return dataset


def load_AMZCoBuy(db, db_dir):
    if db == 'amz-computer':
        dataset = AmazonCoBuyComputerDataset(raw_dir=db_dir)
    elif db == 'amz-photo':
        dataset = AmazonCoBuyPhotoDataset(raw_dir=db_dir)
    else:
        raise ValueError('Unknown dataset: {}'.format(db))

    return dataset


def load_citation(db):
    if db == 'cora':
        return citegrh.load_cora()
    elif db == 'citeseer':
        return citegrh.load_citeseer()
    elif db == 'pubmed':
        return citegrh.load_pubmed()
    else:
        raise ValueError('Unknown dataset: {}'.format(db))


def load_ogb(db, db_dir):
    dataset = DglNodePropPredDataset(name=db, root=db_dir)
    return dataset


def citation_prep(db):
    data = load_citation(db)
    g = data[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    info_dict = {'in_dim': g.ndata['feat'].shape[1],
                 'out_dim': data.num_classes, }
    return g, info_dict


def amz_prep(db, db_dir):
    data = load_AMZCoBuy(db, db_dir)
    g = data[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    info_dict = {'in_dim': g.ndata['feat'].shape[1],
                 'out_dim': g.ndata['label'].max().item() + 1, }
    # g.ndata['feat'] = g.ndata['feat'] / g.ndata['feat'].shape[1]
    g.ndata['feat'] = g.ndata['feat'] / g.ndata['feat'].sum(dim=1, keepdim=True)
    return g, info_dict


def coauthor_prep(db, db_dir):
    data = load_CoAuthor(db, db_dir)
    g = data[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    info_dict = {'in_dim': g.ndata['feat'].shape[1],
                 'out_dim': data.num_classes, }
    # g.ndata['feat'] = g.ndata['feat'] / g.ndata['feat'].shape[1]
    g.ndata['feat'] = g.ndata['feat'] / g.ndata['feat'].sum(dim=1, keepdim=True)
    return g, info_dict


def ogb_prep(db, db_dir):
    dataset = load_ogb(db, db_dir)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    # the original graph is a directional graph, we should convert it to a bi-directional graph
    graph, label = dataset[0]

    # region my code to add reverse edges

    # srcs, dsts = graph.all_edges()
    # ori_edges = set(list(zip(srcs.numpy().tolist(), dsts.numpy().tolist())))
    # rev_edges = set(list(zip(dsts.numpy().tolist(), srcs.numpy().tolist())))
    # mis_edges = ori_edges.symmetric_difference(rev_edges).difference(ori_edges)
    # mis_edges = torch.LongTensor(list(zip(*list(mis_edges))))
    # graph.add_edges(mis_edges[0], mis_edges[1])

    # endregion

    # region official code of dgl to add reverse edges, including adding the self-loop

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    # endregion

    info_dict = {'in_dim': graph.ndata['feat'].shape[1],
                 'out_dim': dataset.num_classes,
                 'evaluator': Evaluator(name=db),
                 'src_root': db_dir}

    train_mask = torch.zeros(label.shape[0]).scatter_(0, train_idx, torch.ones(train_idx.shape[0]))
    valid_mask = torch.zeros(label.shape[0]).scatter_(0, valid_idx, torch.ones(valid_idx.shape[0]))
    test_mask = torch.zeros(label.shape[0]).scatter_(0, test_idx, torch.ones(test_idx.shape[0]))
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = valid_mask
    graph.ndata['test_mask'] = test_mask
    graph.ndata['label'] = label.squeeze()

    return graph, info_dict


def reddit_prep():
    data = RedditDataset(self_loop=True)
    g = data[0]
    info_dict = {'in_dim': g.ndata['feat'].shape[1],
                 'out_dim': data.num_classes, }
    return g, info_dict


def data_prep(db, db_dir=None):
    small_graph = ['cora', 'citeseer', 'pubmed']
    amz_graph = ['amz-computer', 'amz-photo']
    coauthor_graph = ['co-cs', 'co-phy']
    ogb_graph = ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-mag']
    if db in small_graph:
        g, info_dict = citation_prep(db)
    elif db in ogb_graph:
        g, info_dict = ogb_prep(db, db_dir)
    elif db == 'reddit':
        g, info_dict = reddit_prep()
    elif db in amz_graph:
        g, info_dict = amz_prep(db, db_dir)
    elif db in coauthor_graph:
        g, info_dict = coauthor_prep(db, db_dir)
    else:
        raise ValueError('Unknown dataset: {}'.format(db))

    return g, info_dict


def set_seed_and_split(g, seed, splitstr):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if splitstr == 'None':
        return g

    sp_ratio = splitstr.split('-')
    sp_ratio = [float(r) for r in sp_ratio]
    num_nodes = g.num_nodes()
    if len(sp_ratio) == 1:
        # randomly pick the given number of instances from each class to build the training set

        pass
    else:
        # split the dataset by the given tr-val ratio
        randperm_ind = np.random.permutation(num_nodes)
        tr_end_ind = int(num_nodes * sp_ratio[0])
        val_end_ind = int(num_nodes * sum(sp_ratio))
        split_ind = np.split(randperm_ind, [tr_end_ind, val_end_ind])

        g.ndata['train_mask'] = torch.BoolTensor([False] * g.num_nodes())
        g.ndata['val_mask'] = torch.BoolTensor([False] * g.num_nodes())
        g.ndata['test_mask'] = torch.BoolTensor([False] * g.num_nodes())
        g.ndata['train_mask'][torch.LongTensor(split_ind[0])] = True
        g.ndata['val_mask'][torch.LongTensor(split_ind[1])] = True
        g.ndata['test_mask'][torch.LongTensor(split_ind[2])] = True
    return g
