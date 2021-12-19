import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import networkx as nx
import dgl
import argparse
import os
from data_prepare import data_preparation
from tqdm import tqdm
import pickle as pkl
import time


def gen_ego_neighbors(args):
    # _ = get_egodata(args.dir, args.dataset, hops=args.hops)
    _ = get_egodata(args.save_dir, args.dataset, args.hops, db_root=args.src_dir)


# def get_egodata(path, db, g=None, info_dict=None, hops=None):
#     if info_dict is None and hops is None:
#         raise ValueError('info_dict and hops cannot be both None')
#     elif info_dict is not None:
#         hops = info_dict['n_layers']
#
#     egoneb_fname = os.path.join(path, db, 'egoneighbors_{}hops.pkl'.format(hops))
#     if os.path.exists(egoneb_fname):
#         # load ego-net neighbors data if existed
#         with open(egoneb_fname, 'rb') as f:
#             egoneb_list_all = pkl.load(f)
#     else:
#         # generate ego-net neighbors file and store it in the hard disk
#         egoneb_root = os.path.join(path, db)
#         if not os.path.exists(egoneb_root):
#             os.makedirs(egoneb_root)
#         if info_dict is None:
#             dgl_g, info_dict = data_preparation.data_prep(db)
#             g = dgl.to_networkx(dgl_g)
#         else:
#             hops = info_dict['n_layers']
#
#         egoneb_list_all = []
#         max_neb_num = 0
#         nodes_list = list(g.nodes)
#         print('Start extracting neighbors of each ego-net ({})'.format(db))
#         for step, node_i in enumerate(tqdm(nodes_list)):
#             ego_net_i = nx.generators.ego.ego_graph(g, node_i, hops, center=True, undirected=True)
#             egoneb_list_i = list(ego_net_i.nodes())
#             egoneb_list_all.append(egoneb_list_i)
#             max_neb_num = len(egoneb_list_i) if len(egoneb_list_i) > max_neb_num else max_neb_num
#         print('The second round traversal (padding)...')
#         for i, egoneb_list_i in enumerate(tqdm(egoneb_list_all)):
#             egoneb_list_all[i].extend([-1] * (max_neb_num - len(egoneb_list_i)))
#
#         with open(egoneb_fname, 'wb') as f:
#             pkl.dump(egoneb_list_all, f)
#
#     return egoneb_list_all


def get_egodata(path, db, num_hops, g=None, db_root=None):
    # egoneb_fname = os.path.join(path, db, 'egoneighbors_{}hops.txt'.format(num_hops))
    egoneb_fname = os.path.join(path, db, 'egoneighbors_{}hops.pkl'.format(num_hops))
    if os.path.exists(egoneb_fname):
        # load ego-net neighbors data if existed
        print('Find ego-net neighbor indices file, loading...')
        tic = time.time()
        with open(egoneb_fname, 'rb') as f:
            egoneb_list_all = pkl.load(f)
        toc = time.time()
        print('Successfully load it. Elapse time: {:.4f}s'.format(toc - tic))
        # with open(egoneb_fname, 'r') as f:
        #     egoneb_list_all = []
        #     for line in f.readlines():
        #         line = line.strip().split(',')
        #         egoneb_list_all.append([int(nid) for nid in line])
    else:
        # generate ego-net neighbors file and store it in the hard disk
        egoneb_root = os.path.join(path, db)
        if not os.path.exists(egoneb_root):
            os.makedirs(egoneb_root)
        if g is None:
            dgl_g, _ = data_preparation.data_prep(db, db_root)
            g = dgl.to_networkx(dgl_g)

        print('\nstart extracting ego-net neighbors from {} dataset'.format(db))
        egoneb_list_all = []
        max_neb_num = 0
        nodes_list = list(g.nodes)
        neblist_all = []

        # get neighbors of each node
        print('\nextract neighbors for each node...')
        tic = time.time()
        for node_i in tqdm(nodes_list):
            neblist_all.append(list(g.neighbors(node_i)))
        toc = time.time()
        print('Elapse time: {:.4f}s'.format(toc - tic))

        print('\nextract neighbors for each ego-net...')
        tic = time.time()
        for step, node_i in enumerate(tqdm(nodes_list)):
            egoneb_list_i = [node_i]
            last_sum = 0
            for hop in range(num_hops):
                # get all leave nodes in the current hop
                hop_neb_list = []
                for egoneb in egoneb_list_i[last_sum:]:
                    neb_neb_list = neblist_all[egoneb]
                    hop_neb_list.extend(list(set(neb_neb_list).difference({egoneb})))
                last_sum = len(egoneb_list_i)
                egoneb_list_i.extend(hop_neb_list)
            # remove repeated node indices
            egoneb_list_i = list(set(egoneb_list_i))
            egoneb_list_all.append(egoneb_list_i)
            max_neb_num = len(egoneb_list_i) if len(egoneb_list_i) > max_neb_num else max_neb_num
        toc = time.time()
        print('Elapse time: {:.4f}s'.format(toc - tic))

        print('\nThe second round traversal (saving)...')
        tic = time.time()
        egoneb_list_all.insert(0, [max_neb_num])
        # with open(egoneb_fname, 'w') as f:
        #     for i, egoneb_list_i in enumerate(tqdm(egoneb_list_all)):
        #         # egoneb_list_all[i].extend([-1] * (max_neb_num - len(egoneb_list_i)))
                # f.write(','.join([str(nid) for nid in egoneb_list_all[i]]) + '\n')

        with open(egoneb_fname, 'wb') as f:
            pkl.dump(egoneb_list_all, f)
        toc = time.time()
        print('Elapse time: {:.4f}s'.format(toc - tic))

    return egoneb_list_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification with triple branch architecture')
    parser.add_argument("--dataset", type=str, default='ogbn-arxiv',
                        help="dataset for experiment")
    parser.add_argument("--save_dir", type=str, default='./',
                        help="the root directory of stored ego-net neighbors data")
    parser.add_argument("--src_dir", type=str, default='../../dataset',
                        help="the root directory of the dataset (only works for OGB)")
    parser.add_argument("--hops", type=int, default=2,
                        help="the hops of induced graph")
    args = parser.parse_args()

    gen_ego_neighbors(args)