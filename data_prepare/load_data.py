import torch
import networkx as nx
import dgl
from torch.utils.data import Dataset, DataLoader
import pickle as pkl


class EgoNetDataset(Dataset):
    def __init__(self, g, info_dict, state='tr'):
        self.g = g
        self.nxg = dgl.to_networkx(g)
        self.hops = info_dict['n_layers']
        self.labels = g.ndata['label']

        if state == 'tr':
            self.nids = g.ndata['train_mask'].nonzero().squeeze()
        elif state == 'val':
            self.nids = g.ndata['val_mask'].nonzero().squeeze()
        elif state == 'tt':
            self.nids = g.ndata['test_mask'].nonzero().squeeze()
        else:
            raise ValueError('Unknown state: {}'.format(state))

    def __getitem__(self, idx):
        '''
        return the ego-net of the target node and its corresponding label
        '''

        # the center/ target node of the ego-net
        tar_id = self.nids[idx].item()
        # sample ego-net with the help of networkx
        ego_net_nx = nx.generators.ego.ego_graph(self.nxg, tar_id, self.hops, center=True, undirected=True)
        # indices of neighboring nodes in the ego-net
        neighbor_list = list(ego_net_nx.nodes)

        # neighbor indices flag for a sample
        ego_ind = torch.zeros(self.g.num_nodes()).scatter_(0, torch.LongTensor(neighbor_list), 1).bool()
        # center node flag for a sample
        cen_flg = torch.zeros(self.g.num_nodes()).scatter_(0, torch.LongTensor([tar_id]), 1).bool()

        return ego_ind, cen_flg, self.g.ndata['label'][tar_id]

        # # center node flag for computations
        # center_flag = torch.zeros(len(neighbor_list))
        # center_flag[neighbor_list.index(tar_id)] = 1
        # center_flag = center_flag.bool()
        # # convert the ego-net to dgl graph
        # ego_net_dgl = dgl.from_networkx(ego_net_nx)
        # # load features and flags into the newly created ego-net graph
        # ego_net_dgl.ndata['feat'] = self.g.ndata['feat'][torch.LongTensor(neighbor_list)]
        # ego_net_dgl.ndata['cen'] = center_flag
        # return ego_net_dgl, self.g.ndata['label'][tar_id]

    def __len__(self):
        return self.nids.shape[0]


class EgoNebDataset(Dataset):
    def __init__(self, g, egoneb_list_all, max_nebnum, state='tr'):
        self.labels = g.ndata['label']
        self.egoneb_list_all = egoneb_list_all
        self.max_neb_num = max_nebnum

        if state == 'tr':
            self.nids = g.ndata['train_mask'].nonzero().squeeze()
        elif state == 'val':
            self.nids = g.ndata['val_mask'].nonzero().squeeze()
        elif state == 'tt':
            self.nids = g.ndata['test_mask'].nonzero().squeeze()
        else:
            raise ValueError('Unknown state: {}'.format(state))

    def __getitem__(self, idx):
        # the center/ target node of the ego-net
        tar_id = self.nids[idx].item()
        self.egoneb_list_all[tar_id].extend([-1] * (self.max_neb_num - len(self.egoneb_list_all[tar_id])))
        egoneb_list = torch.LongTensor(self.egoneb_list_all[tar_id])
        label = self.labels[tar_id]
        return tar_id, label, egoneb_list

    def __len__(self):
        return self.nids.shape[0]
