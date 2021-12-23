'''
The models in this script is for small dataset, i.e., Cora, Citeseer, Pubmed, Amazon-computer, Amazon-photos,
Coauthor-CS and Coauthor-Physics.
'''
import torch
from torch.nn import functional as F
from torch import nn
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv


class LinearBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0, bn=False, residual=False, act=True):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_dim, hid_dim)
        self.bn = nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU() if act else nn.Identity()
        self.residual = residual

    def forward(self, x):
        x = self.dropout(x)
        new_x = self.linear(x)
        new_x = self.bn(new_x)
        if self.residual:
            x = x + new_x
        else:
            x = new_x
        x = self.act(x)
        return x


class SAGEBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0, agg='gcn', bn=False, residual=False, act=True):
        super(SAGEBlock, self).__init__()
        self.conv = SAGEConv(in_dim, hid_dim, aggregator_type=agg, feat_drop=dropout)
        self.bn = nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        self.residual = residual

    def forward(self, g, x):
        new_x = self.conv(g, x)
        if self.residual:
            x = x[: g.num_dst_nodes()] + new_x
        else:
            x = new_x

        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        self.conv.reset_parameters()


class GCNBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout=0, norm='both', bn=False, residual=False, act=True):
        super(GCNBlock, self).__init__()
        self.conv = GraphConv(in_dim, hid_dim, norm=norm)
        self.bn = nn.BatchNorm1d(hid_dim) if bn else nn.Identity()
        self.act = nn.ReLU() if act else nn.Identity()
        self.residual = residual
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        x = self.dropout(x)
        new_x = self.conv(g, x)
        if self.residual:
            x = x[: g.num_dst_nodes()] + new_x
        else:
            x = new_x
        x = self.bn(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        self.conv.reset_parameters()


class SAGE(nn.Module):
    def __init__(self, info_dict):
        super(SAGE, self).__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if i == info_dict['n_layers'] - 1 else info_dict['hid_dim']
            act = False if i == (info_dict['n_layers'] - 1) else True
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(SAGEBlock(in_dim, out_dim, info_dict['dropout'], info_dict['agg_type'], bn=bn, act=act))

    def forward(self, graph, feat):
        h = feat
        for i, layer in enumerate(self.enc):
            h = layer(graph, h)
        return h

    def reset_param(self):
        for name, module in self.enc.named_children():
            if module._get_name() == 'SAGEBlock':
                module.reset_parameters()


class GCN(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if i == info_dict['n_layers'] - 1 else info_dict['hid_dim']
            act = False if i == (info_dict['n_layers'] - 1) else True
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(GCNBlock(in_dim, out_dim, info_dict['dropout'], bn=bn, act=act))

    def forward(self, graph, feat):
        h = feat
        for i, layer in enumerate(self.enc):
            h = layer(graph, h)
        return h

    def reset_param(self):
        for name, module in self.enc.named_children():
            if module._get_name() == 'GCNBlock':
                module.reset_parameters()


class GAT(nn.Module):
    def __init__(self, info_dict):
        super(GAT, self).__init__()
        self.info_dict = info_dict
        self.num_layers = info_dict['n_layers']
        self.gat_layers = nn.ModuleList()
        self.activation = F.relu

        for l in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if l == 0 else info_dict['hid_dim'] * info_dict['num_heads']
            out_dim = info_dict['out_dim'] if l == info_dict['n_layers'] - 1 else info_dict['hid_dim']
            act = nn.Identity() if l == (info_dict['n_layers'] - 1) else self.activation
            num_heads = info_dict['num_heads'] if l < info_dict['n_layers'] - 1 else 1
            self.gat_layers.append(GATConv(
                in_dim, out_dim, num_heads,
                info_dict['dropout'], info_dict['att_drop'], 0.2, False, activation=act))

    def forward(self, graph, feat):

        h = feat
        for l in range(self.num_layers):
            if l < self.num_layers - 1:
                h = self.gat_layers[l](graph, h).flatten(1)
            else:
                # output projection
                h = self.gat_layers[l](graph, h).mean(1)
        return h

    def reset_param(self):
        for name, module in self.gat_layers.named_children():
            if module._get_name() == 'GATConv':
                module.reset_parameters()


class JKNet(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['hid_dim']
            act = True
            bn = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.enc.append(GCNBlock(in_dim, out_dim, info_dict['dropout'], bn=bn, act=act))

        # use the concatenation version of JK-Net
        rep_dim = info_dict['hid_dim'] * (info_dict['n_layers'])
        self.classifier = nn.Linear(rep_dim, info_dict['out_dim'])
        self.reset_param()

    def forward(self, graph, feat):
        h = feat
        feat_lst = []
        for i, layer in enumerate(self.enc):
            h = layer(graph, h)
            feat_lst.append(h)
        h = torch.cat(feat_lst, dim=-1)

        h = self.classifier(h)
        return h

    def reset_param(self):
        self.classifier.reset_parameters()
        for layer in self.enc:
            layer.reset_parameters()


class SGC(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = SGConv(info_dict['in_dim'], info_dict['out_dim'], k=info_dict['n_layers'], cached=True, bias=True)

    def forward(self, graph, feat):
        h = feat
        h = self.enc(graph, h)
        return h

    def reset_param(self):
        self.enc.reset_parameters()


class MLP(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.mlp = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            input_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            hidden_dim = info_dict['out_dim'] if i == (info_dict['n_layers'] - 1) else info_dict['hid_dim']
            act = False if i == (info_dict['n_layers'] - 1) else True
            self.mlp.append(LinearBlock(input_dim, hidden_dim, 0, act=act))

    def forward(self, feat):
        h = feat
        for layer in self.mlp:
            h = layer(h)

        return h

    def reset_param(self):
        for layer in self.mlp:
            layer.reset_parameters()


class DisMLP(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.mlp = nn.ModuleList()

        for i in range(info_dict['dis_layers']):
            input_dim = 2 * info_dict['out_dim'] if i == 0 else info_dict['emb_hid_dim']
            hidden_dim = 1 if i == (info_dict['dis_layers'] - 1) else info_dict['emb_hid_dim']
            act = False if i == (info_dict['dis_layers'] - 1) else True
            self.mlp.append(LinearBlock(input_dim, hidden_dim, 0, act=act))

    def forward(self, feat):
        h = feat
        for layer in self.mlp:
            h = layer(h)

        return h

    def reset_param(self):
        for layer in self.mlp:
            layer.reset_parameters()