import torch
from torch.nn import functional as F
import dgl
import os
from torch import nn
from dgl.utils import expand_as_pair, check_eq_shape
from dgl.ops import edge_softmax
from dgl import function as fn
from dgl.nn.pytorch import SAGEConv
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
        new_x = self.linear(x)
        new_x = self.bn(new_x)
        if self.residual:
            x = x + new_x
        else:
            x = new_x
        x = self.dropout(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        self.linear.reset_parameters()


class ElementWiseLinear(nn.Module):
    def __init__(self, size, weight=True, bias=True, inplace=False):
        super().__init__()
        if weight:
            self.weight = nn.Parameter(torch.Tensor(size))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.Tensor(size))
        else:
            self.bias = None
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.inplace:
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
        else:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
        return x


class SAGEMLP(nn.Module):
    def __init__(self, info_dict, use_linear=False):
        super(SAGEMLP, self).__init__()
        self.classifier = nn.ModuleList()

        self.info_dict = info_dict
        self.n_layers = info_dict['n_layers']
        self.n_hidden = info_dict['hid_dim']
        self.n_classes = info_dict['out_dim']
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            in_hidden = info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_hidden = info_dict['hid_dim']

            self.convs.append(SAGEConv(in_hidden, out_hidden, info_dict['agg_type'], bias=False))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))

            self.norms.append(nn.BatchNorm1d(out_hidden))

        for i in range(info_dict['cls_layers']):
            in_dim = info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if (i == info_dict['cls_layers'] - 1) else info_dict['hid_dim']
            act = False if i == (info_dict['cls_layers'] - 1) else True
            bn = False if i == (info_dict['cls_layers'] - 1) else True
            dropout = 0 if i == (info_dict['cls_layers'] - 1) else info_dict['dropout']
            self.classifier.append(LinearBlock(in_dim, out_dim, dropout, bn=bn, act=act))

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)
        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)

        for i, layer in enumerate(self.classifier):
            h = layer(h)

        return h

    def reset_param(self):
        for name, module in self.convs.named_children():
            if module._get_name() == 'SAGEConv':
                module.reset_parameters()
        for name, module in self.norms.named_children():
            if module._get_name() == 'BatchNorm1d':
                module.reset_parameters()
        for name, module in self.classifier.named_children():
            if module._get_name() == 'LinearBlock':
                module.reset_parameters()

        if self.use_linear:
            for name, module in self.linear.named_children():
                if module._get_name() =='Linear':
                    module.reset_parameters()


class SAGE(nn.Module):
    def __init__(self, info_dict, use_linear=False):
        super().__init__()
        self.info_dict = info_dict
        self.n_layers = info_dict['n_layers']
        self.n_hidden = info_dict['hid_dim']
        self.n_classes = info_dict['out_dim']
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            in_hidden = info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_hidden = info_dict['hid_dim'] if i < info_dict['n_layers'] - 1 else info_dict['out_dim']
            bias = i == info_dict['n_layers'] - 1

            self.convs.append(SAGEConv(in_hidden, out_hidden, info_dict['agg_type'], bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < info_dict['n_layers'] - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h

    def reset_param(self):
        for name, module in self.convs.named_children():
            if module._get_name() == 'SAGEConv':
                module.reset_parameters()
        for name, module in self.norms.named_children():
            if module._get_name() == 'BatchNorm1d':
                module.reset_parameters()

        if self.use_linear:
            for name, module in self.linear.named_children():
                if module._get_name() =='Linear':
                    module.reset_parameters()


class GCN(nn.Module):
    def __init__(self, info_dict, use_linear=False):
        super().__init__()
        self.info_dict = info_dict
        self.n_layers = info_dict['n_layers']
        self.n_hidden = info_dict['hid_dim']
        self.n_classes = info_dict['out_dim']
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            in_hidden = info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_hidden = info_dict['hid_dim'] if i < info_dict['n_layers'] - 1 else info_dict['out_dim']
            bias = i == info_dict['n_layers'] - 1

            self.convs.append(GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if use_linear:
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < info_dict['n_layers'] - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h

    def reset_param(self):
        for name, module in self.convs.named_children():
            if module._get_name() == 'GraphConv':
                module.reset_parameters()
        for name, module in self.norms.named_children():
            if module._get_name() == 'BatchNorm1d':
                module.reset_parameters()

        if self.use_linear:
            for name, module in self.linear.named_children():
                if module._get_name() =='Linear':
                    module.reset_parameters()


class GATConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads=1,
        feat_drop=0.0,
        attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        use_attn_dst=True,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        use_symmetric_norm=False,
    ):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._use_symmetric_norm = use_symmetric_norm
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        if use_attn_dst:
            self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        else:
            self.register_buffer("attn_r", None)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.edge_drop = edge_drop
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias=False)
        else:
            self.register_buffer("res_fc", None)
        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        if isinstance(self.attn_r, nn.Parameter):
            nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    assert False

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src, feat_dst = h_src, h_dst
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = self.feat_drop(feat)
                feat_src = h_src
                feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                if graph.is_block:
                    h_dst = h_src[: graph.number_of_dst_nodes()]
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
                else:
                    h_dst = h_src
                    feat_dst = feat_src

            if self._use_symmetric_norm:
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            if self.attn_r is not None:
                er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
                graph.dstdata.update({"er": er})
                graph.apply_edges(fn.u_add_v("el", "er", "e"))
            else:
                graph.apply_edges(fn.copy_u("el", "e"))
            e = self.leaky_relu(graph.edata.pop("e"))

            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
                graph.edata["a"] = torch.zeros_like(e)
                graph.edata["a"][eids] = self.attn_drop(edge_softmax(graph, e[eids], eids=eids))
            else:
                graph.edata["a"] = self.attn_drop(edge_softmax(graph, e))

            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]

            if self._use_symmetric_norm:
                degs = graph.in_degrees().float().clamp(min=1)
                norm = torch.pow(degs, 0.5)
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval

            # activation
            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GAT(nn.Module):
    def __init__(
        self, info_dict,
        use_attn_dst=False,
        use_symmetric_norm=True,
    ):
        super().__init__()
        self.info_dict = info_dict
        self.in_feats = info_dict['in_dim']
        self.n_hidden = info_dict['hid_dim']
        self.n_classes = info_dict['out_dim']
        self.n_layers = info_dict['n_layers']
        self.num_heads = info_dict['num_heads']

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(info_dict['n_layers']):
            in_hidden = info_dict['num_heads'] * info_dict['hid_dim'] if i > 0 else info_dict['in_dim']
            out_hidden = info_dict['hid_dim'] if i < info_dict['n_layers'] - 1 else info_dict['out_dim']
            num_heads = info_dict['num_heads'] if i < info_dict['n_layers'] - 1 else 1
            out_channels = info_dict['num_heads']

            self.convs.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    num_heads=num_heads,
                    attn_drop=info_dict['attn_drop'],
                    edge_drop=info_dict['edge_drop'],
                    use_attn_dst=use_attn_dst,
                    use_symmetric_norm=use_symmetric_norm,
                    residual=True,
                )
            )

            if i < info_dict['n_layers'] - 1:
                self.norms.append(nn.BatchNorm1d(out_channels * out_hidden))

        self.bias_last = ElementWiseLinear(info_dict['out_dim'], weight=False, bias=True, inplace=True)

        self.input_drop = nn.Dropout(info_dict['input_drop'])
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            h = conv

            if i < self.n_layers - 1:
                h = h.flatten(1)
                h = self.norms[i](h)
                h = self.activation(h, inplace=True)
                h = self.dropout(h)

        h = h.mean(1)
        h = self.bias_last(h)

        return h

    def reset_param(self):
        for name, module in self.convs.named_children():
            if module._get_name() == 'GATConv':
                module.reset_parameters()
        for name, module in self.norms.named_children():
            if module._get_name() == 'BatchNorm1d':
                module.reset_parameters()

        self.bias_last.reset_parameters()


class JKNet(nn.Module):
    def __init__(self, info_dict):
        super().__init__()
        self.info_dict = info_dict
        self.enc = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(info_dict['n_layers']):
            in_dim = info_dict['in_dim'] if i == 0 else info_dict['hid_dim']
            out_dim = info_dict['hid_dim']
            bias = i == info_dict['n_layers'] - 1
            self.enc.append(GraphConv(in_dim, out_dim, 'both', bias=bias))
            self.norms.append(nn.BatchNorm1d(out_dim))

        # use the concatenation version of JK-Net
        rep_dim = info_dict['hid_dim'] * (info_dict['n_layers'])
        for i in range(info_dict['cls_layers']):
            in_dim = rep_dim if (i == 0) else info_dict['hid_dim']
            out_dim = info_dict['out_dim'] if (i == info_dict['cls_layers'] - 1) else info_dict['hid_dim']
            act = False if i == (info_dict['cls_layers'] - 1) else True
            bn = False if i == (info_dict['cls_layers'] - 1) else True
            dropout = 0 if i == (info_dict['cls_layers'] - 1) else info_dict['dropout']
            self.classifier.append(LinearBlock(in_dim, out_dim, dropout, bn=bn, act=act))

        self.input_drop = nn.Dropout(min(0.1, info_dict['dropout']))
        self.dropout = nn.Dropout(info_dict['dropout'])
        self.activation = F.relu

        self.reset_param()

    def forward(self, graph, feat):
        h = feat
        h = self.input_drop(h)
        feat_lst = []
        for i in range(self.info_dict['n_layers']):
            h = self.enc[i](graph, h)
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            feat_lst.append(h)
        h = torch.cat(feat_lst, dim=-1)

        # graph.ndata['h'] = h
        # graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        # h = graph.ndata['h']
        # h = self.classifier(h)
        for i, layer in enumerate(self.classifier):
            h = layer(h)
        return h

    def reset_param(self):
        for layer in self.enc:
            layer.reset_parameters()
        for layer in self.classifier:
            layer.reset_parameters()


class SGC(nn.Module):
    def __init__(self, info_dict):
        super().__init__()

        def normalize(h):
            return (h - h.mean(0)) / h.std(0)

        self.info_dict = info_dict
        self.enc = SGConv(info_dict['in_dim'], info_dict['out_dim'], k=info_dict['n_layers'], cached=True, bias=True, norm=normalize)

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
            BN = False if i == (info_dict['n_layers'] - 1) else info_dict['bn']
            self.mlp.append(LinearBlock(input_dim, hidden_dim, info_dict['dropout'], bn=BN, act=act))

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
            BN = False if i == (info_dict['dis_layers'] - 1) else info_dict['bn']
            self.mlp.append(LinearBlock(input_dim, hidden_dim, info_dict['dropout'], bn=BN, act=act))

    def forward(self, feat):
        h = feat
        for layer in self.mlp:
            h = layer(h)

        return h

    def reset_param(self):
        for layer in self.mlp:
            layer.reset_parameters()