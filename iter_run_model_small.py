import os
import argparse
import torch
import numpy as np
import dgl


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def main(args):
    # hyper-parameters
    bn = args.bn
    gpu = args.gpu
    lr_group = args.lr.strip().split('-')
    wd = args.weight_decay
    split = args.split
    cls_layers = [args.cls_layers] if args.cls_layers > 0 else range(1, abs(args.cls_layers) + 1)
    dis_layers = args.dis_layers
    n_epochs = args.n_epochs
    pretr_states = ['val', 'fin'] if args.pretr_state == 'two' else [args.pretr_state]
    n_step_epochs = args.n_step_epochs.strip().split('-')
    n_step_epochs = [int(s) for s in n_step_epochs]
    n_cls_pershuf = args.n_cls_pershuf
    hid_dims = args.hid_dim.split('-')
    emb_hid_dims = args.emb_hid_dim.split('-')
    n_layers = args.n_layers
    agg_types = ['gcn', 'mean'] if args.agg_type == 'two' else [args.agg_type]
    if args.model == 'backbone':
        mns = ['GCN', 'GAT', 'SAGE', 'SAGEMLP']
    elif args.model == 'CSS':
        mns = ['GCNTSS', 'GATTSS', 'SAGETSS', 'SAGEMLPTSS']
    else:
        mns = [args.model]
    num_round = args.round
    if args.dataset == 'citation':
        dbs = ['cora', 'citeseer', 'pubmed']
    elif args.dataset == 'coauthor':
        dbs = ['co-cs', 'co-phy']
    elif args.dataset == 'amazon':
        dbs = ['amz-computer', 'amz-photo']
    else:
        dbs = [args.dataset]
    nh = args.num_heads
    nbs = args.num_neighbors
    bs = args.batch_size
    minibatch = args.minibatch
    nw = args.num_cpu
    dropout = args.dropout
    es = args.early_stop
    patience = args.patience
    conf_th = args.conf_th
    wcls = args.wcls
    wctr_group = args.wctr.strip().split('-')

    ctr_mode = args.ctr_mode
    if ctr_mode == 'all':
        ctr_mode_group = ['F', 'M', 'S', 'FM', 'FS', 'MS', 'FMS']
    else:
        ctr_mode_group = ctr_mode.strip().split('-')

    cls_mode_group = ['shuf', 'raw', 'both'] if args.cls_mode == 'all' else args.cls_mode.split('-')

    for db in dbs:
        for at in agg_types:
            for nsep in n_step_epochs:
                for cl in cls_layers:
                    for mn in mns:
                        for ptrst in pretr_states:
                            for hid_dim in hid_dims:
                                for clsm in cls_mode_group:
                                    for emb_hid_dim in emb_hid_dims:
                                        for ctrm in ctr_mode_group:
                                            for wctr in wctr_group:
                                                for lr in lr_group:
                                                    for r in range(num_round):
                                                        # set_seed(r)
                                                        run_cmd = "python -u main_small.py --model {m} --dataset {db} --n_epochs {ep} " \
                                                                  "--pretr_state {ptrst} --n_step_epochs {nsep} --n_cls_pershuf {ncps} " \
                                                                  "--hid_dim {hid_dim} --n_layers {nl} " \
                                                                  "--cls_layers {cl} --dis_layers {dl} " \
                                                                  "--lr {lr}  --weight_decay {wd} --agg_type {at} --gpu {gpu} " \
                                                                  "--num_heads {nh} --num_neighbors {nbs} " \
                                                                  "--batch_size {bs} --num_cpu {nw} --dropout {dp} " \
                                                                  "--emb_hid_dim {ehd} " \
                                                                  "--seed {seed} --split {spt} --conf_th {cth} " \
                                                                  "--wcls {wcls} --wctr {wctr} " \
                                                                  "--cls_mode {clsm} " \
                                                                  "--ctr_mode {ctrm} " \
                                                                  "--patience {patience} " \
                                                                  "{reset} {ktf} {bn} {minibatch} {es} ".format(
                                                            m=mn, db=db, ep=n_epochs, ptrst=ptrst, nsep=nsep, ncps=n_cls_pershuf,
                                                            hid_dim=hid_dim, nl=n_layers,
                                                            cl=cl, dl=dis_layers,
                                                            lr=lr, wd=wd, at=at, gpu=gpu, nh=nh, nbs=nbs, bs=bs,
                                                            nw=nw, dp=dropout, ehd=emb_hid_dim,
                                                            seed=r, spt=split, cth=conf_th,
                                                            wcls=wcls, wctr=wctr,
                                                            clsm=clsm,
                                                            ctrm=ctrm,
                                                            patience=patience,
                                                            reset='--reset' if args.reset else '',
                                                            ktf='--keep_tr_feat' if args.keep_tr_feat else '',
                                                            bn='--bn' if bn else '',
                                                            minibatch='--minibatch' if minibatch else '',
                                                            es='--early_stop' if es else '',)
                                                        os.system(run_cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='iteratively run models')
    parser.add_argument("--model", type=str, default='GCN',
                        help="model name")
    parser.add_argument("--round", type=int, default=10, help='number of rounds to repeat')
    parser.add_argument("--dataset", type=str, default='cora',
                        help="dataset for experiment")
    parser.add_argument("--pretr_state", type=str, default='val',
                        help="number of training epochs")
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n_step_epochs", type=str, default='20',
                        help="epoch gap to update the estimated label, only works for progressively update models")
    parser.add_argument("--n_cls_pershuf", type=int, default=4,
                        help="number of classes for each shuffling, only for ProgClsShuf and ProgClsContra trainer")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--cls_layers", type=int, default=1,
                        help="number of classifier layers")
    parser.add_argument("--dis_layers", type=int, default=2,
                        help="number of mlp discriminator layers")
    parser.add_argument("--hid_dim", type=str, default='16',
                        help="dimension of backbone model hidden layer")
    parser.add_argument("--emb_hid_dim", type=str, default='16',
                        help="dimension of discriminator classifier hidden layer")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="dropout probability")
    parser.add_argument("--att_drop", type=float, default=0.6,
                        help="dropout probability")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="number of attention heads")
    parser.add_argument("--agg_type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=str, default='0.01',
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--split", type=str, default='none',
                        help="samples split setting: {training sample ratio}-{validation sample ratio}, only works "
                             "when seed is non-negative")
    parser.add_argument("--reset", action='store_true', default=False,
                        help="reset the model parameters after pretrain")
    parser.add_argument("--keep_tr_feat", action='store_true', default=False,
                        help="keep the training node features in the shuffling process")
    parser.add_argument("--wcls", type=float, default=1,
                        help="coefficient for the classification loss (only for contrastive training)")
    parser.add_argument("--wctr", type=str, default='1',
                        help="coefficient for the contrastive loss (only for contrastive training)")
    parser.add_argument("--cls_mode", type=str, default='raw',
                        help="a parameter to determine how the classification loss is computed")
    parser.add_argument("--ctr_mode", type=str, default='F',
                        help="contrastive mode for positive samples")
    parser.add_argument("--conf_th", type=float, default=-1,
                        help="confidence threshold for shuffling nodes")
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--patience', type=int, default=100,
                        help="patience for early stop")
    parser.add_argument("--bn", action='store_true', default=False,
                        help="flag to indicate whether use batchnorm in training")
    parser.add_argument("--data_dir", type=str, default='./data_prepare',
                        help="the root directory that stores pre-computed features")
    parser.add_argument("--minibatch", action='store_true', default=False,
                        help="flag to indicate whether use minibatch training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size (only works for minibatch training)")
    parser.add_argument("--num_neighbors", type=str, default='10-15',
                        help="number of neighbors to be sampled in each layer (only works for minibatch training)")
    parser.add_argument("--num_cpu", type=int, default=2,
                        help="number of cpu workers (only works for minibatch training)")
    args = parser.parse_args()

    main(args)