import argparse
import torch
import sys
import os
sys.path.append('.')
from data_prepare import data_preparation
import models_small
import models_ogb
import trainers


def update_info_dict(args, info_dict):
    info_dict.update(args.__dict__)
    info_dict.update({'device': torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu)),})
    return info_dict


def get_db_dir(db):
    if db in ['cora', 'citeseer', 'pubmed', 'amz-computer', 'amz-photo', 'co-cs', 'co-phy', 'ogbn-arxiv']:
        return './dataset/{}'.format(db)
    else:
        raise ValueError('Unknown dataset name')


def main(args):
    g, info_dict = data_preparation.data_prep(args.dataset, get_db_dir(args.dataset))
    info_dict = update_info_dict(args, info_dict)
    if args.seed >= 0:
        g = data_preparation.set_seed_and_split(g, args.seed, args.split)

    if args.model == 'SAGE':
        model = models_ogb.SAGE(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.SAGE(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'GCN':
        model = models_ogb.GCN(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.GCN(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'GAT':
        model = models_ogb.GAT(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.GAT(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'JKNet':
        model = models_ogb.JKNet(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.JKNet(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'SGC':
        model = models_ogb.SGC(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.SGC(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'MLP':
        model = models_ogb.MLP(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.MLP(info_dict)
        trainer = trainers.BaseMLPTrainer(g, model, info_dict)
    elif args.model == 'GCNCoCoS':
        info_dict.update({'backbone': 'GCN'})
        model = models_ogb.GCN(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.GCN(info_dict)
        Dis = models_ogb.DisMLP(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.DisMLP(info_dict)
        trainer = trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis) if args.dataset == 'ogbn-arxiv' \
            else trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'GATCoCoS':
        info_dict.update({'backbone': 'GAT'})
        model = models_ogb.GAT(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.GAT(info_dict)
        Dis = models_ogb.DisMLP(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.DisMLP(info_dict)
        trainer = trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis) if args.dataset == 'ogbn-arxiv' \
            else trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'SAGECoCoS':
        info_dict.update({'backbone': 'SAGE'})
        model = models_ogb.SAGE(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.SAGE(info_dict)
        Dis = models_ogb.DisMLP(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.DisMLP(info_dict)
        trainer = trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis) if args.dataset == 'ogbn-arxiv' \
            else trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'JKNetCoCoS':
        info_dict.update({'backbone': 'JKNet'})
        model = models_ogb.JKNet(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.JKNet(info_dict)
        Dis = models_ogb.DisMLP(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.DisMLP(info_dict)
        trainer = trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis) if args.dataset == 'ogbn-arxiv' \
            else trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'SGCCoCoS':
        info_dict.update({'backbone': 'SGC'})
        model = models_ogb.SGC(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.SGC(info_dict)
        Dis = models_ogb.DisMLP(info_dict) if args.dataset == 'ogbn-arxiv' else models_small.DisMLP(info_dict)
        trainer = trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis) if args.dataset == 'ogbn-arxiv' \
            else trainers.CoCoSTrainer(g, model, info_dict, Dis=Dis)
    else:
        raise ValueError("unknown model: {}".format(args.model))

    model.to(info_dict['device'])
    if 'Dis' in locals().keys():
        Dis.to(info_dict['device'])
    print(model)
    print(info_dict)
    print('\nSTART TRAINING\n')
    val_acc, tt_acc, val_acc_fin, tt_acc_fin, microf1, macrof1 = trainer.train()

    # save experimental results in the local storage
    suffix = 'ori' if args.split == 'None' else '_'.join(args.split.split('-'))
    save_root = os.path.join('exp', '{}_{}'.format(args.model, suffix))
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    record_dir = os.path.join(save_root, '{}_{}_result.csv'.format(args.model, args.dataset))
    if not os.path.exists(record_dir):
        NEED_TITLE = True
    else:
        NEED_TITLE = False
    with open(os.path.join(save_root, '{}_{}_result.csv'.format(args.model, args.dataset)), 'a') as f:
        if 'ogb' in args.dataset:
            info_dict.pop('evaluator')
            info_dict.pop('src_root')
        if NEED_TITLE:
            title = [str(k) for k in info_dict.keys()]
            metrics_list = ['tt_acc', 'val_acc', 'tt_acc_fin', 'val_acc_fin', 'Micro-F1', 'Macro-F1']
            title = metrics_list + title
            f.write(','.join(title) + '\n')
        result = [str(s) for s in list(info_dict.values())]
        metric_result = [str(tt_acc), str(val_acc), str(tt_acc_fin), str(val_acc_fin), str(microf1), str(macrof1)]
        result = metric_result + result
        f.write(','.join(result) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='the main program to run experiments on small datasets')
    parser.add_argument("--model", type=str, default='GCN',
                        help="model name")
    parser.add_argument("--dataset", type=str, default='cora',
                        help="the dataset for the experiment")
    parser.add_argument("--pretr_state", type=str, default='val',
                        help="the version of the pretraining model")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="the number of training epochs")
    parser.add_argument("--eta", type=int, default=10,
                        help="the interval (epoch) to override/ update the estimated labels")
    parser.add_argument("--n_cls_pershuf", type=int, default=4,
                        help="the number of classes for each shuffling operation, only works for Ogbn-arxiv")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="the number of hidden layers")
    parser.add_argument("--cls_layers", type=int, default=1,
                        help="the number of MLP classifier layers, if any")
    parser.add_argument("--dis_layers", type=int, default=2,
                        help="the number of MLP discriminator layers, only for CoCoS-enhanced models")
    parser.add_argument("--hid_dim", type=int, default=16,
                        help="the hidden dimension of hidden layers in the backbone model")
    parser.add_argument("--emb_hid_dim", type=int, default=64,
                        help="the hidden dimension of the hidden layers in the MLP discriminator")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="dropout rate")
    parser.add_argument("--input_drop", type=float, default=0.25,
                        help="dropout for input features, only for models for Ogbn-arxiv")
    parser.add_argument("--attn_drop", type=float, default=0,
                        help="attention dropout rate, only for GAT")
    parser.add_argument("--edge_drop", type=float, default=0,
                        help="edge dropout rate, only for GAT for Ogbn-arxiv")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="the number of attention heads, only for GAT")
    parser.add_argument("--agg_type", type=str, default="gcn",
                        help="the aggregation type of GraphSAGE")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="specify the gpu index, set -1 to train on cpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="the learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="the weight decay for optimizer")
    parser.add_argument("--seed", type=int, default=0,
                        help="the random seed for reproduce the result")
    parser.add_argument("--split", type=str, default='None',
                        help="the samples split settings, should be in the format of "
                             "{training set ratio}-{validation set ratio};"
                             "set None to use the standard train-val-test split of the dataset")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="coefficient for the contrastive loss")
    parser.add_argument("--cls_mode", type=str, default='both',
                        help="the type of the classification loss")
    parser.add_argument("--ctr_mode", type=str, default='FS',
                        help="the type of positive pairs for contrastive loss")
    parser.add_argument("--bn", action='store_true', default=False,
                        help="a flag to indicate whether use batchnorm for training")
    args = parser.parse_args()

    main(args)