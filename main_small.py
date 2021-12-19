import argparse
import torch
import sys
import os
sys.path.append('.')
from data_prepare import data_preparation
import models_small as models
import trainers


def update_info_dict(args, info_dict):
    info_dict.update(args.__dict__)
    info_dict.update({'device': torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu)),})

    if args.model == 'node2vec':
        info_dict['in_dim'] = 128
    else:
        info_dict['in_dim'] = info_dict['in_dim']
    return info_dict


def get_db_dir(db):
    if db in ['cora', 'citeseer', 'pubmed']:
        return None
    elif db in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'ogbn-mag',
                'amz-computer', 'amz-photo',
                'co-cs', 'co-phy']:
        return '../dataset'
    else:
        raise ValueError('Unknown dataset name')


def main(args):
    g, info_dict = data_preparation.data_prep(args.dataset, get_db_dir(args.dataset))
    info_dict = update_info_dict(args, info_dict)
    if args.seed >= 0:
        g = data_preparation.set_seed_and_split(g, args.seed, args.split)

    if args.model == 'SAGEMLP':
        model = models.SAGEMLP(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'SAGE':
        model = models.SAGE(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'GCN':
        model = models.GCN(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'GCNMLP':
        model = models.GCNMLP(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'GAT':
        model = models.GAT(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'JKNet':
        model = models.JKNet(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'SGC':
        model = models.SGC(info_dict)
        trainer = trainers.BaseTrainer(g, model, info_dict)
    elif args.model == 'MLP':
        model = models.MLP(info_dict)
        trainer = trainers.BaseMLPTrainer(g, model, info_dict)
    elif args.model == 'node2vec':
        model = models.MLP(info_dict)
        trainer = trainers.BaseMLPTrainer(g, model, info_dict)
    elif args.model == 'GCNPS':
        info_dict.update({'backbone': 'GCN'})
        model = models.GCN(info_dict)
        trainer = trainers.ProgShufTrainer(g, model, info_dict)
    elif args.model == 'SAGEPS':
        info_dict.update({'backbone': 'SAGE'})
        model = models.SAGE(info_dict)
        trainer = trainers.ProgShufTrainer(g, model, info_dict)
    elif args.model == 'SAGEMLPPS':
        info_dict.update({'backbone': 'SAGEMLP'})
        model = models.SAGEMLP(info_dict)
        trainer = trainers.ProgShufTrainer(g, model, info_dict)
    elif args.model == 'GATPS':
        info_dict.update({'backbone': 'GAT'})
        model = models.GAT(info_dict)
        trainer = trainers.ProgShufTrainer(g, model, info_dict)
    elif args.model == 'JKNetPS':
        info_dict.update({'backbone': 'JKNet'})
        model = models.JKNet(info_dict)
        trainer = trainers.ProgShufTrainer(g, model, info_dict)
    elif args.model == 'SGCPS':
        info_dict.update({'backbone': 'SGC'})
        model = models.SGC(info_dict)
        trainer = trainers.ProgShufTrainer(g, model, info_dict)
    elif args.model == 'GCNCTRS':
        info_dict.update({'backbone': 'GCN'})
        model = models.GCN(info_dict)
        Dis = models.DisMLP(info_dict)
        # trainer = trainers.ProgContraTrainer(g, model, info_dict, Dis=Dis)
        trainer = trainers.ProgPosContraTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'GATCTRS':
        info_dict.update({'backbone': 'GAT'})
        model = models.GAT(info_dict)
        Dis = models.DisMLP(info_dict)
        # trainer = trainers.ProgContraTrainer(g, model, info_dict, Dis=Dis)
        trainer = trainers.ProgPosContraTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'SAGECTRS':
        info_dict.update({'backbone': 'SAGE'})
        model = models.SAGE(info_dict)
        Dis = models.DisMLP(info_dict)
        # trainer = trainers.ProgContraTrainer(g, model, info_dict, Dis=Dis)
        trainer = trainers.ProgPosContraTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'SAGEMLPCTRS':
        info_dict.update({'backbone': 'SAGEMLP'})
        model = models.SAGEMLP(info_dict)
        Dis = models.DisMLP(info_dict)
        # trainer = trainers.ProgContraTrainer(g, model, info_dict, Dis=Dis)
        trainer = trainers.ProgPosContraTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'JKNetCTRS':
        info_dict.update({'backbone': 'JKNet'})
        model = models.JKNet(info_dict)
        Dis = models.DisMLP(info_dict)
        # trainer = trainers.ProgContraTrainer(g, model, info_dict, Dis=Dis)
        trainer = trainers.ProgPosContraTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'SGCCTRS':
        info_dict.update({'backbone': 'SGC'})
        model = models.SGC(info_dict)
        Dis = models.DisMLP(info_dict)
        # trainer = trainers.ProgContraTrainer(g, model, info_dict, Dis=Dis)
        trainer = trainers.ProgPosContraTrainer(g, model, info_dict, Dis=Dis)
    elif args.model == 'GCNGTFeat':
        model = models.GCN(info_dict)
        trainer = trainers.BaseTrainerGTFeat(g, model, info_dict)
    elif args.model == 'SAGEGTFeat':
        model = models.SAGE(info_dict)
        trainer = trainers.BaseTrainerGTFeat(g, model, info_dict)
    elif args.model == 'SAGEMLPGTFeat':
        model = models.SAGEMLP(info_dict)
        trainer = trainers.BaseTrainerGTFeat(g, model, info_dict)
    elif args.model == 'GATGTFeat':
        model = models.GAT(info_dict)
        trainer = trainers.BaseTrainerGTFeat(g, model, info_dict)
    elif args.model == 'JKNetGTFeat':
        model = models.JKNet(info_dict)
        trainer = trainers.BaseTrainerGTFeat(g, model, info_dict)
    elif args.model == 'SGCGTFeat':
        model = models.SGC(info_dict)
        trainer = trainers.BaseTrainerGTFeat(g, model, info_dict)
    else:
        raise ValueError("unknown model: {}".format(args.model))

    model.to(info_dict['device'])
    if 'Dis' in locals().keys():
        Dis.to(info_dict['device'])
    print(model)
    print(info_dict)
    print('\nSTART TRAINING\n')
    val_acc, tt_acc, val_acc_fin, tt_acc_fin, microf1, macrof1, epoch_tc = trainer.train()

    suffix = 'ori' if args.split == 'none' else '_'.join(args.split.split('-'))
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
            title.insert(0, 'per_epoch_time_cost')
            title.insert(0, 'Macro-F1')
            title.insert(0, 'Micro-F1')
            title.insert(0, 'val_acc_fin')
            title.insert(0, 'tt_acc_fin')
            title.insert(0, 'val_acc')
            title.insert(0, 'tt_acc')
            f.write(','.join(title) + '\n')
        result = [str(s) for s in list(info_dict.values())]
        result.insert(0, str(epoch_tc))
        result.insert(0, str(macrof1))
        result.insert(0, str(microf1))
        result.insert(0, str(val_acc_fin))
        result.insert(0, str(tt_acc_fin))
        result.insert(0, str(val_acc))
        result.insert(0, str(tt_acc))
        f.write(','.join(result) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification with triple branch architecture')
    parser.add_argument("--model", type=str, default='GCN',
                        help="model name")
    parser.add_argument("--dataset", type=str, default='pubmed',
                        help="dataset for experiment")
    parser.add_argument("--pretr_state", type=str, default='val',
                        help="number of training epochs")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n_step_epochs", type=int, default=10,
                        help="epoch gap to update the estimated label, only works for progressively update models")
    parser.add_argument("--n_cls_pershuf", type=int, default=4,
                        help="number of classes for each shuffling, only for ProgClsShuf and ProgClsContra trainer")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--cls_layers", type=int, default=1,
                        help="number of classifier layers")
    parser.add_argument("--dis_layers", type=int, default=2,
                        help="number of mlp discriminator layers")
    parser.add_argument("--hid_dim", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--emb_hid_dim", type=int, default=16,
                        help="number of hidden gcn units")
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
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for re-assign samples")
    parser.add_argument("--split", type=str, default='None',
                        help="samples split setting: {training sample ratio}-{validation sample ratio}, only works "
                             "when seed is non-negative")
    parser.add_argument("--reset", action='store_true', default=False,
                        help="reset the model parameters after pretrain")
    parser.add_argument("--keep_tr_feat", action='store_true', default=False,
                        help="keep the training node features in the shuffling process")
    parser.add_argument("--wcls", type=float, default=1,
                        help="coefficient for the classification loss (only for contrastive training)")
    parser.add_argument("--wctr", type=float, default=1,
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