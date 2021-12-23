import torch
import numpy as np
from torch import nn
import os
import time
import utils
from copy import deepcopy
from sklearn import metrics


class BaseMLPTrainer(object):
    '''
    Base trainer for training mlp
    '''

    def __init__(self, g, model, info_dict, *args, **kwargs):
        self.g = g

        self.model = model
        self.info_dict = info_dict
        self.feat = g.ndata['feat'].to(info_dict['device'])

        # load train/val/test split
        self.tr_nid = g.ndata['train_mask'].nonzero().squeeze()
        self.val_nid = g.ndata['val_mask'].nonzero().squeeze()
        self.tt_nid = g.ndata['test_mask'].nonzero().squeeze()
        self.labels = g.ndata['label']
        self.tr_y = self.labels[self.tr_nid]
        self.val_y = self.labels[self.val_nid]
        self.tt_y = self.labels[self.tt_nid]

        self.crs_entropy_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

        self.best_val_acc = 0
        self.best_tt_acc = 0
        self.best_microf1 = 0
        self.best_macrof1 = 0

    def train(self):
        self.g = self.g.int().to(self.info_dict['device'])
        for i in range(self.info_dict['n_epochs']):
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                _ = utils.save_model(self.model, self.info_dict)

            print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                  .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        # save the model in the final epoch
        _ = utils.save_model(self.model, self.info_dict, state='fin')
        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        # training samples and labels
        nids = self.tr_nid
        labels = self.tr_y

        tic = time.time()
        self.model.train()
        labels = labels.to(self.info_dict['device'])
        feat = self.feat
        with torch.set_grad_enabled(True):
            logits = self.model(feat)
            epoch_loss = self.crs_entropy_fn(logits[nids], labels)

            self.opt.zero_grad()
            epoch_loss.backward()
            self.opt.step()

            _, preds = torch.max(logits[nids], dim=1)
            if 'ogb' in self.info_dict['dataset']:
                epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": labels.unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
            else:
                epoch_acc = torch.sum(preds == labels).cpu().item() * 1.0 / labels.shape[0]
            epoch_micro_f1 = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
            epoch_macro_f1 = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.cpu().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.cpu().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def eval_epoch(self, epoch_i):
        tic = time.time()
        self.model.eval()
        val_labels = self.val_y.to(self.info_dict['device'])
        tt_labels = self.tt_y.to(self.info_dict['device'])
        feat = self.feat
        with torch.set_grad_enabled(False):
            logits = self.model.forward(feat)
            val_epoch_loss = self.crs_entropy_fn(logits[self.val_nid], val_labels)
            tt_epoch_loss = self.crs_entropy_fn(logits[self.tt_nid], tt_labels)

            _, val_preds = torch.max(logits[self.val_nid], dim=1)
            _, tt_preds = torch.max(logits[self.tt_nid], dim=1)


            if 'ogb' in self.info_dict['dataset']:
                val_epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": val_labels.unsqueeze(-1), "y_pred": val_preds.unsqueeze(-1)})['acc']
                tt_epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": tt_labels.unsqueeze(-1), "y_pred": tt_preds.unsqueeze(-1)})['acc']
            else:
                val_epoch_acc = torch.sum(val_preds == val_labels).cpu().item() * 1.0 / val_labels.shape[0]
                tt_epoch_acc = torch.sum(tt_preds == tt_labels).cpu().item() * 1.0 / tt_labels.shape[0]
            val_epoch_micro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="micro")
            val_epoch_macro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="macro")
            tt_epoch_micro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="micro")
            tt_epoch_macro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | validation accuracy: {:.4f}".format(epoch_i, val_epoch_loss.cpu().item(),
                                                                             val_epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(val_epoch_micro_f1, val_epoch_macro_f1))
        print("Epoch {} | Loss: {:.4f} | testing accuracy: {:.4f}".format(epoch_i, tt_epoch_loss.cpu().item(),
                                                                             tt_epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(tt_epoch_micro_f1, tt_epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return (val_epoch_loss.cpu().item(), val_epoch_acc, val_epoch_micro_f1, val_epoch_macro_f1), \
               (tt_epoch_loss.cpu().item(), tt_epoch_acc, tt_epoch_micro_f1, tt_epoch_macro_f1)


class BaseTrainer(object):
    '''
    Base trainer for training.
    For baseline GNN models, i.e., GCN, GraphSAGE, GAT, JKNet, SGC.
    '''

    def __init__(self, g, model, info_dict, *args, **kwargs):
        self.g = g

        self.model = model
        self.info_dict = info_dict

        # load train/val/test split
        self.tr_nid = g.ndata['train_mask'].nonzero().squeeze()
        self.val_nid = g.ndata['val_mask'].nonzero().squeeze()
        self.tt_nid = g.ndata['test_mask'].nonzero().squeeze()
        self.labels = g.ndata['label']
        self.tr_y = self.labels[self.tr_nid]
        self.val_y = self.labels[self.val_nid]
        self.tt_y = self.labels[self.tt_nid]

        self.crs_entropy_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

        self.best_val_acc = 0
        self.best_tt_acc = 0
        self.best_microf1 = 0
        self.best_macrof1 = 0

    def train(self):
        self.g = self.g.int().to(self.info_dict['device'])
        for i in range(self.info_dict['n_epochs']):
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                _ = utils.save_model(self.model, self.info_dict)

            print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                  .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        # save the model in the final epoch
        _ = utils.save_model(self.model, self.info_dict, state='fin')
        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        # training sample indices and labels
        nids = self.tr_nid
        labels = self.tr_y

        tic = time.time()
        self.model.train()
        labels = labels.to(self.info_dict['device'])
        with torch.set_grad_enabled(True):
            logits = self.model(self.g, self.g.ndata['feat'])
            epoch_loss = self.crs_entropy_fn(logits[nids], labels)

            self.opt.zero_grad()
            epoch_loss.backward()
            self.opt.step()

            _, preds = torch.max(logits[nids], dim=1)
            if 'ogb' in self.info_dict['dataset']:
                epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": labels.unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
            else:
                epoch_acc = torch.sum(preds == labels).cpu().item() * 1.0 / labels.shape[0]
            epoch_micro_f1 = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
            epoch_macro_f1 = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.cpu().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.cpu().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def eval_epoch(self, epoch_i):
        tic = time.time()
        self.model.eval()
        val_labels = self.val_y.to(self.info_dict['device'])
        tt_labels = self.tt_y.to(self.info_dict['device'])
        with torch.set_grad_enabled(False):
            logits = self.model.forward(self.g, self.g.ndata['feat'])
            val_epoch_loss = self.crs_entropy_fn(logits[self.val_nid], val_labels)
            tt_epoch_loss = self.crs_entropy_fn(logits[self.tt_nid], tt_labels)

            _, val_preds = torch.max(logits[self.val_nid], dim=1)
            _, tt_preds = torch.max(logits[self.tt_nid], dim=1)


            if 'ogb' in self.info_dict['dataset']:
                val_epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": val_labels.unsqueeze(-1), "y_pred": val_preds.unsqueeze(-1)})['acc']
                tt_epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": tt_labels.unsqueeze(-1), "y_pred": tt_preds.unsqueeze(-1)})['acc']
            else:
                val_epoch_acc = torch.sum(val_preds == val_labels).cpu().item() * 1.0 / val_labels.shape[0]
                tt_epoch_acc = torch.sum(tt_preds == tt_labels).cpu().item() * 1.0 / tt_labels.shape[0]
            val_epoch_micro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="micro")
            val_epoch_macro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="macro")
            tt_epoch_micro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="micro")
            tt_epoch_macro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | validation accuracy: {:.4f}".format(epoch_i, val_epoch_loss.cpu().item(),
                                                                             val_epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(val_epoch_micro_f1, val_epoch_macro_f1))
        print("Epoch {} | Loss: {:.4f} | testing accuracy: {:.4f}".format(epoch_i, tt_epoch_loss.cpu().item(),
                                                                             tt_epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(tt_epoch_micro_f1, tt_epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return (val_epoch_loss.cpu().item(), val_epoch_acc, val_epoch_micro_f1, val_epoch_macro_f1), \
               (tt_epoch_loss.cpu().item(), tt_epoch_acc, tt_epoch_micro_f1, tt_epoch_macro_f1)


class CoCoSTrainer(BaseTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)
        self.pred_labels = None
        self.pred_conf = None

        self.best_pretr_val_acc = None
        suffix = 'ori' if info_dict['split'] == 'None' else '_'.join(info_dict['split'].split('-'))
        self.pretr_model_dir = os.path.join('exp', info_dict['backbone'] + '_' + suffix, info_dict['dataset'],
                                            '{model}_{db}_{seed}{agg}_{state}.pt'.
                                            format(model=info_dict['backbone'],
                                                   db=info_dict['dataset'],
                                                   seed=info_dict['seed'],
                                                   agg='_' + info_dict['agg_type'] if
                                                   'SAGE' in self.info_dict['model'] else '',
                                                   state=self.info_dict['pretr_state']
                                                   )
                                            )
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.Dis = kwargs['Dis']
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.Dis.parameters()}],
                                    lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

    def train(self):
        self.g = self.g.int().to(self.info_dict['device'])

        self.get_pred_labels()
        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['eta'] == 0:
                # progressively update/ override the predicted labels
                self.get_pred_labels()

            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                save_model_dir = utils.save_model(self.model, self.info_dict)
                if val_acc_epoch > self.best_pretr_val_acc:
                    # update the pretraining model's parameter directory, we will use the updated pretraining model to
                    # generate estimated labels in the following epochs
                    self.pretr_model_dir = save_model_dir

            print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                  .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        # training sample indices and labels, for the supervised loss
        cls_nids = self.tr_nid
        cls_labels = self.tr_y
        cls_labels = cls_labels.to(self.info_dict['device'])
        # node indices for contrastive learning, for the contrastive loss
        ctr_nids = torch.cat((self.val_nid, self.tt_nid))
        # positive and negative labels for contrastive learning
        ctr_labels_pos = torch.ones_like(ctr_nids).to(self.info_dict['device']).unsqueeze(dim=-1).float()
        ctr_labels_neg = torch.zeros_like(ctr_nids).to(self.info_dict['device']).unsqueeze(dim=-1).float()

        tic = time.time()
        self.model.train()
        with torch.set_grad_enabled(True):
            feat = self.g.ndata['feat']
            shuf_feat = self.shuffle_feat(feat)
            ori_logits = self.model(self.g, feat)
            shuf_logits = self.model(self.g, shuf_feat)

            # generate positive samples
            pos_nids = self.shuffle_nids()
            tp_ori_logits = ori_logits[pos_nids]
            tp_shuf_logits = shuf_logits[pos_nids]
            # generate negative samples
            neg_nids = self.gen_neg_nids()
            neg_ori_logits = ori_logits[neg_nids].detach()

            epoch_ctr_loss_pos = torch.Tensor([]).to(self.info_dict['device'])
            if 'F' in self.info_dict['ctr_mode']:
                pos_score = self.Dis(torch.cat((shuf_logits, ori_logits), dim=-1))
                pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                epoch_ctr_loss_pos = torch.cat((epoch_ctr_loss_pos, pos_loss.unsqueeze(dim=0)), dim=0)
            if 'T' in self.info_dict['ctr_mode']:
                pos_score = self.Dis(torch.cat((tp_ori_logits, ori_logits), dim=-1))
                pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                epoch_ctr_loss_pos = torch.cat((epoch_ctr_loss_pos, pos_loss.unsqueeze(dim=0)), dim=0)
            if 'M' in self.info_dict['ctr_mode']:
                pos_score = self.Dis(torch.cat((tp_shuf_logits, ori_logits), dim=-1))
                pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                epoch_ctr_loss_pos = torch.cat((epoch_ctr_loss_pos, pos_loss.unsqueeze(dim=0)), dim=0)
            if 'S' in self.info_dict['ctr_mode']:
                pos_score = self.Dis(torch.cat((tp_shuf_logits, shuf_logits), dim=-1))
                pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                epoch_ctr_loss_pos = torch.cat((epoch_ctr_loss_pos, pos_loss.unsqueeze(dim=0)), dim=0)
            epoch_ctr_loss_pos = epoch_ctr_loss_pos.mean()

            neg_score = self.Dis(torch.cat((ori_logits, neg_ori_logits), dim=-1))
            epoch_ctr_loss_neg = self.bce_fn(neg_score[ctr_nids], ctr_labels_neg)

            if self.info_dict['cls_mode'] == 'shuf':
                epoch_cls_loss = self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels)
            elif self.info_dict['cls_mode'] == 'raw':
                epoch_cls_loss = self.crs_entropy_fn(ori_logits[cls_nids], cls_labels)
            elif self.info_dict['cls_mode'] == 'both':
                epoch_cls_loss = 0.5 * (self.crs_entropy_fn(ori_logits[cls_nids], cls_labels) +
                                        self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels))
            else:
                raise ValueError("Unexpected cls_mode parameter: {}".format(self.info_dict['cls_mode']))

            epoch_ctr_loss = epoch_ctr_loss_pos + epoch_ctr_loss_neg

            epoch_loss = epoch_cls_loss + self.info_dict['alpha'] * epoch_ctr_loss

            self.opt.zero_grad()
            epoch_loss.backward()
            self.opt.step()

            _, preds = torch.max(shuf_logits[cls_nids], dim=1)
            if 'ogb' in self.info_dict['dataset']:
                epoch_acc = self.info_dict['evaluator'].eval(
                    {"y_true": cls_labels.unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
            else:
                epoch_acc = torch.sum(preds == cls_labels).cpu().item() * 1.0 / cls_labels.shape[0]
            epoch_micro_f1 = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
            epoch_macro_f1 = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.cpu().item(), epoch_acc))
        print("cls loss: {:.4f} | ctr pos loss: {:.4f} | ctr neg loss: {:.4f}".format(epoch_cls_loss.cpu().item(),
                                                                                      epoch_ctr_loss_pos.cpu().item(),
                                                                                      epoch_ctr_loss_neg.cpu().item(),
                                                                                      ))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.cpu().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def get_pred_labels(self):

        # load the pretrained model and use it to estimate the labels
        cur_model_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.model.eval()
        with torch.set_grad_enabled(False):
            feat = self.g.ndata['feat']
            logits = self.model(self.g, feat)

            _, preds = torch.max(logits, dim=1)
            conf = torch.softmax(logits, dim=1).max(dim=1)[0]
            self.pred_labels = preds
            # for training nodes, the estimated labels will be replaced by their ground-truth labels
            self.pred_labels[self.tr_nid] = self.labels[self.tr_nid].to(self.info_dict['device'])
            self.pred_conf = conf

            pretr_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / self.labels[self.val_nid].shape[0]
            pretr_tt_acc = torch.sum(preds[self.tt_nid].cpu() == self.labels[self.tt_nid]).item() * 1.0 / self.labels[self.tt_nid].shape[0]
            self.best_pretr_val_acc = pretr_val_acc

        # reload the current model's parameters
        self.model.load_state_dict(cur_model_state_dict)

    def shuffle_feat(self, nfeat):
        pos_feat = nfeat.clone().detach()

        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes have not been estimated!')

        # generate positive features
        shuf_nid = torch.zeros_like(nid).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement for the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])
        pos_feat[nid] = nfeat[shuf_nid].detach()

        return pos_feat

    def shuffle_nids(self):
        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes have not been estimated!')

        # randomly sample a positive counterpart for each node
        shuf_nid = torch.arange(self.g.num_nodes()).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])

        return shuf_nid

    def gen_neg_nids(self):
        num_nodes = self.g.num_nodes()
        nid = torch.arange(num_nodes)
        labels = self.pred_labels

        # randomly sample an instance as the negative sample, which is from a (estimated) different class
        shuf_nid = torch.randperm(num_nodes).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # set the sampling prob to be 0 so that the node from the same class will not be sampled
            sample_prob[i_pos] = 0
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        return shuf_nid


class CoCoSTrainerOGB(BaseTrainer):
    '''
    The trainer for model training on Ogbn-arxiv dataset.
    '''
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)
        self.pred_labels = None
        self.pred_conf = None
        self.best_pretr_val_acc = None

        # load the pretrained model
        suffix = 'ori' if info_dict['split'] == 'None' else '_'.join(info_dict['split'].split('-'))
        self.pretr_model_dir = os.path.join('exp', info_dict['backbone'] + '_' + suffix, info_dict['dataset'],
                                            '{model}_{db}_{seed}{agg}_{state}.pt'.
                                            format(model=info_dict['backbone'],
                                                   db=info_dict['dataset'],
                                                   seed=info_dict['seed'],
                                                   agg='_' + info_dict['agg_type'] if
                                                   'SAGE' in self.info_dict['model'] else '',
                                                   state=self.info_dict['pretr_state']
                                                   )
                                            )
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.Dis = kwargs['Dis']
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.Dis.parameters()}],
                                    lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

    def train(self):
        self.g = self.g.int().to(self.info_dict['device'])

        self.get_pred_labels()
        if self.info_dict['reset']:
            self.model.reset_param()
        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['eta'] == 0:
                # override the estimated labels by the given epoch gap
                self.get_pred_labels()
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                save_model_dir = utils.save_model(self.model, self.info_dict)
                if val_acc_epoch > self.best_pretr_val_acc:
                    self.pretr_model_dir = save_model_dir


            print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                  .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        # training samples and labels, for supervised loss
        cls_nids = self.tr_nid
        cls_labels = self.tr_y
        cls_labels = cls_labels.to(self.info_dict['device'])
        # nodes for the contrastive loss part
        ctr_nids = torch.cat((self.val_nid, self.tt_nid))
        ctr_labels_pos = torch.ones_like(ctr_nids).to(self.info_dict['device']).unsqueeze(dim=-1).float()
        ctr_labels_neg = torch.zeros_like(ctr_nids).to(self.info_dict['device']).unsqueeze(dim=-1).float()

        epoch_acc = []
        epoch_loss = torch.FloatTensor([])
        epoch_cls_loss = torch.FloatTensor([])
        epoch_ctr_loss_pos = torch.FloatTensor([])
        epoch_ctr_loss_neg = torch.FloatTensor([])
        epoch_micro_f1 = []
        epoch_macro_f1 = []

        tic = time.time()
        self.model.train()

        # We will only shuffle a part of nodes (determined by n_cls_pershuf) in one forward-backward pass. This
        # value determines how many update step will be conducted in each epoch.
        iter_round = int(np.ceil(self.info_dict['out_dim'] / self.info_dict['n_cls_pershuf']))
        shufarray = np.random.permutation(self.info_dict['out_dim'])
        for k in range(iter_round):
            cls_array = shufarray[k * self.info_dict['n_cls_pershuf']: (k + 1) * self.info_dict['n_cls_pershuf']]
            with torch.set_grad_enabled(True):
                feat = self.g.ndata['feat']
                shuf_feat = self.shuffle_cls_feat(feat, cls_array)
                ori_logits = self.model(self.g, feat)
                shuf_logits = self.model(self.g, shuf_feat)

                pos_nids = self.shuffle_cls_nids(cls_array)
                tp_ori_logits = ori_logits[pos_nids]
                tp_shuf_logits = shuf_logits[pos_nids]

                neg_nids = self.gen_neg_nids()
                neg_ori_logits = ori_logits[neg_nids].detach()

                # the positive part of the contrastive loss
                epoch_ctr_loss_pos_k = torch.Tensor([]).to(self.info_dict['device'])
                if 'F' in self.info_dict['ctr_mode']:
                    pos_score = self.Dis(torch.cat((shuf_logits, ori_logits), dim=-1))
                    pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                    epoch_ctr_loss_pos_k = torch.cat((epoch_ctr_loss_pos_k, pos_loss.unsqueeze(dim=0)), dim=0)
                if 'T' in self.info_dict['ctr_mode']:
                    pos_score = self.Dis(torch.cat((tp_ori_logits, ori_logits), dim=-1))
                    pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                    epoch_ctr_loss_pos_k = torch.cat((epoch_ctr_loss_pos_k, pos_loss.unsqueeze(dim=0)), dim=0)
                if 'M' in self.info_dict['ctr_mode']:
                    pos_score = self.Dis(torch.cat((tp_shuf_logits, ori_logits), dim=-1))
                    pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                    epoch_ctr_loss_pos_k = torch.cat((epoch_ctr_loss_pos_k, pos_loss.unsqueeze(dim=0)), dim=0)
                if 'S' in self.info_dict['ctr_mode']:
                    pos_score = self.Dis(torch.cat((tp_shuf_logits, shuf_logits), dim=-1))
                    pos_loss = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                    epoch_ctr_loss_pos_k = torch.cat((epoch_ctr_loss_pos_k, pos_loss.unsqueeze(dim=0)), dim=0)
                epoch_ctr_loss_pos_k = epoch_ctr_loss_pos_k.mean()
                # the negative part of the contrastive loss
                neg_score = self.Dis(torch.cat((ori_logits, neg_ori_logits), dim=-1))
                epoch_ctr_loss_neg_k = self.bce_fn(neg_score[ctr_nids], ctr_labels_neg)
                epoch_ctr_loss_k = epoch_ctr_loss_pos_k + epoch_ctr_loss_neg_k

                # different settings of supervised loss
                if self.info_dict['cls_mode'] == 'shuf':
                    epoch_cls_loss_k = self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels)
                elif self.info_dict['cls_mode'] == 'raw':
                    epoch_cls_loss_k = self.crs_entropy_fn(ori_logits[cls_nids], cls_labels)
                elif self.info_dict['cls_mode'] == 'both':
                    epoch_cls_loss_k = 0.5 * (self.crs_entropy_fn(ori_logits[cls_nids], cls_labels) +
                                            self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels))
                else:
                    raise ValueError("Unexpected cls_mode parameter: {}".format(self.info_dict['cls_mode']))


                epoch_loss_k = epoch_cls_loss_k + self.info_dict['alpha'] * epoch_ctr_loss_k

                self.opt.zero_grad()
                epoch_loss_k.backward()
                self.opt.step()

                _, preds = torch.max(shuf_logits[cls_nids], dim=1)
                if 'ogb' in self.info_dict['dataset']:
                    epoch_acc_k = self.info_dict['evaluator'].eval(
                        {"y_true": cls_labels.unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
                else:
                    epoch_acc_k = torch.sum(preds == cls_labels).cpu().item() * 1.0 / cls_labels.shape[0]
                epoch_micro_f1_k = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
                epoch_macro_f1_k = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

                epoch_acc.append(epoch_acc_k)
                epoch_loss = torch.cat((epoch_loss, epoch_loss_k.cpu().unsqueeze(dim=0)), dim=0)
                epoch_cls_loss = torch.cat((epoch_cls_loss, epoch_cls_loss_k.cpu().unsqueeze(dim=0)), dim=0)
                epoch_ctr_loss_pos = torch.cat((epoch_ctr_loss_pos, epoch_ctr_loss_pos_k.cpu().unsqueeze(dim=0)), dim=0)
                epoch_ctr_loss_neg = torch.cat((epoch_ctr_loss_neg, epoch_ctr_loss_neg_k.cpu().unsqueeze(dim=0)), dim=0)
                epoch_micro_f1.append(epoch_micro_f1_k)
                epoch_macro_f1.append(epoch_macro_f1_k)


        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), np.mean(epoch_acc)))
        print("cls loss: {:.4f} | ctr pos loss: {:.4f} | ctr neg loss: {:.4f}".format(epoch_cls_loss.mean().item(),
                                                                                      epoch_ctr_loss_pos.mean().item(),
                                                                                      epoch_ctr_loss_neg.mean().item(),
                                                                                      ))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(np.mean(epoch_micro_f1), np.mean(epoch_macro_f1)))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), np.mean(epoch_acc), np.mean(epoch_micro_f1), np.mean(epoch_macro_f1)

    def get_pred_labels(self):

        # load the pretrained model and use it to estimate labels
        cur_model_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.model.eval()
        with torch.set_grad_enabled(False):
            feat = self.g.ndata['feat']
            logits = self.model(self.g, feat)

            _, preds = torch.max(logits, dim=1)
            conf = torch.softmax(logits, dim=1).max(dim=1)[0]
            self.pred_labels = preds
            # training nodes will use their given ground-truth labels instead of the estimated labels
            self.pred_labels[self.tr_nid] = self.labels[self.tr_nid].to(self.info_dict['device'])
            self.pred_conf = conf

            pretr_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / \
                                self.labels[self.val_nid].shape[0]
            pretr_tt_acc = torch.sum(preds[self.tt_nid].cpu() == self.labels[self.tt_nid]).item() * 1.0 / \
                               self.labels[self.tt_nid].shape[0]
            self.best_pretr_val_acc = pretr_val_acc


        # reload the current model's parameters
        self.model.load_state_dict(cur_model_state_dict)

    def shuffle_cls_feat(self, nfeat, cls_array):
        # since there are too many nodes in Ogbn-arxiv, we will shuffle the node indices instead of directly
        # shuffling the feature matrix
        pos_feat = nfeat.clone().detach()

        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes has not been estimated!')

        # generate positive features
        shuf_nid = torch.arange(self.g.num_nodes()).to(self.info_dict['device'])
        for i in cls_array:
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])
        pos_feat[nid] = nfeat[shuf_nid].detach()

        return pos_feat

    def shuffle_cls_nids(self, cls_array):
        # shuffle the nodes in the classes given by cls_array, so as to generate positive paris
        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes are not determined!')

        # shuffle intra-class nids
        shuf_nid = torch.arange(self.g.num_nodes()).to(self.info_dict['device'])
        for i in cls_array:
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])

        return shuf_nid

    def gen_neg_nids(self):
        # randomly generate negative pairs for contrastive learning

        num_nodes = self.g.num_nodes()
        nid = torch.arange(num_nodes)
        labels = self.pred_labels

        shuf_nid = torch.randperm(num_nodes).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # filter out the nodes with the same class as the target we interested in
            sample_prob[i_pos] = 0
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        return shuf_nid
