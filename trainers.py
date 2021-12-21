import torch
import numpy as np
import dgl
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


class ProgShufTrainer(BaseTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super(ProgShufTrainer, self).__init__(g, model, info_dict, *args, **kwargs)
        if 'TSS' in info_dict['model'] and info_dict['n_step_epochs'] < info_dict['n_epochs']:
            raise ValueError("For TSS model, the update epoch gap should not smaller than the training epochs!")
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

    def train(self):
        self.g = self.g.int().to(self.info_dict['device'])

        self.get_pred_labels(reset_val=False)
        if self.info_dict['reset']:
            self.model.reset_param()
        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['n_step_epochs'] == 0:
                # progressively update the predicted labels
                self.get_pred_labels()
            tic = time.time()
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            toc = time.time()
            self.time_cost_list.append(toc - tic)
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
            if self.info_dict['early_stop']:
                stop_flag = self.stopper.step(val_acc_epoch)
                if stop_flag:
                    print('Early-stop with {} patience'.format(self.info_dict['patience']))
                    break

        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1, np.mean(self.time_cost_list)

    def train_epoch(self, epoch_i):
        # training samples and labels
        nids = self.tr_nid
        labels = self.tr_y

        tic = time.time()
        self.model.train()
        labels = labels.to(self.info_dict['device'])
        with torch.set_grad_enabled(True):
            feat = self.g.ndata['feat']
            shuf_feat = self.shuffle_feat(feat)
            ori_logits = self.model(self.g, feat)
            shuf_logits = self.model(self.g, shuf_feat)

            if self.info_dict['cls_mode'] == 'shuf':
                epoch_loss = self.crs_entropy_fn(shuf_logits[nids], labels)
                logits = shuf_logits
            elif self.info_dict['cls_mode'] == 'raw':
                epoch_loss = self.crs_entropy_fn(ori_logits[nids], labels)
                logits = ori_logits
            elif self.info_dict['cls_mode'] == 'both':
                epoch_loss = 0.5 * (self.crs_entropy_fn(ori_logits[nids], labels) +
                                        self.crs_entropy_fn(shuf_logits[nids], labels))
                logits = 0.5 * (shuf_logits + ori_logits)
            else:
                raise ValueError("Unexpected cls_mode parameter: {}".format(self.info_dict['cls_mode']))

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

    def shuffle_feat(self, nfeat):
        pos_feat = nfeat.clone().detach()

        # nid = self.tr_nid
        # labels = self.tr_y

        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes are not determined!')

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
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])
        pos_feat[nid] = nfeat[shuf_nid].detach()

        if self.info_dict['keep_tr_feat']:
            pos_feat[self.tr_nid] = nfeat[self.tr_nid].clone().detach()

        return pos_feat

    def get_pred_labels(self, reset_val=False):

        # load pretrained model and use it to estimate the labels
        cur_model_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.model.eval()
        with torch.set_grad_enabled(False):
            feat = self.g.ndata['feat']
            logits = self.model(self.g, feat)

            _, preds = torch.max(logits, dim=1)
            conf = torch.softmax(logits, dim=1).max(dim=1)[0]
            self.pred_labels = preds
            self.pred_labels[self.tr_nid] = self.labels[self.tr_nid].to(self.info_dict['device'])
            self.pred_conf = conf

            pretr_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / self.labels[self.val_nid].shape[0]
            pretr_tt_acc = torch.sum(preds[self.tt_nid].cpu() == self.labels[self.tt_nid]).item() * 1.0 / self.labels[self.tt_nid].shape[0]
            self.best_pretr_val_acc = pretr_val_acc

            if reset_val:
                self.best_val_acc = pretr_val_acc
                self.best_tt_acc = pretr_tt_acc

        # reload the current model's parameters
        self.model.load_state_dict(cur_model_state_dict)


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
            if i % self.info_dict['n_step_epochs'] == 0:
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

            epoch_loss = epoch_cls_loss + self.info_dict['wctr'] * epoch_ctr_loss

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


class ProgClsShufTrainer(BaseTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)
        if 'TSS' in info_dict['model'] and info_dict['n_step_epochs'] < info_dict['n_epochs']:
            raise ValueError("For TSS model, the update epoch gap should not smaller than the training epochs!")
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

    def train(self):
        self.g = self.g.int().to(self.info_dict['device'])

        self.get_pred_labels(reset_val=False)
        if self.info_dict['reset']:
            self.model.reset_param()
        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['n_step_epochs'] == 0:
                # progressively update the predicted labels
                self.get_pred_labels()
            tic = time.time()
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            toc = time.time()
            self.time_cost_list.append(toc - tic)
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
            if self.info_dict['early_stop']:
                stop_flag = self.stopper.step(val_acc_epoch)
                if stop_flag:
                    print('Early-stop with {} patience'.format(self.info_dict['patience']))
                    break

        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1, np.mean(self.time_cost_list)

    def train_epoch(self, epoch_i):
        # training samples and labels
        nids = self.tr_nid
        labels = self.tr_y

        epoch_acc = []
        epoch_loss = torch.FloatTensor([])
        epoch_micro_f1 = []
        epoch_macro_f1 = []

        tic = time.time()
        self.model.train()
        labels = labels.to(self.info_dict['device'])

        iter_round = int(np.ceil(self.info_dict['out_dim'] / self.info_dict['n_cls_pershuf']))
        shufarray = np.random.permutation(self.info_dict['out_dim'])
        for k in range(iter_round):
            cls_array = shufarray[k * self.info_dict['n_cls_pershuf']: (k + 1) * self.info_dict['n_cls_pershuf']]
            with torch.set_grad_enabled(True):
                feat = self.g.ndata['feat']
                shuf_feat = self.shuffle_cls_feat(feat, cls_array)
                logits = self.model(self.g, shuf_feat)
                epoch_loss_cls = self.crs_entropy_fn(logits[nids], labels)

                self.opt.zero_grad()
                epoch_loss_cls.backward()
                self.opt.step()

                _, preds = torch.max(logits[nids], dim=1)
                if 'ogb' in self.info_dict['dataset']:
                    epoch_acc_cls = self.info_dict['evaluator'].eval(
                        {"y_true": labels.unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
                else:
                    epoch_acc_cls = torch.sum(preds == labels).cpu().item() * 1.0 / labels.shape[0]
                epoch_micro_f1_cls = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
                epoch_macro_f1_cls = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

                epoch_acc.append(epoch_acc_cls)
                epoch_loss = torch.cat((epoch_loss, epoch_loss_cls.cpu().unsqueeze(dim=0)), dim=0)
                epoch_micro_f1.append(epoch_micro_f1_cls)
                epoch_macro_f1.append(epoch_macro_f1_cls)

            # print("Epoch {} step {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, k, epoch_loss_cls.cpu().item(), epoch_acc_cls))
            # print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), np.mean(epoch_acc)))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(np.mean(epoch_micro_f1), np.mean(epoch_macro_f1)))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), np.mean(epoch_acc), np.mean(epoch_micro_f1), np.mean(epoch_macro_f1)

    def shuffle_cls_feat(self, nfeat, cls_array):
        pos_feat = nfeat.clone().detach()

        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes are not determined!')

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

        if self.info_dict['keep_tr_feat']:
            pos_feat[self.tr_nid] = nfeat[self.tr_nid].clone().detach()

        return pos_feat

    def get_pred_labels(self, reset_val=False):

        # load pretrained model and use it to estimate the labels
        cur_model_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.model.eval()
        with torch.set_grad_enabled(False):
            feat = self.g.ndata['feat']
            logits = self.model(self.g, feat)

            _, preds = torch.max(logits, dim=1)
            conf = torch.softmax(logits, dim=1).max(dim=1)[0]
            self.pred_labels = preds
            self.pred_labels[self.tr_nid] = self.labels[self.tr_nid].to(self.info_dict['device'])
            self.pred_conf = conf

            pretr_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / \
                                self.labels[self.val_nid].shape[0]
            pretr_tt_acc = torch.sum(preds[self.tt_nid].cpu() == self.labels[self.tt_nid]).item() * 1.0 / \
                               self.labels[self.tt_nid].shape[0]
            self.best_pretr_val_acc = pretr_val_acc

            if reset_val:
                self.best_val_acc = pretr_val_acc
                self.best_tt_acc = pretr_tt_acc

        # reload the current model's parameters
        self.model.load_state_dict(cur_model_state_dict)


class ProgClsContraTrainer(ProgClsShufTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)

        self.Dis = kwargs['Dis']
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.Dis.parameters()}],
                                    lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

    def train_epoch(self, epoch_i):
        # training samples and labels
        cls_nids = self.tr_nid
        cls_labels = self.tr_y
        cls_labels = cls_labels.to(self.info_dict['device'])
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

        iter_round = int(np.ceil(self.info_dict['out_dim'] / self.info_dict['n_cls_pershuf']))
        shufarray = np.random.permutation(self.info_dict['out_dim'])
        for k in range(iter_round):
            cls_array = shufarray[k * self.info_dict['n_cls_pershuf']: (k + 1) * self.info_dict['n_cls_pershuf']]
            with torch.set_grad_enabled(True):
                feat = self.g.ndata['feat']
                shuf_feat = self.shuffle_cls_feat(feat, cls_array)
                ori_logits = self.model(self.g, feat)
                shuf_logits = self.model(self.g, shuf_feat)
                # neg_logits = self.gen_neg_feat(shuf_logits)
                neg_logits = self.gen_neg_feat(ori_logits)

                pos_score = self.Dis(torch.cat((shuf_logits, ori_logits), dim=-1))
                neg_score = self.Dis(torch.cat((ori_logits, neg_logits), dim=-1))

                if self.info_dict['shuf_cls']:
                    epoch_cls_loss_k = self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels)
                else:
                    epoch_cls_loss_k = self.crs_entropy_fn(ori_logits[cls_nids], cls_labels)
                epoch_ctr_loss_pos_k = self.bce_fn(pos_score[ctr_nids], ctr_labels_pos)
                epoch_ctr_loss_neg_k = self.bce_fn(neg_score[ctr_nids], ctr_labels_neg)
                epoch_ctr_loss_k = epoch_ctr_loss_pos_k + epoch_ctr_loss_neg_k

                epoch_loss_k = self.info_dict['wcls'] * epoch_cls_loss_k + self.info_dict['wctr'] * epoch_ctr_loss_k

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

            # print("Epoch {} step {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, k, epoch_loss_k.cpu().item(), epoch_acc_k))
            # print("cls loss: {:.4f} | ctr pos loss: {:.4f} | ctr neg loss: {:.4f}".format(epoch_cls_loss_k.cpu().item(),
            #                                                                               epoch_ctr_loss_pos_k.cpu().item(),
            #                                                                               epoch_ctr_loss_neg_k.cpu().item(),
            #                                                                               ))
            # print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1_k, epoch_macro_f1_k))

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), np.mean(epoch_acc)))
        print("cls loss: {:.4f} | ctr pos loss: {:.4f} | ctr neg loss: {:.4f}".format(epoch_cls_loss.mean().item(),
                                                                                      epoch_ctr_loss_pos.mean().item(),
                                                                                      epoch_ctr_loss_neg.mean().item(),
                                                                                      ))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(np.mean(epoch_micro_f1), np.mean(epoch_macro_f1)))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), np.mean(epoch_acc), np.mean(epoch_micro_f1), np.mean(epoch_macro_f1)

    def gen_neg_feat(self, nfeat):
        neg_feat = nfeat.clone().detach()
        nid = torch.arange(self.g.num_nodes()).to(self.info_dict['device'])
        labels = self.pred_labels

        shuf_nid = torch.zeros_like(nid).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            sample_prob[i_pos] = 0
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        neg_feat = neg_feat[shuf_nid]
        return neg_feat


class ProgClsModeContraTrainer(ProgClsShufTrainer):
    '''
    for each update step, only shuffle a fix number (self.info_dict['n_cls_pershuf']) of classes
    '''
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)
        self.Dis = kwargs['Dis']
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.Dis.parameters()}],
                                    lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

    def train_epoch(self, epoch_i):
        # training samples and labels
        cls_nids = self.tr_nid
        cls_labels = self.tr_y
        cls_labels = cls_labels.to(self.info_dict['device'])
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
                neg_shuf_logits = shuf_logits[neg_nids]
                # neg_logits = self.gen_neg_feat(ori_logits)

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

                neg_score = self.Dis(torch.cat((ori_logits, neg_ori_logits), dim=-1))
                epoch_ctr_loss_neg_k = self.bce_fn(neg_score[ctr_nids], ctr_labels_neg)
                epoch_ctr_loss_k = epoch_ctr_loss_pos_k + epoch_ctr_loss_neg_k

                if self.info_dict['cls_mode'] == 'shuf':
                    epoch_cls_loss_k = self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels)
                elif self.info_dict['cls_mode'] == 'raw':
                    epoch_cls_loss_k = self.crs_entropy_fn(ori_logits[cls_nids], cls_labels)
                elif self.info_dict['cls_mode'] == 'both':
                    epoch_cls_loss_k = 0.5 * (self.crs_entropy_fn(ori_logits[cls_nids], cls_labels) +
                                            self.crs_entropy_fn(shuf_logits[cls_nids], cls_labels))
                else:
                    raise ValueError("Unexpected cls_mode parameter: {}".format(self.info_dict['cls_mode']))


                epoch_loss_k = self.info_dict['wcls'] * epoch_cls_loss_k + self.info_dict['wctr'] * epoch_ctr_loss_k

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

    def gen_neg_feat(self, nfeat):
        neg_feat = nfeat.clone().detach()
        nid = torch.arange(self.g.num_nodes()).to(self.info_dict['device'])
        labels = self.pred_labels

        shuf_nid = torch.zeros_like(nid).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            sample_prob[i_pos] = 0
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        neg_feat = neg_feat[shuf_nid]
        return neg_feat

    def shuffle_cls_nids(self, cls_array):
        nid = torch.arange(self.g.num_nodes())
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes are not determined!')

        # shuffle intra-class nids
        shuf_nid = torch.arange(self.g.num_nodes()).to(self.info_dict['device'])
        for i in cls_array:
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]

            # workaround: skip the classes that doesn't appear in the training set (the case of ogbn-products)
            if len(i_pos) == 0:
                continue
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])

        if self.info_dict['keep_tr_feat']:
            shuf_nid[self.tr_nid] = nid[self.tr_nid].to(self.info_dict['device'])

        return shuf_nid

    def gen_neg_nids(self):
        num_nodes = self.g.num_nodes()
        nid = torch.arange(num_nodes)
        labels = self.pred_labels

        shuf_nid = torch.randperm(num_nodes).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]

            # workaround: skip the classes that doesn't appear in the training set (the case of ogbn-products)
            if len(i_pos) == 0:
                continue
            sample_prob[i_pos] = 0
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        return shuf_nid


class BaseMBTrainer(object):
    '''
    Base minibatch trainer for training.
    Works for models with only one output.
    '''

    def __init__(self, graphs, model, info_dict, *args, **kwargs):
        self.model = model
        self.info_dict = info_dict
        self.tr_g, self.val_g, self.tt_g = graphs
        # self.tr_g = self.tr_g.to(self.info_dict['device'])
        # self.val_g = self.val_g.to(self.info_dict['device'])
        # self.tt_g = self.tt_g.to(self.info_dict['device'])

        # load train/val/test split
        self.tr_nid = self.tt_g.ndata['train_mask'].nonzero().squeeze()
        self.val_nid = self.tt_g.ndata['val_mask'].nonzero().squeeze()
        self.tt_nid = self.tt_g.ndata['test_mask'].nonzero().squeeze()
        self.labels = self.tt_g.ndata['label']
        self.tr_y = self.labels[self.tr_nid]
        self.val_y = self.labels[self.val_nid]
        self.tt_y = self.labels[self.tt_nid]

        # construct dataloaders for each phases
        self.build_dataloader()

        self.crs_entropy_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

        self.best_val_acc = 0
        self.best_tt_acc = 0
        self.best_microf1 = 0
        self.best_macrof1 = 0

        if info_dict['early_stop']:
            self.stopper = EarlyStopping(info_dict['patience'])

    def build_dataloader(self):
        # build dataloaders
        # training loader
        tr_sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(num_neb) for num_neb in self.info_dict['num_neighbors'].split('-')]
        )
        self.tr_dataloader = dgl.dataloading.NodeDataLoader(
            self.tr_g,
            self.tr_nid,
            tr_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )

        val_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.info_dict['n_layers'])
        self.val_dataloader = dgl.dataloading.NodeDataLoader(
            self.val_g,
            self.val_nid,
            val_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )
        self.tt_dataloader = dgl.dataloading.NodeDataLoader(
            self.tt_g,
            self.tt_nid,
            val_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )

    def train(self):
        for i in range(self.info_dict['n_epochs']):
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            val_loss_epoch, val_acc_epoch, val_microf1, val_macrof1 = self.eval_epoch(state='val')
            tt_loss_epoch, tt_acc_epoch, tt_microf1, tt_macrof1 = self.eval_epoch(state='test')
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1
                self.best_macrof1 = tt_macrof1
                _ = utils.save_model(self.model, self.info_dict)

            print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                  .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))
            if self.info_dict['early_stop']:
                stop_flag = self.stopper.step(val_acc_epoch)
                if stop_flag:
                    print('Early-stop with {} patience'.format(self.info_dict['patience']))
                    break

        # save the final epoch model
        _ = utils.save_model(self.model, self.info_dict, state='fin')
        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        tic = time.time()
        self.model.train()

        preds = torch.LongTensor([])
        epoch_loss = torch.Tensor([])
        nids = torch.LongTensor([])
        for step, (input_nodes, seeds, blocks) in enumerate(self.tr_dataloader):
            batch_inputs = self.tr_g.ndata['feat'][input_nodes].to(self.info_dict['device'])
            batch_labels = self.labels[seeds.cpu()].to(self.info_dict['device'])
            blocks = [block.int().to(self.info_dict['device']) for block in blocks]

            nids = torch.cat((nids, seeds.cpu()))

            with torch.set_grad_enabled(True):
                batch_logits = self.model(blocks, batch_inputs)
                batch_loss = self.crs_entropy_fn(batch_logits, batch_labels)
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

                preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)
                epoch_loss = torch.cat((epoch_loss, batch_loss.cpu().unsqueeze(dim=0)))

                # print training info
                if step % 50 == 0:
                    step_acc = torch.sum(torch.max(batch_logits.cpu(), dim=1)[1] == self.labels[seeds.cpu()]).item() * 1.0 / seeds.shape[0]
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                            epoch_i, step, batch_loss.item(), step_acc, gpu_mem_alloc))

        if 'ogb' in self.info_dict['dataset']:
            epoch_acc = self.info_dict['evaluator'].eval(
                {"y_true": self.labels[nids].unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
        else:
            epoch_acc = torch.sum(preds == self.labels[nids]).cpu().item() * 1.0 / nids.shape[0]
        epoch_micro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="micro")
        epoch_macro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def eval_epoch(self, state='val'):
        tic = time.time()
        self.model.eval()

        if state == 'val':
            dataloader = self.val_dataloader
            # labels = self.val_y
            g = self.val_g
        else:
            dataloader = self.tt_dataloader
            # labels = self.tt_y
            g = self.tt_g

        preds = torch.LongTensor([])
        epoch_loss = torch.Tensor([])
        nids = torch.LongTensor([])
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_inputs = g.ndata['feat'][input_nodes].to(self.info_dict['device'])
            batch_labels = self.labels[seeds.cpu()].to(self.info_dict['device'])
            blocks = [block.int().to(self.info_dict['device']) for block in blocks]

            nids = torch.cat((nids, seeds.cpu()))

            with torch.set_grad_enabled(False):
                batch_logits = self.model(blocks, batch_inputs)
                batch_loss = self.crs_entropy_fn(batch_logits, batch_labels)

                preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)
                epoch_loss = torch.cat((epoch_loss, batch_loss.cpu().unsqueeze(dim=0)))

        if 'ogb' in self.info_dict['dataset']:
            epoch_acc = self.info_dict['evaluator'].eval(
                {"y_true": self.tr_y[nids].unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
        else:
            epoch_acc = torch.sum(preds == self.labels[nids]).cpu().item() * 1.0 / nids.shape[0]
        epoch_micro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="micro")
        epoch_macro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print(
            "{} stage | Loss: {:.4f} | accuracy: {:.4f}".format(state, epoch_loss.mean().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1


class BaseMBTrainerGTFeat(BaseMBTrainer):
    '''
    Base minibatch trainer for training with ground-truth shuffling.
    '''

    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)

    def train_epoch(self, epoch_i):
        tic = time.time()
        self.model.train()

        preds = torch.LongTensor([])
        epoch_loss = torch.Tensor([])
        nids = torch.LongTensor([])
        shuf_nids = self.shuffle_nids().to(self.info_dict['device'])
        for step, (input_nodes, seeds, blocks) in enumerate(self.tr_dataloader):
            shuf_input_nodes = shuf_nids[input_nodes]
            batch_inputs = self.tr_g.ndata['feat'][shuf_input_nodes].to(self.info_dict['device'])
            batch_labels = self.labels[seeds.cpu()].to(self.info_dict['device'])
            blocks = [block.int().to(self.info_dict['device']) for block in blocks]

            nids = torch.cat((nids, seeds.cpu()))

            with torch.set_grad_enabled(True):
                batch_logits = self.model(blocks, batch_inputs)
                batch_loss = self.crs_entropy_fn(batch_logits, batch_labels)
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

                preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)
                epoch_loss = torch.cat((epoch_loss, batch_loss.cpu().unsqueeze(dim=0)))

                # print training info
                if step % 50 == 0:
                    step_acc = torch.sum(torch.max(batch_logits.cpu(), dim=1)[1] == self.labels[seeds.cpu()]).item() * 1.0 / seeds.shape[0]
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                            epoch_i, step, batch_loss.item(), step_acc, gpu_mem_alloc))

        if 'ogb' in self.info_dict['dataset']:
            epoch_acc = self.info_dict['evaluator'].eval(
                {"y_true": self.labels[nids].unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
        else:
            epoch_acc = torch.sum(preds == self.labels[nids]).cpu().item() * 1.0 / nids.shape[0]
        epoch_micro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="micro")
        epoch_macro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def shuffle_nids(self):

        num_nodes = self.tr_g.num_nodes()
        nid = torch.arange(num_nodes)
        labels = self.tr_g.ndata['label']

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
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])

        if self.info_dict['keep_tr_feat']:
            shuf_nid[self.tr_nid] = self.tr_nid

        return shuf_nid


class BaseMBTrainerClsGTFeat(BaseMBTrainer):
    '''
    Base minibatch trainer for training with ground-truth shuffling.
    '''

    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)

    def train_epoch(self, epoch_i):
        tic = time.time()
        self.model.train()

        preds = torch.LongTensor([])
        epoch_loss = torch.Tensor([])
        nids = torch.LongTensor([])
        for step, (input_nodes, seeds, blocks) in enumerate(self.tr_dataloader):
            shufarray = np.random.permutation(self.info_dict['out_dim'])
            shuf_nids = self.shuffle_nids(shufarray[: self.info_dict['n_cls_pershuf']]).to(self.info_dict['device'])
            shuf_input_nodes = shuf_nids[input_nodes]
            batch_inputs = self.tr_g.ndata['feat'][shuf_input_nodes].to(self.info_dict['device'])
            batch_labels = self.labels[seeds.cpu()].to(self.info_dict['device'])
            blocks = [block.int().to(self.info_dict['device']) for block in blocks]

            nids = torch.cat((nids, seeds.cpu()))

            with torch.set_grad_enabled(True):
                batch_logits = self.model(blocks, batch_inputs)
                batch_loss = self.crs_entropy_fn(batch_logits, batch_labels)
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

                preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)
                epoch_loss = torch.cat((epoch_loss, batch_loss.cpu().unsqueeze(dim=0)))

                # print training info
                if step % 50 == 0:
                    step_acc = torch.sum(torch.max(batch_logits.cpu(), dim=1)[1] == self.labels[seeds.cpu()]).item() * 1.0 / seeds.shape[0]
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                            epoch_i, step, batch_loss.item(), step_acc, gpu_mem_alloc))

        if 'ogb' in self.info_dict['dataset']:
            epoch_acc = self.info_dict['evaluator'].eval(
                {"y_true": self.labels[nids].unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
        else:
            epoch_acc = torch.sum(preds == self.labels[nids]).cpu().item() * 1.0 / nids.shape[0]
        epoch_micro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="micro")
        epoch_macro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def shuffle_nids(self, cls_array):

        num_nodes = self.tr_g.num_nodes()
        nid = torch.arange(num_nodes)
        labels = self.tr_g.ndata['label']

        # generate positive features
        shuf_nid = torch.arange(num_nodes).to(self.info_dict['device'])
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

        if self.info_dict['keep_tr_feat']:
            shuf_nid[self.tr_nid] = self.tr_nid.to(self.info_dict['device'])

        return shuf_nid


class ProgShufMBTrainer(BaseMBTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)
        if 'TSS' in info_dict['model'] and info_dict['n_step_epochs'] < info_dict['n_epochs']:
            raise ValueError("For TSS model, the update epoch gap should not smaller than the training epochs!")
        self.pred_labels = None
        self.pred_conf = None
        self.pretr_model_dir = os.path.join('exp', info_dict['backbone'], info_dict['dataset'],
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

    def build_dataloader(self):
        # build dataloaders
        # training loader
        tr_sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(num_neb) for num_neb in self.info_dict['num_neighbors'].split('-')]
        )
        self.tr_dataloader = dgl.dataloading.NodeDataLoader(
            self.tr_g,
            self.tr_nid,
            tr_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=True,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )

        val_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.info_dict['n_layers'])
        self.val_dataloader = dgl.dataloading.NodeDataLoader(
            self.val_g,
            self.val_nid,
            val_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )
        self.tt_dataloader = dgl.dataloading.NodeDataLoader(
            self.tt_g,
            self.tt_nid,
            val_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )

        self.full_dataloader = dgl.dataloading.NodeDataLoader(
            self.tt_g,
            torch.arange(self.tt_g.num_nodes()),
            val_sampler,
            batch_size=self.info_dict['batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=self.info_dict['num_cpu'],
        )

    def train(self):
        self.get_pred_labels(reset_val=False)
        if self.info_dict['reset']:
            self.model.reset_param()
        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['n_step_epochs'] == 0:
                # progressively update the predicted labels
                self.get_pred_labels()
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            val_loss_epoch, val_acc_epoch, val_microf1, val_macrof1 = self.eval_epoch(state='val')
            tt_loss_epoch, tt_acc_epoch, tt_microf1, tt_macrof1 = self.eval_epoch(state='test')
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1
                self.best_macrof1 = tt_macrof1
                _ = utils.save_model(self.model, self.info_dict)

            print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                  .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))
            if self.info_dict['early_stop']:
                stop_flag = self.stopper.step(val_acc_epoch)
                if stop_flag:
                    print('Early-stop with {} patience'.format(self.info_dict['patience']))
                    break

        # save the final epoch model
        self.pretr_model_dir = utils.save_model(self.model, self.info_dict, state='fin')
        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        tic = time.time()
        self.model.train()

        preds = torch.LongTensor([])
        epoch_loss = torch.Tensor([])
        nids = torch.LongTensor([])
        shuf_nids = self.shuffle_nids().to(self.info_dict['device'])
        for step, (input_nodes, seeds, blocks) in enumerate(self.tr_dataloader):
            shuf_input_nodes = shuf_nids[input_nodes]
            batch_inputs = self.tr_g.ndata['feat'][shuf_input_nodes].to(self.info_dict['device'])
            batch_labels = self.labels[seeds.cpu()].to(self.info_dict['device'])
            blocks = [block.int().to(self.info_dict['device']) for block in blocks]

            nids = torch.cat((nids, seeds.cpu()))

            with torch.set_grad_enabled(True):
                batch_logits = self.model(blocks, batch_inputs)
                batch_loss = self.crs_entropy_fn(batch_logits, batch_labels)
                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

                preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)
                epoch_loss = torch.cat((epoch_loss, batch_loss.cpu().unsqueeze(dim=0)))

                # print training info
                if step % 50 == 0:
                    step_acc = torch.sum(torch.max(batch_logits.cpu(), dim=1)[1] == self.labels[seeds.cpu()]).item() * 1.0 / seeds.shape[0]
                    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                            epoch_i, step, batch_loss.item(), step_acc, gpu_mem_alloc))

        if 'ogb' in self.info_dict['dataset']:
            epoch_acc = self.info_dict['evaluator'].eval(
                {"y_true": self.labels[nids].unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
        else:
            epoch_acc = torch.sum(preds == self.labels[nids]).cpu().item() * 1.0 / nids.shape[0]
        epoch_micro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="micro")
        epoch_macro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), epoch_acc))
        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def shuffle_nids(self):

        labels = self.pred_labels
        num_nodes = self.pred_labels.shape[0]
        nid = torch.arange(num_nodes)

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
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])

        if self.info_dict['keep_tr_feat']:
            shuf_nid[self.tr_nid] = self.tr_nid

        return shuf_nid

    def get_pred_labels(self, reset_val=False):
        if self.info_dict['inductive']:
            self.pred_labels = self.tr_g.ndata['label']
            self.pred_conf = torch.ones_like(self.pred_labels).float()
            if reset_val:
                preds = torch.LongTensor([])
                for step, (input_nodes, seeds, blocks) in enumerate(self.full_dataloader):
                    batch_inputs = self.tt_g.ndata['feat'][input_nodes].to(self.info_dict['device'])
                    blocks = [block.int().to(self.info_dict['device']) for block in blocks]

                    with torch.set_grad_enabled(False):
                        batch_logits = self.model(blocks, batch_inputs)
                        batch_conf = torch.softmax(batch_logits, dim=1).max(dim=1)[0].cpu()

                        preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)

                self.best_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / \
                                    self.labels[self.val_nid].shape[0]
                self.best_tt_acc = torch.sum(preds[self.tt_nid].cpu() == self.labels[self.tt_nid]).item() * 1.0 / \
                                   self.labels[self.tt_nid].shape[0]
        else:
            # load pretrained model and use it to estimate the labels
            cur_model_state_dict = deepcopy(self.model.state_dict())
            self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))
            self.model = self.model.eval()
            preds = torch.LongTensor([])
            nids = torch.LongTensor([])
            conf = torch.FloatTensor([])
            for step, (input_nodes, seeds, blocks) in enumerate(self.full_dataloader):
                batch_inputs = self.tt_g.ndata['feat'][input_nodes].to(self.info_dict['device'])
                blocks = [block.int().to(self.info_dict['device']) for block in blocks]

                nids = torch.cat((nids, seeds.cpu()))
                with torch.set_grad_enabled(False):
                    batch_logits = self.model(blocks, batch_inputs)
                    batch_conf = torch.softmax(batch_logits, dim=1).max(dim=1)[0].cpu()

                    preds = torch.cat((preds, torch.max(batch_logits.cpu(), dim=1)[1]), dim=0)
                    conf = torch.cat((conf, batch_conf.cpu()), dim=0)

            self.pred_labels = preds
            self.pred_labels[self.tr_nid] = self.labels[self.tr_nid]
            self.pred_conf = conf
            self.pred_conf[self.tr_nid] = torch.ones_like(self.tr_nid).float()

            if reset_val:
                self.best_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / \
                                    self.labels[self.val_nid].shape[0]
                self.best_tt_acc = torch.sum(preds[self.tt_nid].cpu() == self.labels[self.tt_nid]).item() * 1.0 / \
                                   self.labels[self.tt_nid].shape[0]

            # reload the current model's parameters
            self.model.load_state_dict(cur_model_state_dict)


class ProgContraMBTrainer(ProgShufMBTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)
        self.pred_labels = None
        self.pred_conf = None
        self.pretr_model_dir = os.path.join('exp', info_dict['backbone'], info_dict['dataset'],
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

    def train_epoch(self, epoch_i):
        tic = time.time()
        self.model.train()

        preds = torch.LongTensor([])
        epoch_loss = torch.Tensor([])
        epoch_cls_loss = torch.Tensor([])
        epoch_ctr_loss_pos = torch.Tensor([])
        epoch_ctr_loss_neg = torch.Tensor([])
        nids = torch.LongTensor([])
        shuf_nids = self.shuffle_nids().to(self.info_dict['device'])
        for step, (input_nodes, seeds, blocks) in enumerate(self.tr_dataloader):
            shuf_input_nodes = shuf_nids[input_nodes]
            batch_inputs = self.tr_g.ndata['feat'][input_nodes].to(self.info_dict['device'])
            batch_shuffle_inputs = self.tr_g.ndata['feat'][shuf_input_nodes].to(self.info_dict['device'])
            batch_labels = self.labels[seeds.cpu()].to(self.info_dict['device'])
            blocks = [block.int().to(self.info_dict['device']) for block in blocks]

            nids = torch.cat((nids, seeds.cpu()))

            ctr_labels_pos = torch.ones_like(seeds).to(self.info_dict['device']).unsqueeze(dim=-1).float()
            ctr_labels_neg = torch.zeros_like(seeds).to(self.info_dict['device']).unsqueeze(dim=-1).float()

            with torch.set_grad_enabled(True):
                batch_logits = self.model(blocks, batch_inputs)
                batch_shuffle_logits = self.model(blocks, batch_shuffle_inputs)
                batch_shuf_neg_ind = self.gen_batch_neg_nids(seeds, batch_labels)
                batch_neg_logits = batch_logits[batch_shuf_neg_ind]

                pos_score = self.Dis(torch.cat((batch_shuffle_logits, batch_logits), dim=-1))
                neg_score = self.Dis(torch.cat((batch_logits, batch_neg_logits), dim=-1))
                if self.info_dict['shuf_cls']:
                    batch_cls_loss = self.crs_entropy_fn(batch_shuffle_logits, batch_labels)
                else:
                    batch_cls_loss = self.crs_entropy_fn(batch_logits, batch_labels)

                batch_ctr_loss_pos = self.bce_fn(pos_score, ctr_labels_pos)
                batch_ctr_loss_neg = self.bce_fn(neg_score, ctr_labels_neg)
                batch_ctr_loss = batch_ctr_loss_pos + batch_ctr_loss_neg

                batch_loss = self.info_dict['wcls'] * batch_cls_loss + self.info_dict['wctr'] * batch_ctr_loss

                self.opt.zero_grad()
                batch_loss.backward()
                self.opt.step()

            preds = torch.cat((preds, torch.max(batch_shuffle_logits.cpu(), dim=1)[1]), dim=0)
            epoch_loss = torch.cat((epoch_loss, batch_loss.cpu().unsqueeze(dim=0)))
            epoch_cls_loss = torch.cat((epoch_cls_loss, batch_cls_loss.cpu().unsqueeze(dim=0)))
            epoch_ctr_loss_pos = torch.cat((epoch_ctr_loss_pos, batch_ctr_loss_pos.cpu().unsqueeze(dim=0)))
            epoch_ctr_loss_neg = torch.cat((epoch_ctr_loss_neg, batch_ctr_loss_neg.cpu().unsqueeze(dim=0)))

            # print training info
            if step % 50 == 0:
                step_acc = torch.sum(torch.max(batch_logits.cpu(), dim=1)[1] == self.labels[seeds.cpu()]).item() * 1.0 / seeds.shape[0]
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                        epoch_i, step, batch_loss.item(), step_acc, gpu_mem_alloc))

        if 'ogb' in self.info_dict['dataset']:
            epoch_acc = self.info_dict['evaluator'].eval(
                {"y_true": self.labels[nids].unsqueeze(-1), "y_pred": preds.unsqueeze(-1)})['acc']
        else:
            epoch_acc = torch.sum(preds == self.labels[nids]).cpu().item() * 1.0 / nids.shape[0]
        epoch_micro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="micro")
        epoch_macro_f1 = metrics.f1_score(self.labels[nids].cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.mean().item(), epoch_acc))
        print("cls loss: {:.4f} | ctr pos loss: {:.4f} | ctr neg loss: {:.4f}".format(epoch_cls_loss.mean().item(),
                                                                                      epoch_ctr_loss_pos.mean().item(),
                                                                                      epoch_ctr_loss_neg.mean().item(),
                                                                                      ))

        print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.mean().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def gen_batch_neg_nids(self, seeds, labels):
        num_nodes = seeds.shape[0]
        nid = torch.arange(num_nodes)

        shuf_nid = torch.zeros_like(nid).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            if len(i_pos) == 0:
                continue
            sample_prob[i_pos] = 1e-8
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        return shuf_nid