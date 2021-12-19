import numpy as np
import torch
import os

class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc):
        score = acc
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

    # def step(self, acc, model):
    #     score = acc
    #     if self.best_score is None:
    #         self.best_score = score
    #         self.save_checkpoint(model)
    #     elif score < self.best_score:
    #         self.counter += 1
    #         print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
    #         if self.counter >= self.patience:
    #             self.early_stop = True
    #     else:
    #         self.best_score = score
    #         self.save_checkpoint(model)
    #         self.counter = 0
    #     return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'es_checkpoint.pt')


def save_model(model, info_dict, state='val'):
    suffix = 'ori' if info_dict['split'] == 'none' else '_'.join(info_dict['split'].split('-'))
    save_root = os.path.join('exp', info_dict['model'] + '_' + suffix, info_dict['dataset'])
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # checkpoint name
    ckpname = '{model}_{db}_{seed}{agg}_{state}.pt'.\
        format(model=info_dict['model'], db=info_dict['dataset'],
               seed=info_dict['seed'],
               agg='_' + info_dict['agg_type'] if 'SAGE' in info_dict['model'] else '',
               state=state)
    savedir = os.path.join(save_root, ckpname)
    torch.save(model.state_dict(), savedir)
    return savedir