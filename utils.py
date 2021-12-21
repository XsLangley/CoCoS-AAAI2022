import torch
import os


def save_model(model, info_dict, state='val'):
    suffix = 'ori' if info_dict['split'] == 'None' else '_'.join(info_dict['split'].split('-'))
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