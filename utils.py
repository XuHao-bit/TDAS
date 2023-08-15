import torch
import random
import math
from math import log2
import numpy as np
import os
import pickle
from torch.utils.tensorboard import SummaryWriter

config = {
    # movielens
    'n_rate': 6,
    'n_year': 81,
    'n_genre': 25,
    'n_director': 2186,
    'n_gender': 2,
    'n_age': 7,
    'n_occupation': 21,
    # bookcrossing
    'n_year_bk': 97,
    'n_author': 84725,
    'n_publisher': 13896,
    'n_age_bk': 99,
    'n_location': 123,
    # dbook
    'n_location_db': 333,
    'n_publisher_db': 1636,
    'n_author_db': 9913,
    'n_year_db': 65,
}

config_settings = { 
    # base model
    'emb_size': 100,
    'task_dim': 20, # task emb size 
    'h_dims': [350, 120, 30], 
    # 'dropout_rate': 0.01, 
    'device': 'cuda:1', 
    # 'device': 'cpu',
    'inner_loop_steps':3,  
    
    # mem
    # 'mem_cluster_k': 3, 
    # 'mem_up_rate': 0.5, 
    
    # hyper param gen model & trainer
    'use_learnable_lr': True, 
    'use_learnable_wd': True,
    'init_lr': 0.01,
    'init_wd': 5e-4,
    'max_shot': 5,
    'n_epoch':25,
    'min_lr':0.001,
    'meta_lr': 0.001, # global update lr
    'local_lr': 0.01, # melu local update lr
    # 'tdmeta_lr': 0.001, # task-adaptive module update lr
    'meta_wd': 0,
    'writer_log': './log/tdmeta_tdmeta_iloop3_train', # 训练超参数变化的log
    'train_log': './log/train_loss_log', # 训练超参数变化的log
    'use_writer': False,
    'use_gen_hypr': True, # 是否使用生成的超参数
    'use_grad_clip': True,
    'clip_norm': 5.,
    'use_ginfo_only': False,
    'use_tinfo_only': False,
    'pstep_only': False,
    'player_only': False,
    'use_tmem': False,
    'clusters_k': 7,
    'temperature': 1.0,
    'n_train_log':'no_0',
    'model_name':'test_model',
    'use_mean_task': False,
    # 'batch_size':5,
    # ablation
    'ada_lr_only': False,
    'ada_wd_only': False,
    
    # dataset
    'data_split':'user_set_0',
    'sample_size': 40,
    'support_size':15,
    'query_size':25,
    # 'data_split':'tdiff_user_0',

}

dataset_info = {
    'movielens': {'n_y': 1, 'u_in_dim': 3, 'i_in_dim': 4, 'num_class':5},
    'bookcrossing': {'n_y': 1, 'u_in_dim': 2, 'i_in_dim': 3, 'num_class':10},
    'dbook': {'n_y': 1, 'u_in_dim': 1, 'i_in_dim': 3, 'num_class':5, 'user_idx':5546, 'item_idx':19320}
}

# ==== [score] ====
def MAE(test_result, ground_truth):
    if len(ground_truth) > 0:
        # pred_y = torch.argmax(test_result, dim=1)
        out = torch.mean(torch.abs(ground_truth-test_result).float(), dim=0)
    else:
        out = 1
    return out

def RMSE(test_result, ground_truth):
    if len(ground_truth) > 0:
        out = torch.sqrt(torch.mean((ground_truth - test_result).float()**2))
    else:
        out = 1
    return out

def NDCG(test_result, ground_truth, top_k=3):
    # print(ground_truth, test_result)
    # pred_y = torch.argmax(test_result, dim=1)
    sort_real_y, sort_real_y_index = ground_truth.clone().detach().sort(descending=True)
    sort_pred_y, sort_pred_y_index = test_result.clone().detach().sort(descending=True)
    pred_sort_y = ground_truth[sort_pred_y_index][:top_k]
    top_pred_y = pred_sort_y
    # top_pred_y, _ = pred_sort_y.sort(descending=True)

    # value: 0-ny-1
    ideal_dcg = 0
    n = 1
    for value in sort_real_y[:top_k]:
        i_dcg = (2**float(value+1) - 1)/log2(n+1)
        ideal_dcg += i_dcg
        n += 1

    pred_dcg = 0
    n = 1
    for value in top_pred_y:
        p_dcg = (2**float(value+1) - 1)/log2(n+1)
        pred_dcg += p_dcg
        n += 1

    n_dcg = pred_dcg/ideal_dcg
    return n_dcg

# ===========
# def nDCG(ranked_list, ground_truth, topn):
#     dcg = 0
#     idcg = IDCG(ground_truth, topn)
#     # print(ranked_list)
#     # input()
#     for i in range(topn):
#         id = ranked_list[i]
#         dcg += ((2 ** ground_truth[id]) -1)/ math.log(i+2, 2)
#     # print('dcg is ', dcg, " n is ", topn)
#     # print('idcg is ', idcg, " n is ", topn)
#     return dcg / idcg, dcg, idcg

# def IDCG(ground_truth,topn):
#     t = [a for a in ground_truth]
#     t.sort(reverse=True)
#     idcg = 0
#     for i in range(topn):
#         idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
#     return idcg

# def add_metric(test_result, ground_truth, ndcg_l, top_k):
#     ndcg_ = NDCG(test_result, ground_truth, top_k)
#     # mae_ = mae(test_result, ground_truth)
#     ndcg_l.append(ndcg_)
#     # mae_l.append(mae_)

# def cal_metric(ndcg_list):
#     # mae = sum(mae_list) / len(mae_list)
#     ndcg = sum(ndcg_list) / len(ndcg_list)
#     return ndcg

def seed_all(seed=1023):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def apply_grad_clip_norm(grads, max_norm):
    device = grads[0].device
    norm_type = 2.0
    total_norm = torch.norm(torch.stack([torch.norm(grad.detach(), norm_type).to(device) for grad in grads]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads:
        grad.detach().mul_(clip_coef_clamped.to(grad.device))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    # ref: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
