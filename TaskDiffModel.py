# from pickle import NEXT_BUFFER
import torch
from torch import nn
import numpy as np

from modules.BaseRecModel import *
from modules.HyperParamModel import TaskAdaptiveHyperParameterOptimizer
from modules.TaskEncoder import TaskEncoder, TaskMemEncoder, TaskMeanEncoder, TaskMeanMemEncoder

from DataLoading import *

# 
class TaskDiffModel(nn.Module):
    def __init__(self, dataset, emb_info, dataset_info, config_settings) -> None:
        super().__init__()
        # dataset info
        self.dataset = dataset
        self.n_y = dataset_info[self.dataset]['n_y']
        self.num_class = dataset_info[self.dataset]['num_class']
        self.u_in_dim = dataset_info[self.dataset]['u_in_dim']
        self.i_in_dim = dataset_info[self.dataset]['i_in_dim']
        # base model
        self.emb_size = config_settings['emb_size']
        self.x_dim = (self.u_in_dim+self.i_in_dim) * self.emb_size
        self.task_dim = config_settings['task_dim']
        # self.h_dims = config_settings['h_dims']
        # self.dropout_rate = config_settings['dropout_rate']
        self.device = config_settings['device']
        self.inner_loop_steps = config_settings['inner_loop_steps']
        # mem
        # self.mem_cluster_k = config_settings['mem_cluster_k']
        # self.mem_up_rate = config_settings['mem_up_rate']
        # hyper param gen model
        self.use_learnable_lr = config_settings['use_learnable_lr']
        self.use_learnable_wd = config_settings['use_learnable_wd']
        self.init_lr = config_settings['init_lr']
        self.init_wd = config_settings['init_wd']
        # task encoder
        self.feat_num = self.u_in_dim+self.i_in_dim
        self.max_shot = config_settings['max_shot']
        # 
        self.n_epoch = config_settings['n_epoch']
        self.use_tmem = config_settings['use_tmem']
        self.clusters_k = config_settings['clusters_k']
        self.temperature = config_settings['temperature']

        if self.dataset == 'movielens':
            self.user_emb_layer = User_emb(self.emb_size, config=emb_info)
            self.item_emb_layer = Item_emb(self.emb_size, config=emb_info)
        elif self.dataset == 'bookcrossing':
            self.user_emb_layer = BK_User_emb(self.emb_size, config=emb_info)
            self.item_emb_layer = BK_Item_emb(self.emb_size, config=emb_info)
        else:
            self.user_emb_layer = DB_User_emb(self.emb_size, config=emb_info)
            self.item_emb_layer = DB_Item_emb(self.emb_size, config=emb_info)
        
        self.decoder = Decoder(self.x_dim, self.n_y)
        self.base_model = RecModel(self.user_emb_layer, self.item_emb_layer, self.decoder)
        
        if not config_settings['use_tmem'] and not config_settings['use_mean_task']:
            self.task_encoder = TaskEncoder(self.task_dim, self.num_class, self.emb_size, self.feat_num, self.max_shot, self.device)
        elif not config_settings['use_tmem'] and config_settings['use_mean_task']:
            self.task_encoder = TaskMeanEncoder(self.task_dim, self.num_class, self.emb_size, self.feat_num, self.max_shot, self.device)
        elif config_settings['use_tmem'] and not config_settings['use_mean_task']:
            self.task_encoder = TaskMemEncoder(self.task_dim, self.num_class, self.emb_size, self.feat_num, self.max_shot, self.clusters_k, self.temperature, self.device)
        else:
            self.task_encoder = TaskMeanMemEncoder(self.task_dim, self.num_class, self.emb_size, self.feat_num, self.max_shot, self.clusters_k, self.temperature, self.device)
        self.task_adaptive_optimizer = TaskAdaptiveHyperParameterOptimizer(init_lr=self.init_lr,
                                                            init_wd=self.init_wd,
                                                            n_inner_loop=self.inner_loop_steps,
                                                            use_learnable_lr=self.use_learnable_lr,
                                                            use_learnable_wd=self.use_learnable_wd,
                                                            task_dim=self.task_dim,
                                                            config_settings=config_settings)

    def task_adaptive_optimizer_init(self, names_weight_copy):
        self.task_adaptive_optimizer.alpha_beta_initialise(names_weight_copy)
        self.task_adaptive_optimizer.to(self.device)

    def get_decoder_params_with_grads(self):
        return self.base_model.decoder.get_param_with_grads()

    def get_task_embedding(self, sup_x1, sup_x2, sup_y):
        x_emb = self.base_model.get_embedding(sup_x1, sup_x2)
        task_emb = self.task_encoder(x_emb, sup_y.int())
        return task_emb

    # forward
    def forward(self, x1, x2, weights_params=None):
        return self.base_model(x1, x2)

# deep model
class DeepModel(nn.Module):
    def __init__(self, dataset, emb_info, dataset_info, config_settings) -> None:
        super().__init__()
        # dataset info
        self.dataset = dataset
        self.n_y = dataset_info[self.dataset]['n_y']
        self.u_in_dim = dataset_info[self.dataset]['u_in_dim']
        self.i_in_dim = dataset_info[self.dataset]['i_in_dim']
        # base model
        self.emb_size = config_settings['emb_size']
        # self.x_dim = (self.u_in_dim+self.i_in_dim) * self.emb_size
        self.task_dim = config_settings['task_dim']
        # self.h_dims = config_settings['h_dims']
        # self.dropout_rate = config_settings['dropout_rate']
        self.device = config_settings['device']
        self.inner_loop_steps = config_settings['inner_loop_steps']
        # mem
        # self.mem_cluster_k = config_settings['mem_cluster_k']
        # self.mem_up_rate = config_settings['mem_up_rate']
        # hyper param gen model
        self.use_learnable_lr = config_settings['use_learnable_lr']
        self.use_learnable_wd = config_settings['use_learnable_wd']
        self.init_lr = config_settings['init_lr']
        self.init_wd = config_settings['init_wd']
        # task encoder
        self.feat_num = self.u_in_dim+self.i_in_dim
        self.max_shot = config_settings['max_shot']
        # 
        self.n_epoch = config_settings['n_epoch']
        # self.batch_size = config_settings['batch_size']


        if self.dataset == 'movielens':
            self.user_emb_layer = User_emb(self.emb_size, config=emb_info)
            self.item_emb_layer = Item_emb(self.emb_size, config=emb_info)
        elif self.dataset == 'bookcrossing':
            self.user_emb_layer = BK_User_emb(self.emb_size, config=emb_info)
            self.item_emb_layer = BK_Item_emb(self.emb_size, config=emb_info)
        else:
            self.user_emb_layer = DB_User_emb(self.emb_size, config=emb_info)
            self.item_emb_layer = DB_Item_emb(self.emb_size, config=emb_info)
            
        # self.user_deep_layer = User_deep(self.u_in_dim, self.emb_size)
        # self.item_deep_layer = Item_deep(self.i_in_dim, self.emb_size)
        self.decoder = Decoder(self.emb_size*(self.u_in_dim+self.i_in_dim), self.n_y)
        self.base_model = RecModel(self.user_emb_layer, self.item_emb_layer, self.decoder)
    
    def get_decoder_params_with_grads(self):
        return self.base_model.decoder.get_param_with_grads()

    def get_task_embedding(self, sup_x1, sup_x2, sup_y):
        x_emb = self.base_model.get_embedding(sup_x1, sup_x2)
        task_emb = self.task_encoder(x_emb, sup_y)
        return task_emb

    # forward
    def forward(self, x1, x2):
        return self.base_model(x1, x2)
