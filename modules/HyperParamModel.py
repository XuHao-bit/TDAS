from collections import UserList
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import sys 
sys.path.append("..") 
# from utils import config_settings

class TaskAdaptiveHyperParameterOptimizer(nn.Module):
    def __init__(self, init_lr, init_wd, n_inner_loop, use_learnable_lr, use_learnable_wd, task_dim, config_settings) -> None:
        super().__init__()
        self.init_lr_val = init_lr
        self.init_wd_val = init_wd
        self.n_inner_loop = n_inner_loop
        self.use_learnable_lr = use_learnable_lr
        self.use_learnable_wd = use_learnable_wd
        self.task_dim = task_dim

        # per-step per-layer meta-learnable weight decay and learning rate
        self.names_alpha_dict = nn.ParameterDict()
        self.names_beta_dict = nn.ParameterDict()

        # config
        self.use_ginfo_only = config_settings['use_ginfo_only']
        self.use_tinfo_only = config_settings['use_tinfo_only']
        self.pstep_only = config_settings['pstep_only']
        self.player_only = config_settings['player_only']
        self.n_train_log = config_settings['n_train_log'] # 第几次train
        self.use_tmem = config_settings['use_tmem']
        # adaptive hyper-params generator
        # self.tahp_generator = nn.Sequential()
        self.adap_info = {'task':[],'step':[],'layer':[],'lr':[],'wd':[]}

    def alpha_beta_initialise(self, names_weights_dict):
        self.layer_num = len(names_weights_dict)
        if self.use_tmem:
            self.task_dim = self.task_dim * 2
        
        if self.use_ginfo_only:
            self.input_dim = 2*self.layer_num+1 # loss, per-layer-avg-weight, per-layer-avg-grad
        elif self.use_tinfo_only:
            self.input_dim = self.task_dim# task_diff_emb(mean,var,N-c,sim)
        else:
            self.input_dim = self.task_dim+1+2*self.layer_num # task_diff_emb(mean,var,N-c,sim), loss, per-layer-avg-weight, per-layer-avg-grad
        
        if self.pstep_only:
            self.output_dim = 2
        elif self.player_only:
            self.output_dim = 2*self.layer_num
        else:
            self.output_dim = 2*self.layer_num
        
        # hyper parameter generator
        self.hyper_param_generator = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.output_dim),
            nn.Sigmoid()
            )

        # hyper paramter meta-learner
        if self.pstep_only:
            step_size = self.n_inner_loop
        elif self.player_only:
            step_size = 0
        else:
            step_size = self.n_inner_loop
        
        # names_weights_dict 将model的所有参数按照原始的key-value存储；key: layer.weight or layer.bias
        for key in names_weights_dict.keys():
            # init per-step per-layer meta-learnable learning rate term
            self.names_alpha_dict[key.replace('.','-')] = nn.Parameter(
                data=torch.full([step_size+1], self.init_lr_val),
                requires_grad=self.use_learnable_lr
            )
            # init per-step per-layer meta-learnable weight decay bias term
            self.names_beta_dict[key.replace('.','-')] = nn.Parameter(
                data=torch.full([step_size+1], self.init_lr_val*self.init_wd_val),
                requires_grad=self.use_learnable_wd
            )
    
    def gen_hyper_params(self, task_diff_emb, task_loss, task_model_info):
        if self.use_ginfo_only:
            task_adaptive_input = torch.cat([task_loss, task_model_info])
        elif self.use_tinfo_only:
            task_adaptive_input = task_diff_emb
        else:
            task_adaptive_input = torch.cat([task_diff_emb, task_loss, task_model_info])
        
        gen_hyper_params = self.hyper_param_generator(task_adaptive_input)
        gen_alpha, gen_beta = torch.split(gen_hyper_params, split_size_or_sections=int(self.output_dim/2))
        if self.pstep_only:
            gen_alpha = gen_alpha.repeat(self.layer_num)
            gen_beta = gen_beta.repeat(self.layer_num)
        # print(gen_alpha.shape)
        return gen_alpha, gen_beta

    # def update_params2(self, names_weights_dict, names_grads_dict, gen_alpha_dict, gen_beta_dict, num_step):
    #     """
    #     update the names_weights_dict and return a new dict
    #     """
    #     # gen_alpha and beta are per-layer params
    #     updated_names_weight_dict = dict()
    #     for key in names_grads_dict.keys():
    #         adaptive_wd_bias = gen_beta_dict[key] * self.names_beta_dict[key.replace('.','-')][num_step]
    #         adaptive_lr = gen_alpha_dict[key] * self.names_alpha_dict[key.replace('.','-')][num_step]
    #         updated_names_weight_dict[key] = (1 - adaptive_wd_bias) * names_weights_dict[key] - adaptive_lr * names_grads_dict[key]
    #     return updated_names_weight_dict
    
    # def update_params(self, model, names_grads_dict, gen_alpha_dict, gen_beta_dict, num_step):
    #     """
    #     update the names_weights_dict and return a new dict
    #     """
    #     # gen_alpha and beta are per-layer params
    #     # updated_names_weight_dict = dict()
    #     for param, key in zip(model.parameters(), names_grads_dict.keys()):
    #         # beta = 
    #         adaptive_wd_bias = gen_beta_dict[key] * self.names_beta_dict[key.replace('.','-')][num_step]
    #         adaptive_lr = gen_alpha_dict[key] * self.names_alpha_dict[key.replace('.','-')][num_step]
    #         param.data = (1 - adaptive_wd_bias) * param.data - adaptive_lr * names_grads_dict[key]

        # return updated_names_weight_dict

    def update_params3(self, model, names_grads_dict, gen_alpha_dict, gen_beta_dict, num_step, writer_info=None):
        """
        只使用model来更新
        """
        # gen_alpha and beta are per-layer params
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                key = name + '.weight'
                adaptive_wd_bias = gen_beta_dict[key] * self.names_beta_dict[key.replace('.','-')][num_step]
                adaptive_lr = gen_alpha_dict[key] * self.names_alpha_dict[key.replace('.','-')][num_step]
                new_param = (1 - adaptive_wd_bias) * module._parameters['weight'] - adaptive_lr * names_grads_dict[key]
                self.write_board(writer_info, key, adaptive_lr, adaptive_wd_bias, num_step)
                del module._parameters['weight']
                module._parameters['weight'] = new_param
                # module._parameters['weight'] = torch.tensor(new_param, requires_grad=True)
            elif isinstance(module, torch.nn.Linear):
                w_key = name + '.weight'
                adaptive_wd_bias = gen_beta_dict[w_key] * self.names_beta_dict[w_key.replace('.','-')][num_step]
                adaptive_lr = gen_alpha_dict[w_key] * self.names_alpha_dict[w_key.replace('.','-')][num_step]
                new_param_w = (1 - adaptive_wd_bias) * module._parameters['weight'] - adaptive_lr * names_grads_dict[w_key]
                self.write_board(writer_info, w_key, adaptive_lr, adaptive_wd_bias, num_step)
                del module._parameters['weight']
                module._parameters['weight'] = new_param_w
                # module._parameters['weight'] = torch.tensor(new_param_w, requires_grad=True)

                if module._parameters['bias'] is not None:
                    b_key = name + '.bias'
                    adaptive_wd_bias = gen_beta_dict[b_key] * self.names_beta_dict[b_key.replace('.','-')][num_step]
                    adaptive_lr = gen_alpha_dict[b_key] * self.names_alpha_dict[b_key.replace('.','-')][num_step]
                    new_param_b = (1 - adaptive_wd_bias) * module._parameters['bias'] - adaptive_lr * names_grads_dict[b_key]
                    self.write_board(writer_info, b_key, adaptive_lr, adaptive_wd_bias, num_step)
                    del module._parameters['bias']
                    module._parameters['bias'] = new_param_b
                    # module._parameters['bias'] = torch.tensor(new_param_b, requires_grad=True)


    def update_params4(self, model, local_model, names_grads_dict, gen_alpha_dict, gen_beta_dict, num_step, writer_info=None):
        """
        update the names_weights_dict and return a new dict
        用model的参数计算得到更新后的参数new_param
        删除local_module的weight和bias, 并替换成new_param, 此时local_model变为non-leaf node, 但是model仍然为leaf
        """
        # gen_alpha and beta are per-layer params
        for (name, module), l_module in zip(model.named_modules(), local_model.modules()):
            if isinstance(module, torch.nn.Embedding):
                key = name + '.weight'
                adaptive_wd_bias = gen_beta_dict[key] * self.names_beta_dict[key.replace('.','-')][num_step]
                adaptive_lr = gen_alpha_dict[key] * self.names_alpha_dict[key.replace('.','-')][num_step]
                new_param = (1 - adaptive_wd_bias) * module._parameters['weight'] - adaptive_lr * names_grads_dict[key]
                self.write_board(writer_info, key, adaptive_lr, adaptive_wd_bias, num_step)
                del l_module._parameters['weight']
                l_module._parameters['weight'] = new_param
                # module._parameters['weight'] = torch.tensor(new_param, requires_grad=True)
            elif isinstance(module, torch.nn.Linear):
                w_key = name + '.weight'
                adaptive_wd_bias = gen_beta_dict[w_key] * self.names_beta_dict[w_key.replace('.','-')][num_step]
                adaptive_lr = gen_alpha_dict[w_key] * self.names_alpha_dict[w_key.replace('.','-')][num_step]
                new_param_w = (1 - adaptive_wd_bias) * module._parameters['weight'] - adaptive_lr * names_grads_dict[w_key]
                self.write_board(writer_info, w_key, adaptive_lr, adaptive_wd_bias, num_step)
                del l_module._parameters['weight']
                l_module._parameters['weight'] = new_param_w
                # module._parameters['weight'] = torch.tensor(new_param_w, requires_grad=True)

                if module._parameters['bias'] is not None:
                    b_key = name + '.bias'
                    adaptive_wd_bias = gen_beta_dict[b_key] * self.names_beta_dict[b_key.replace('.','-')][num_step]
                    adaptive_lr = gen_alpha_dict[b_key] * self.names_alpha_dict[b_key.replace('.','-')][num_step]
                    new_param_b = (1 - adaptive_wd_bias) * module._parameters['bias'] - adaptive_lr * names_grads_dict[b_key]
                    self.write_board(writer_info, b_key, adaptive_lr, adaptive_wd_bias, num_step)
                    del l_module._parameters['bias']
                    l_module._parameters['bias'] = new_param_b
                    # module._parameters['bias'] = torch.tensor(new_param_b, requires_grad=True)
    
    # ==== Writer ====
    def write_board(self, writer_info, key, lr, wd, step):
        # print(lr)
        if writer_info:
            # writer = writer_info['writer']
            task = writer_info['task']
            # epoch = writer_info.get('epoch', 'e')
            # stage = writer_info.get('stage', 'train')
            # writer.add_scalars(f'{self.n_train_log}/task{task}_{stage}_lr', {f'{key}_e{epoch}':lr}, step)
            # writer.add_scalars(f'{self.n_train_log}/task{task}_{stage}_wd', {f'{key}_e{epoch}':wd}, step)
            # print(f'{task},{step},{key},{lr},{wd}')
            # self.adap_info = {'task':[],'step':[],'layer':[],'lr':[],'wd':[]}
            self.adap_info['task'].append(task)
            self.adap_info['step'].append(step)
            self.adap_info['layer'].append(key)
            self.adap_info['lr'].append(lr.item())
            self.adap_info['wd'].append(wd.item())
