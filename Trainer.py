# from pickle import NEXT_BUFFER
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from modules.MyOptim import MyAdam
import pandas as pd

from utils import *
from DataLoading import *
# from torch.utils.tensorboard import SummaryWriter



# 优化器，参考few_shot_learning_system.py的一部分（原来的代码不太好懂, 以及TaNP的Trainer
class Modeling:
    def __init__(self, model, config_settings, mode='Meta') -> None:
        self.device = config_settings['device']
        self.n_inner_loop = config_settings['inner_loop_steps']
        self.n_epoch = config_settings['n_epoch']
        self.meta_lr = config_settings['meta_lr'] # global lr
        self.local_lr = config_settings['local_lr'] # local lr for melu
        # self.tdmeta_lr = config_settings['tdmeta_lr']
        self.min_lr = config_settings['min_lr']
        self.meta_wd = config_settings['meta_wd'] # global wd
        # self.writer = SummaryWriter(config_settings['writer_log'])
        self.train_writer = SummaryWriter(config_settings['train_log'])
        self.use_writer = config_settings['use_writer']
        self.use_gen_hypr = config_settings['use_gen_hypr']
        self.use_grad_clip = config_settings['use_grad_clip']
        self.pstep_only = config_settings['pstep_only']
        self.player_only = config_settings['player_only']
        self.clip_norm = config_settings['clip_norm']
        self.model = model.to(self.device)
        self.local_model = deepcopy(self.model.base_model)
        self.model_name = config_settings['model_name']
        self.early_stop = EarlyStopping(5, path=f'./saved_models/{self.model_name}_cp.pth')
        
        # ablation
        self.ada_lr_only = config_settings['ada_lr_only']
        self.ada_wd_only = config_settings['ada_wd_only']


        mode = mode.lower()
        # task adaptive optimizer init
        if mode == 'tdmeta':
            phi_copy = self.model.base_model.state_dict()
            self.model.task_adaptive_optimizer_init(phi_copy)

            self.optimizer = torch.optim.Adam([
                {'params': self.model.task_encoder.parameters()},
                {'params': self.model.task_adaptive_optimizer.parameters()},
            ], lr=self.meta_lr, amsgrad=False, weight_decay=self.meta_wd)

            # global phi optimizer
            self.global_optimizer = MyAdam(phi_copy, weight_decay=self.meta_wd)

        elif mode == 'melu':
            # local update optimizer should be sgd
            # melu model didn't use weight decay item when local updating
            self.local_optimizer = torch.optim.SGD(self.model.decoder.parameters(), lr=self.local_lr)
            # self.local_optimizer_adam = torch.optim.Adam(self.model.decoder.parameters(), lr=self.local_lr)
            # when global updating, it doesn't use weight decay too.
            self.global_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        else:
            self.local_optimizer = torch.optim.SGD(self.model.decoder.parameters(), lr=self.local_lr)
            self.global_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)

        # update rules
        self.loss_fn = nn.MSELoss()
        
        # self.local_optim = torch.optim.Adam([
        #     {'params': self.model.parameters()},
        # ], lr=self.meta_lr, amsgrad=False)

    # [local update] ===================================
    def local_update(self, sup_x1, sup_x2, sup_y, optimizer, theta=None):
        # load theta for meta learning local update
        if theta:
            self.model.decoder.load_state_dict(theta)
        
        for i_loop in range(self.n_inner_loop):
            # applying task adaptive local update
            y_hat = self.model(sup_x1, sup_x2)
            # print(y_hat, sup_y)
            sup_loss = self.loss_fn(y_hat, sup_y.view(-1, 1))
            optimizer.zero_grad()
            sup_loss.backward()
            optimizer.step()

    def tdmeta_local_update(self, sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y, task_emb, writer_info=None):
        # with torch.autograd.set_detect_anomaly(True):
        model_keys = self.model.base_model.state_dict().keys()
        task_losses = []
        for i_loop in range(self.n_inner_loop):
            if i_loop == 0:
                this_model = self.model.base_model
            else:
                this_model = self.local_model
        
            # applying task adaptive local update
            # self.model.base_model.load_state_dict(theta)
            y_hat = this_model(sup_x1, sup_x2)
            # print(y_hat, sup_y)
            sup_loss = self.loss_fn(y_hat, sup_y.view(-1, 1))
            sup_grad = torch.autograd.grad(sup_loss, this_model.parameters(), retain_graph=True)            
            
            # sup_grad_dict = dict(zip(model_keys, sup_grad))
            # for key, grad in sup_grad_dict.items():
            #     if grad is None:
            #         print('Grads not found for inner loop parameter', key)
            #     sup_grad_dict[key] = sup_grad_dict[key].sum(dim=0)

            # grad clip here.
            if self.use_grad_clip:
                apply_grad_clip_norm(sup_grad, max_norm=self.clip_norm)
            
            sup_grad_dict = dict(zip(model_keys, sup_grad))
            for key, grad in sup_grad_dict.items():
                if grad is None:
                    print('Grads not found for inner loop parameter', key)
                sup_grad_dict[key] = sup_grad_dict[key].sum(dim=0)

            # get generated hyper parameters
            gen_alpha_dict, gen_beta_dict = {}, {}
            # If generated hyper parameters are not used, the rec_model will be MeLU
            if self.use_gen_hypr:
                # get task info by theta and grad
                per_step_task_info = []
                for v in this_model.state_dict().values(): # self.model.base_model.state_dict().values()
                    per_step_task_info.append(v.mean())
                for grad in sup_grad_dict.values():
                    per_step_task_info.append(grad.mean())
                per_step_task_info = torch.stack(per_step_task_info)
                
                # generate per-layer adaptive hyper-params by task_diff and task_info
                gen_alpha, gen_beta = self.model.task_adaptive_optimizer.gen_hyper_params(task_emb, sup_loss.reshape(1),\
                                                                                        per_step_task_info)
            
                # print(gen_alpha.shape)
                # make the generated params have the same key as theta
                for idx, key in enumerate(model_keys): # self.model.base_model.state_dict().keys()
                    # ablation
                    if self.ada_lr_only:
                        gen_alpha_dict[key] = gen_alpha[idx]
                        gen_beta_dict[key] = 1
                    elif self.ada_wd_only:
                        gen_alpha_dict[key] = 1
                        gen_beta_dict[key] = gen_beta[idx]
                    else:
                        gen_alpha_dict[key] = gen_alpha[idx]
                        gen_beta_dict[key] = gen_beta[idx]
            else:
                for idx, key in enumerate(model_keys):
                    gen_alpha_dict[key] = 1
                    gen_beta_dict[key] = 1

            if i_loop == 0:
                # 第1轮, local_model经过model计算得来, local_model变为non-leaf
                self.model.task_adaptive_optimizer.update_params4(model=self.model.base_model, # self.model.base_model.state_dict().values()
                                                            local_model=self.local_model,
                                                            names_grads_dict=sup_grad_dict,
                                                            gen_alpha_dict=gen_alpha_dict,
                                                            gen_beta_dict=gen_beta_dict,
                                                            num_step=i_loop,
                                                            writer_info=writer_info)
                # if not use per-step adaptive update
                if self.player_only: 
                    self.gen_alpha = gen_alpha_dict
                    self.gen_beta = gen_beta_dict
            else:
                # if not use per-step adaptive update
                if self.player_only: 
                    gen_alpha_dict = self.gen_alpha
                    gen_beta_dict = self.gen_beta
                    this_step = 0
                else:
                    this_step = i_loop
                # 经过update_param之后，is_leaf为false
                self.model.task_adaptive_optimizer.update_params3(model=self.local_model,
                                                                names_grads_dict=sup_grad_dict,
                                                                gen_alpha_dict=gen_alpha_dict,
                                                                gen_beta_dict=gen_beta_dict,
                                                                num_step=this_step,
                                                                writer_info=writer_info)
            # self.model.zero_grad()
            que_yhat = self.local_model(que_x1, que_x2)
            # print(que_yhat, que_y)
            que_loss = self.loss_fn(que_yhat, que_y.view(-1, 1))
            # que_loss.backward(retain_graph=True)
            task_losses.append(que_loss)
            # print('[local] task encoder--')
            # grads = torch.autograd.grad(que_loss, self.model.task_encoder.parameters(), retain_graph=True, allow_unused=True)
            # for n, g in zip(self.model.task_encoder.state_dict().keys(), grads):
            #     print(n, g)
            
        return task_losses
                
    # [model test] =====================================
    def model_test(self, data, optimizer):
        # 用optimizer来控制local update更新的内容
        all_loss = 0
        rmse = []
        mae = []
        ndcg1, ndcg3, ndcg5, ndcg7, ndcg10 = [], [], [], [], []
        recall1, recall3, recall5, recall7, recall10 = [], [], [], [], []
        phi = deepcopy(self.model.state_dict())

        for i in tqdm(range(len(data))):
            # print('before theta:{}'.format(theta['hidden_layer_1.weight']))
            sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
            sup_x1, sup_x2, sup_y, que_x1, que_x2 = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                sup_y.to(self.device), que_x1.to(self.device),\
                                                                que_x2.to(self.device)

            # local update
            self.model.load_state_dict(phi)
            self.local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, optimizer=optimizer)
            
            # que loss
            with torch.no_grad():
                que_y_hat = self.model(que_x1, que_x2).cpu()
                local_que_loss = self.loss_fn(que_y_hat, que_y.view(-1, 1))
                all_loss += local_que_loss

            # print(que_y_hat)
            # que_y_pred = torch.argmax(que_y_hat, dim=1) # range: 0-4
            mae.append(MAE(que_y_hat.view(-1), que_y.cpu()))
            rmse.append(RMSE(que_y_hat.view(-1), que_y.cpu()))
            ndcg3.append(NDCG(que_y_hat.view(-1), que_y.cpu(), 3))
            ndcg5.append(NDCG(que_y_hat.view(-1), que_y.cpu(), 5))

        mmae = sum(mae) / len(mae)
        mrmse = sum(rmse) / len(rmse)
        mndcg3 = sum(ndcg3) / len(ndcg3)
        mndcg5 = sum(ndcg5) / len(ndcg5)
        mloss = all_loss / len(data) 
        
        # print(mae, rmse)
        print(f'mae:{mmae.item()}, rmse:{mrmse.item()}, ndcg3:{mndcg3}, ndcg5:{mndcg5}')
        return mmae, rmse
    
    # [deep learning train] ===================================
    def deep_learning_train(self, data, test_data):
        best_model, best_mae = None, None
        for epoch in range(self.n_epoch):
            print(f'epoch: {epoch}')
            for i in tqdm(range(len(data))):
                sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
                sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                sup_y.to(self.device), que_x1.to(self.device),\
                                                                que_x2.to(self.device), que_y.to(self.device)

                # local update the decoder
                # self.local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, optimizer=self.local_optimizer)
                
                # global update the whole model
                que_yhat = self.model(que_x1, que_x2)
                que_loss = self.loss_fn(que_yhat, que_y.view(-1, 1))
                self.global_optimizer.zero_grad()
                que_loss.backward()
                self.global_optimizer.step()
        
            mmae, mrmse, mndcg3 = self.model_test(test_data, self.global_optimizer)
            if best_mae is None or best_mae > mmae:
                best_model = deepcopy(self.model.state_dict())
                best_mae = mmae
        return best_model
    
    # [melu train] ============================================
    def melu_train(self, data, test_data):
        # phi is the whole model param
        phi = deepcopy(self.model.base_model.state_dict())
        best_model, best_mae = None, None
        for epoch in range(self.n_epoch):
            print(f'epoch: {epoch}')
            for i in tqdm(range(len(data))):
                # init theta: theta<-phi
                self.model.base_model.load_state_dict(phi)
                sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
                # print(data[i])
                sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                sup_y.to(self.device), que_x1.to(self.device),\
                                                                que_x2.to(self.device), que_y.to(self.device)

                # local update on sup set, only update decoder model
                # self.local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, optimizer=self.local_optimizer_adam)
                self.local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, optimizer=self.local_optimizer)
                # self.local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, optimizer=self.global_optimizer) # 检查local updata 更新embedding会不会有效果

                # global update, update whole phi(emb, emb_deep, decoder)
                que_yhat = self.model(que_x1, que_x2)
                que_loss = self.loss_fn(que_yhat, que_y.view(-1, 1))
                self.model.zero_grad()
                grads = torch.autograd.grad(que_loss, self.model.parameters(), retain_graph=False)
                apply_grad_clip_norm(grads, max_norm=self.clip_norm)
                # global update phi
                for k, grad in zip(phi.keys(), grads):
                    phi[k] = phi[k] - self.meta_lr * grad
                self.model.base_model.load_state_dict(phi)

            # test every 2 epoches
            if epoch % 2 == 0:
                mmae, mrmse = self.model_test(test_data, self.local_optimizer) # local update only update decoder
                if best_mae is None or best_mae > mmae:
                    best_model = deepcopy(self.model.state_dict())
                    best_mae = mmae

        return best_model

    # [meta learning train & test] ===================================
    def train(self, data, test_data):
        # init phi
        phi = deepcopy(self.model.base_model.state_dict())
        best_model, best_mae = None, None
        for epoch in range(self.n_epoch):
            print(f'epoch: {epoch}')
            all_loss = 0
            for i in tqdm(range(len(data))):
                # init theta: theta <- phi
                # theta is the local param (which means the model's param)
                if i > 0:
                    self.model.base_model.load_state_dict(phi)
                # self.local_model.load_state_dict(phi)
                sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
                sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                sup_y.to(self.device), que_x1.to(self.device),\
                                                                que_x2.to(self.device), que_y.to(self.device)

                # no.1 get per-step task embedding.
                task_emb = self.model.get_task_embedding(sup_x1, sup_x2, sup_y)

                # no.2 local loop
                writer_info = None
                if self.use_writer and epoch==2:
                    writer_info = {'task':i, 'writer':None, 'stage':'train', 'epoch':epoch}
                que_losses = self.tdmeta_local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, 
                                        que_x1=que_x1, que_x2=que_x2, que_y=que_y, task_emb=task_emb, writer_info=writer_info)

                # no.3 que_set loss
                que_loss = torch.sum(torch.stack(que_losses))
                self.local_model.zero_grad()
                phi_grads = torch.autograd.grad(que_losses[-1], self.local_model.parameters(), retain_graph=True, allow_unused=True)
                # phi_grads = torch.autograd.grad(que_loss, self.model.base_model.parameters(), retain_graph=True, allow_unused=True)
                all_loss += torch.mean(torch.stack(que_losses)).item()

                # grad clip here.
                if self.use_grad_clip:
                    apply_grad_clip_norm(phi_grads, max_norm=self.clip_norm)

                # update phi
                # cur_lr = self.scheduler.get_lr()[0]
                # this_lr = self.meta_lr*cur_lr/self.tdmeta_lr
                # this_lr = cur_lr
                self.global_optimizer.step(phi, phi_grads, self.meta_lr)

                # 可能是这里的grad过大，导致hypr_param_gen的参数过大，生成的超参就大；
                # for i, phi_grad in zip(phi.keys(), phi_grads):
                #     phi[i] = (1-self.meta_wd*self.meta_lr)*phi[i] - self.meta_lr * phi_grad
                # for i, phi_grad in zip(phi.keys(), phi_grads):
                #     phi[i] = phi[i] - self.meta_lr * phi_grad
                
                # update task_adaptive_optim, task_encoder
                self.optimizer.zero_grad()
                que_loss.backward() # task_encoder, task_adaptive_optimizer
                # grad clip here.
                if self.use_grad_clip:
                    nn.utils.clip_grad_norm_(self.model.task_encoder.parameters(), max_norm=self.clip_norm)
                    nn.utils.clip_grad_norm_(self.model.task_adaptive_optimizer.parameters(), max_norm=self.clip_norm)
                self.optimizer.step()
            
                if i == len(data) - 1:
                    self.model.base_model.load_state_dict(phi)

            # self.scheduler.step()
            # print(f'cur_lr:{cur_lr}')
            # self.global_optimizer.epoch_step()
            train_loss = all_loss / len(data)
            mmae, mrmse, val_loss = self.test(test_data, epoch)
            if writer_info:
                df = pd.DataFrame(self.model.task_adaptive_optimizer.adap_info)
                df.to_csv('/home/zhaoxuhao/TDMeta/res/paper_figures/ada_hp_train_e2_i5.csv', sep=',', index=False, header=True)
            
            # write loss
            self.train_writer.add_scalar(f'{config_settings["n_train_log"]}/train_loss', train_loss, epoch)
            self.train_writer.add_scalar(f'{config_settings["n_train_log"]}/val_loss', val_loss, epoch)

            self.early_stop(val_loss=mmae, model=self.model)
            if self.early_stop.early_stop:
                print('Early_stop!')
                break

        return best_model

    def test(self, data, tr_epoch=0):
        all_loss = 0
        rmse = []
        mae = []
        ndcg3, ndcg5 = [], []
        # phi = deepcopy(self.model.base_model.state_dict())

        for i in tqdm(range(len(data))):
            # # theta = deepcopy(phi)
            # self.model.base_model.load_state_dict(theta)
            sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = data[i]
            sup_x1, sup_x2, sup_y, que_x1, que_x2, que_y = sup_x1.to(self.device), sup_x2.to(self.device),\
                                                                sup_y.to(self.device), que_x1.to(self.device),\
                                                                que_x2.to(self.device), que_y.to(self.device)

            task_emb = self.model.get_task_embedding(sup_x1, sup_x2, sup_y)

            writer_info = None
            # if self.use_writer and tr_epoch==2:
            #     writer_info = {'task':i, 'writer':None, 'stage':'test', 'epoch':tr_epoch}

            # local update
            que_losses = self.tdmeta_local_update(sup_x1=sup_x1, sup_x2=sup_x2, sup_y=sup_y, 
                                        que_x1=que_x1, que_x2=que_x2, que_y=que_y, task_emb=task_emb, writer_info=writer_info)
        
            
            # print('after theta:{}'.format(theta['hidden_layer_1.weight']))
            # que loss
            with torch.no_grad():
                que_y_hat = self.local_model(que_x1, que_x2).cpu()
                all_loss += torch.mean(torch.stack(que_losses)).item()
                # print(que_y_hat)

            # que_y_pred = torch.argmax(que_y_hat, dim=1) # range: 0-4
            mae.append(MAE(que_y_hat.view(-1), que_y.cpu()))
            rmse.append(RMSE(que_y_hat.view(-1), que_y.cpu()))
            ndcg3.append(NDCG(que_y_hat.view(-1), que_y.cpu(), 3))
            ndcg5.append(NDCG(que_y_hat.view(-1), que_y.cpu(), 5))
        
        # visual experiment
        if writer_info:
            df = pd.DataFrame(self.model.task_adaptive_optimizer.adap_info)
            df.to_csv('/home/zhaoxuhao/TDMeta/res/visual/ada_hp_test.csv', sep=',', index=False, header=True)

        mmae = sum(mae) / len(mae)
        mrmse = sum(rmse) / len(rmse)
        mndcg3 = sum(ndcg3) / len(ndcg3)
        mndcg5 = sum(ndcg5) / len(ndcg5)
        mloss = all_loss / len(data) 
        
        # print(mae, rmse)
        print(f'mae:{mmae.item()}, rmse:{mrmse.item()}, ndcg3:{mndcg3}, ndcg5:{mndcg5}')
        return mmae, rmse, mloss


        

