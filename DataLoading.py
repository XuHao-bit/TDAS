import pickle
from utils import config_settings, config
import os
import random
import numpy as np
import torch

def to_torch(in_list):
    return torch.from_numpy(np.array(in_list))

def get_train_test_user_set(dataset='movielens'):
    path_store = 'data_processed/' + dataset + '/'
    train_ = 'train_'+config_settings['data_split']+'.p'
    test_ = 'test_'+config_settings['data_split']+'.p'
    # val_ = 'val_'+config_settings['data_split']+'.p'
    if not os.path.exists(path_store + train_):
        train_user_set, test_user_set = train_test_user_list(dataset, rand=True, store=True)
    else:
        train_user_set = pickle.load(open(path_store + train_, 'rb'))
        test_user_set = pickle.load(open(path_store + test_, 'rb'))
        # val_user_set = pickle.load(open(path_store + val_, 'rb'))
    return train_user_set, test_user_set

def train_test_user_list(dataset='movielens', rand=True, random_state=32, train_ratio=0.7, val_ratio=0.1, store=False):
    path = 'data_processed/' + dataset + '/raw/'
    path_store = 'data_processed/' + dataset + '/'
    len_user = int(len(os.listdir(path)) / 3)  # movielens 4836
    training_size = int(len_user * (val_ratio+train_ratio))
    # val_size = int(len_user * )
    user_id_list = list(range(1, len_user))
    if rand:
        random.shuffle(user_id_list)
    else:
        random.seed(random_state)
        random.shuffle(user_id_list)

    train_user_set, test_user_set = user_id_list[:training_size], user_id_list[training_size:]
    if store:
        train_ = 'train_'+config_settings['data_split']+'.p'
        test_ = 'test_'+config_settings['data_split']+'.p'
        val_ = 'val_'+config_settings['data_split']+'.p'
        pickle.dump(train_user_set, open(path_store + train_, 'wb'))
        pickle.dump(test_user_set, open(path_store + test_, 'wb'))
        # pickle.dump(val_user_set, open(path_store + val_, 'wb'))
    return train_user_set, test_user_set


def load_dataset(user_set, dataset='movielens', support_size=16, query_size=4):
    # pickle 中是np array类型
    sup_x1_list, sup_x2_list, sup_y_list, que_x1_list, que_x2_list, que_y_list = [], [], [], [], [], []
    path = 'data_processed/' + dataset + '/raw/'
    
    for user_id in user_set:
        u_x1 = pickle.load(open('{}sample_{}_x1.p'.format(path, str(user_id)), 'rb'))
        u_x2 = pickle.load(open('{}sample_{}_x2.p'.format(path, str(user_id)), 'rb'))
        u_y = pickle.load(open('{}sample_{}_y.p'.format(path, str(user_id)), 'rb'))
        
        u_x1 = np.tile(u_x1, (len(u_x2), 1))
        # if dataset == 'movielens':
        #     u_y = u_y - 1
        
        sup_x1, que_x1 = to_torch(u_x1[:support_size]), to_torch(u_x1[support_size:support_size+query_size])
        sup_x2, que_x2 = to_torch(u_x2[:support_size]), to_torch(u_x2[support_size:support_size+query_size])
        sup_y, que_y = to_torch(u_y[:support_size]).float(), to_torch(u_y[support_size:support_size+query_size]).float()
        sup_y, que_y = sup_y-1, que_y-1

        sup_x1_list.append(sup_x1)
        sup_x2_list.append(sup_x2)
        sup_y_list.append(sup_y)
        que_x1_list.append(que_x1)
        que_x2_list.append(que_x2)
        que_y_list.append(que_y)
    
    dataset_data = list(zip(sup_x1_list, sup_x2_list, sup_y_list, que_x1_list, que_x2_list, que_y_list))
    del(sup_x1_list, sup_x2_list, sup_y_list, que_x1_list, que_x2_list, que_y_list)
    
    return dataset_data

def fix_load_dataset(user_set, dataset='movielens', support_size=16, query_size=4):
    # pickle 中是np array类型
    sup_x1_list, sup_x2_list, sup_y_list, que_x1_list, que_x2_list, que_y_list = [], [], [], [], [], []
    path = 'data_processed/' + dataset + '/raw/'
    fix_sup_size = 15
    for user_id in user_set:
        u_x1 = pickle.load(open('{}sample_{}_x1.p'.format(path, str(user_id)), 'rb'))
        u_x2 = pickle.load(open('{}sample_{}_x2.p'.format(path, str(user_id)), 'rb'))
        u_y = pickle.load(open('{}sample_{}_y.p'.format(path, str(user_id)), 'rb'))
        
        u_x1 = np.tile(u_x1, (len(u_x2), 1))
        # if dataset == 'movielens':
        #     u_y = u_y - 1
        
        sup_x1, que_x1 = to_torch(u_x1[:support_size]), to_torch(u_x1[fix_sup_size:fix_sup_size+query_size])
        sup_x2, que_x2 = to_torch(u_x2[:support_size]), to_torch(u_x2[fix_sup_size:fix_sup_size+query_size])
        sup_y, que_y = to_torch(u_y[:support_size]).float(), to_torch(u_y[fix_sup_size:fix_sup_size+query_size]).float()
        sup_y, que_y = sup_y-1, que_y-1

        sup_x1_list.append(sup_x1)
        sup_x2_list.append(sup_x2)
        sup_y_list.append(sup_y)
        que_x1_list.append(que_x1)
        que_x2_list.append(que_x2)
        que_y_list.append(que_y)
    
    dataset_data = list(zip(sup_x1_list, sup_x2_list, sup_y_list, que_x1_list, que_x2_list, que_y_list))
    del(sup_x1_list, sup_x2_list, sup_y_list, que_x1_list, que_x2_list, que_y_list)
    
    return dataset_data
