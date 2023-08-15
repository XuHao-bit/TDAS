import argparse
import datetime
import os

from TaskDiffModel import TaskDiffModel, DeepModel
from Trainer import Modeling
from DataLoading import *
from utils import config, dataset_info, config_settings, seed_all

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bookcrossing')
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--mode', type=str, default='Deep')
parser.add_argument('--save', type=str, default='False')
# parser.add_argument('--use_writer', type=str, default='False')
parser.add_argument('--save_name', type=str, default=f'model_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
parser.add_argument('--epoch', type=int, default=35)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--use_gen_hypr', type=str, default='True')
parser.add_argument('--use_lea_hypr', type=str, default='True')
parser.add_argument('--use_ginfo_only', type=str, default='False')   # 代表generating adaptive hyper-parameters的时候，仅使用avg grad 以及avg weight信息；
parser.add_argument('--use_tinfo_only', type=str, default='False')   # 代表generating adaptive hyper-parameters的时候，仅使用task emb信息；
parser.add_argument('--nshot', type=int, default=15)
parser.add_argument('--use_tmem', type=str, default='False')

# === prepare log ===
today_time = datetime.datetime.now().strftime("%Y-%m-%d")
log_dir = f'./log/{today_time}'
os.makedirs(log_dir, exist_ok=True)

# === prepare args === 
args = vars(parser.parse_args())
seed_all(args['seed'])
print(args)
# if args['use_writer'] != 'False':
#     config_settings['use_writer'] = True
#     config_settings['writer_log'] = f"{log_dir}/{args['use_writer']}"
if args['epoch'] > 0 and args['epoch']<200:
    config_settings['n_epoch'] = args['epoch']
if torch.cuda.is_available():
    config_settings['device'] = 'cuda:{}'.format(args['cuda'])
if args['use_gen_hypr'] == 'False':
    config_settings['use_gen_hypr'] = False
if args['use_lea_hypr'] == 'False':
    config_settings['use_learnable_lr'] = False
    config_settings['use_learnable_wd'] = False
if args['use_ginfo_only'] == 'True':
    config_settings['use_ginfo_only'] = True
if args['use_tinfo_only'] == 'True':
    config_settings['use_tinfo_only'] = True
if args['nshot'] > 15:
    args['nshot'] = 15
if args['use_tmem'] == 'True':
    config_settings['use_tmem'] = True

# === prepare data ===
train_user, test_user = get_train_test_user_set(args['dataset'])
train_dataset = fix_load_dataset(train_user,args['dataset'],args['nshot'],25)
test_dataset = fix_load_dataset(test_user,args['dataset'],args['nshot'],25)

# === prepare hyper-param ===
meta_lrs = [0.001] # global update rate, global < local
local_lrs = [0.1] # local update rate
tdmeta_lrs = [0.001]

param_settings = []
for meta_lr in meta_lrs:
    for local_lr in local_lrs:
        for tdmeta_lr in tdmeta_lrs:
            param_settings.append([3, meta_lr, local_lr, tdmeta_lr])

for param in param_settings:
    inner_loop, meta_lr, local_lr, tdmeta_lr = param
    config_settings['inner_loop_steps'] = inner_loop
    config_settings['meta_lr'] = meta_lr
    config_settings['local_lr'] = local_lr
    config_settings['init_lr'] = local_lr # tdmeta 的local lr 参数需要修改init lr
    config_settings['tdmeta_lr'] = tdmeta_lr
    config_settings['min_lr'] = meta_lr
    # log
    config_settings['model_name'] = f'tdmeta+adam_i{inner_loop}_gl{meta_lr}_ll{local_lr}'
    config_settings['train_log'] = f'{log_dir}/{config_settings["model_name"]}' # train and val loss log

    if os.path.exists(config_settings['train_log']):    # 同一个模型，训练的第几次；
        config_settings['n_train_log'] = f"no_{len(os.listdir(config_settings['train_log']))}"

    print(config_settings)
    model = {
        'deep': DeepModel,
        'melu': DeepModel,
        'tdmeta': TaskDiffModel
    }[args['mode'].lower()](args['dataset'], emb_info=config, dataset_info=dataset_info, config_settings=config_settings)

    # train and val
    modeling = Modeling(model, config_settings=config_settings, mode=args['mode'])
    train_dict = {
        'deep': modeling.deep_learning_train,
        'melu': modeling.melu_train,
        'tdmeta': modeling.train
    }
    best_model = train_dict[args['mode'].lower()](train_dataset, test_dataset)

    # save param 
    if args['save'] == "True":
        torch.save(best_model, 'saved_models/{}.pth'.format(args['save_name']))