from __future__ import print_function
from prepare_data.prepareMovielens import *
# from prepare_data.prepareBookCrossing import *
# from prepare_data.prepareDoubanBook import *


def id_storing(dataset='movielens', max_count=20):
    # dataset option: 'movielens', 'bookcrossing'
    # max_count: storing_size
    storing_path = 'data_processed'
    if not os.path.exists(storing_path):
        os.mkdir(storing_path)
    if not os.path.exists('{}/{}'.format(storing_path, dataset)):
        os.mkdir('{}/{}'.format(storing_path, dataset))

    if dataset == 'movielens':
        # id_storing_movielens_cold_only(max_count=max_count)
        id_storing_movielens(max_count=max_count)


def dict_storing(dataset='movielens'):
    if dataset == 'movielens':
        dict_storing_movielens()
    # else:
    #     dict_storing_bookcrossing()



def data_generation(dataset='movielens', save_state=True):
    # inputs: movies info, ratings info, state ids
    # outputs: training and testing samples.
    #           Each sample is from a user.
    #           Each sample includes support set and query set.
    # output type: mixed, warm, cold_user, cold_item, cold_user_item
    storing_path = 'data_processed'

    if not os.path.exists('{}/{}/raw'.format(storing_path, dataset)):
        os.mkdir('{}/{}/raw/'.format(storing_path, dataset))
        processing_code = 1
    elif not os.path.exists('{}/{}/raw/sample_1_x1.p'.format(storing_path, dataset)):
        processing_code = 1
    else:
        processing_code = 0
        print('Data already generated! ')

    if processing_code == 1:
        sorted_ratings = pickle.load(open('{}/{}/ratings_sorted.p'.format(storing_path, dataset), 'rb'))
        user_state_ids = pickle.load(open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'rb'))
        item_state_ids = pickle.load(open('{}/{}/item_state_ids.p'.format(storing_path, dataset), 'rb'))
        user_dict = pickle.load(open('{}/{}/user_dict.p'.format(storing_path, dataset), 'rb'))
        item_dict = pickle.load(open('{}/{}/item_dict.p'.format(storing_path, dataset), 'rb'))

        u_all_ids = user_state_ids['user_all_ids']

        file_index = 1

        for u_id in tqdm(u_all_ids):
            u_info = sorted_ratings.loc[sorted_ratings.user_id == u_id]
            ratings = np.array(u_info.rating)
            u_feature = user_dict[u_id]

            u_i_ids = u_info.item_id

            i_feature_file = []
            state_codes = []

            for i_id in u_i_ids:
                if i_id in item_dict.keys():
                    i_feature = item_dict[i_id]
                    i_feature_file.append(i_feature)

            pickle.dump(u_feature, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_x1.p', 'wb'))
            pickle.dump(i_feature_file, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_x2.p', 'wb'))
            pickle.dump(ratings, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_y.p', 'wb'))
            if save_state:
                pickle.dump(state_codes, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_y0.p', 'wb'))

            file_index += 1

def data_generation_dropnet_ml(dataset='movielens', save_state=True):
    
    # inputs: movies info, ratings info, state ids
    # outputs: training and testing samples.
    #           Each sample is from a user.
    #           Each sample includes support set and query set.
    # output type: mixed, warm, cold_user, cold_item, cold_user_item
    storing_path = 'data_processed'

    if not os.path.exists('{}/{}/raw'.format(storing_path, dataset)):
        os.mkdir('{}/{}/raw/'.format(storing_path, dataset))
    # processing_code = 1
    elif not os.path.exists('{}/{}/raw/sample_1_x1.p'.format(storing_path, dataset)):
        processing_code = 1
    else:
        processing_code = 0
        print('Data already generated! ')

    if processing_code == 1:
        sorted_ratings = pickle.load(open('{}/{}/ratings_sorted.p'.format(storing_path, dataset), 'rb'))
        user_state_ids = pickle.load(open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'rb'))
        item_state_ids = pickle.load(open('{}/{}/item_state_ids.p'.format(storing_path, dataset), 'rb'))
        user_dict = pickle.load(open('{}/{}/user_dict.p'.format(storing_path, dataset), 'rb'))
        item_dict = pickle.load(open('{}/{}/item_dict.p'.format(storing_path, dataset), 'rb'))

        u_all_ids = user_state_ids['user_all_ids']

        file_index = 1

        for u_id in tqdm(u_all_ids):
            u_info = sorted_ratings.loc[sorted_ratings.user_id == u_id]
            ratings = np.array(u_info.rating)
            u_feature = user_dict[u_id]

            u_i_ids = u_info.item_id

            i_feature_file = []
            state_codes = []

            for i_id in u_i_ids:
                if i_id in item_dict.keys():
                    i_feature = item_dict[i_id]
                    i_feature_file.append(i_feature)

            pickle.dump(u_feature, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_x1.p', 'wb'))
            pickle.dump(i_feature_file, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_x2.p', 'wb'))
            pickle.dump(u_i_ids, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_x3.p', 'wb')) # this file is used by the baseline `DropoutNet`
            pickle.dump(ratings, open('{}/{}/raw/'.format(storing_path, dataset)+'sample_'+str(file_index)+'_y.p', 'wb'))


            file_index += 1

if __name__ == '__main__':
    # movielens data generating
    id_storing_movielens_cold_only(max_count=40) # 交互数量超过40的才存储，40~253
    dict_storing()
    data_generation_dropnet_ml(dataset='movielens', save_state=False)
    # bookcrossing data generating
    # id_storing_bookcrossing_cold_only(max_count=40)
    # dict_storing_bookcrossing()
    # data_generation_dropnet_bk(save_state=False)
    # dbook 
    # id_storing_dbook_cold_only(max_count=40)
    # dict_storing_dbook()
    # data_generation_dropnet_db(save_state=False)