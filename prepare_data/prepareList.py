'''
Author: Xuhao Zhao, 2023
'''

import re
import pickle
# import pandas as pd
# from prepareBookCrossing import load_bookcrossing


def load_list(f_name):
    list_ = []
    with open(f_name, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

def dump_list(f_name, list_):
    with open(f_name, 'w', encoding="utf-8") as f:
        for l in list_:
            f.write(str(l).strip()+'\n')

def pickle_load(file_name):
    path = 'data_raw/book_crossing/'
    target_file = pickle.load(open(path+file_name, 'rb'))
    return target_file


# movieLens
list_movieLens = {
    'list_age': [1, 18, 25, 35, 45, 50, 56],
    'list_gender': ['M', 'F'],
    'list_occupation': list(range(0, 21)),
    'list_genre': load_list('data_raw/ml-1m/List_genre.txt'),
    'list_rate': ['PG-13', 'UNRATED', 'NC-17', 'PG', 'G', 'R'],
    'list_director': load_list('data_raw/ml-1m/List_director.txt')
}

# bookcrossing
"""
'n_year_bk': 80,
'n_author': 25593,
'n_publisher': 5254,
'n_age_bk': 106,
'n_location': 65,
"""
# list_bookcrossing = {
#     'list_age': [float(l) for l in load_list('data_raw/book_crossing/list_age.txt')],
#     'list_location': load_list('data_raw/book_crossing/list_location.txt'),
#     'list_year': [float(l) for l in load_list('data_raw/book_crossing/list_year.txt')],
#     'list_author': load_list('data_raw/book_crossing/list_author.txt'),
#     'list_publisher': load_list('data_raw/book_crossing/list_publisher.txt')
# }

def user_converting_ml(user_row, age_list, gender_list, occupation_list):
    # gender_dim: 2, age_dim: 7, occupation: 21
    gender_idx = gender_list.index(user_row.iat[0, 1])
    age_idx = age_list.index(user_row.iat[0, 2])
    occupation_idx = occupation_list.index(user_row.iat[0, 3])
    return [gender_idx, age_idx, occupation_idx]


def item_converting_ml(item_row, rate_list, genre_list, director_list, year_list):
    # rate_dim: 6, year_dim: 1,  genre_dim:25, director_dim: 2186,
    rate_idx = rate_list.index(item_row.iat[0, 3])
    genre_idx = [0] * 25
    for genre in str(item_row.iat[0, 4]).split(", "):
        idx = genre_list.index(genre)
        genre_idx[idx] = 1
    director_idx = [0] * 2186
    for director in str(item_row.iat[0, 5]).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[idx] = 1
    year_idx = year_list.index(item_row.iat[0, 2])
    out_list = list([rate_idx, year_idx])
    out_list.extend(genre_idx)
    out_list.extend(director_idx)
    return out_list

# def get_bk_list():
#     bu, bb, br = load_bookcrossing()
#     res = pd.merge(bu, br, on='User-ID')
#     res = pd.merge(res, bb, on='ISBN')
#     res = res.dropna()
#     age_list = res['Age'].drop_duplicates().to_list() # len:99
#     location_list = res['Location'].drop_duplicates().to_list() # len:123
#     author_list = res['Book-Author'].drop_duplicates().to_list() # 84725 
#     year_list = res['Year-Of-Publication'].drop_duplicates().to_list() # len:97
#     publisher_list = res['Publisher'].drop_duplicates().to_list() # len:13896
#     print(len(age_list), len(location_list), len(author_list),
#         len(year_list), len(publisher_list))
    # list.txt应该在data_raw下面
    # dump_list('list_age.txt', age_list)
    # dump_list('list_location.txt', location_list)
    # dump_list('list_author.txt', author_list)
    # dump_list('list_year.txt', year_list)
    # dump_list('list_publisher.txt', publisher_list)


def user_converting_bk(user_row, age_list, location_list):
    age_idx = age_list.index(user_row.iat[0, 2])
    location_idx = location_list.index(user_row.iat[0, 1])
    return [age_idx, location_idx]


def item_converting_bk(item_row, author_list, year_list, publisher_list):
    # print(type(item_row.iat[0, 2]))
    author_idx = author_list.index(str(item_row.iat[0, 1]).strip())
    year_idx = year_list.index(item_row.iat[0, 2])
    publisher_idx = publisher_list.index(str(item_row.iat[0, 3]).strip())
    return [author_idx, year_idx, publisher_idx]