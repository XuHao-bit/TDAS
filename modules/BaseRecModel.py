import torch
from torch import nn
# from utils import config

# embedding model
class User_emb(nn.Module):
    def __init__(self, embedding_size, config) -> None:
        super().__init__()
        self.num_gender = config['n_gender']
        self.num_age = config['n_age']
        self.num_occupation = config['n_occupation']
        self.embedding_size = embedding_size

        self.embedding_gender = nn.Embedding(self.num_gender, self.embedding_size)
        self.embedding_age = nn.Embedding(self.num_age, self.embedding_size)
        self.embedding_occupation = nn.Embedding(self.num_occupation, self.embedding_size)
    
    def forward(self, x):
        gender_idx, age_idx, occupation_idx = x[:,0], x[:,1], x[:,2]
        gender_emb = self.embedding_gender(gender_idx)
        age_emb = self.embedding_age(age_idx)
        occupation_emb = self.embedding_occupation(occupation_idx)
        concat_emb = torch.cat((gender_emb, age_emb, occupation_emb), 1)
        # print(concat_emb)
        return concat_emb # b,n*es

class Item_emb(nn.Module):
    def __init__(self, embedding_size, config) -> None:
        super().__init__()
        self.num_rate = config['n_rate']
        self.num_genre = config['n_genre']
        self.num_director = config['n_director']
        self.num_year = config['n_year']
        self.embedding_size = embedding_size

        self.embedding_rate = nn.Embedding(self.num_rate, self.embedding_size)
        self.embedding_genre = nn.Linear(self.num_genre, self.embedding_size, bias=False)
        self.embedding_director = nn.Linear(self.num_director, self.embedding_size, bias=False)
        self.embedding_year = nn.Embedding(self.num_year, self.embedding_size)

        self.sigmoid_layer = nn.Sigmoid()

    def forward(self, x):
        rate_idx, year_idx, genre_idx, director_idx = x[:,0], x[:,1], x[:,2:27], x[:,27:]
        rate_emb = self.embedding_rate(rate_idx)
        year_emb = self.embedding_year(year_idx)
        genre_emb = self.sigmoid_layer(self.embedding_genre(genre_idx.float()))
        director_emb = self.sigmoid_layer(self.embedding_director(director_idx.float()))
        concat_emb = torch.cat((rate_emb, year_emb, genre_emb, director_emb), 1)
        return concat_emb # b,n*es

class BK_User_emb(torch.nn.Module):
    def __init__(self, embedding_dim, config):
        super(BK_User_emb, self).__init__()
        self.age_dim = config['n_age_bk']
        self.location_dim = config['n_location']
        self.embedding_dim = embedding_dim

        self.emb_age = torch.nn.Embedding(num_embeddings=self.age_dim, embedding_dim=self.embedding_dim)
        self.emb_location = torch.nn.Embedding(num_embeddings=self.location_dim, embedding_dim=self.embedding_dim)

    def forward(self, x1):
        age_idx, location_idx = x1[:,0], x1[:,1]
        age_emb = self.emb_age(age_idx)
        location_emb = self.emb_location(location_idx)
        concat_emb = torch.cat((age_emb, location_emb), 1)
        return concat_emb

class BK_Item_emb(torch.nn.Module):
    def __init__(self, embedding_dim, config):
        super(BK_Item_emb, self).__init__()
        self.year_dim = config['n_year_bk']
        self.author_dim = config['n_author']
        self.publisher_dim = config['n_publisher']
        self.embedding_dim = embedding_dim

        self.emb_year = torch.nn.Embedding(num_embeddings=self.year_dim, embedding_dim=self.embedding_dim)
        self.emb_author = torch.nn.Embedding(num_embeddings=self.author_dim, embedding_dim=self.embedding_dim)
        self.emb_publisher = torch.nn.Embedding(num_embeddings=self.publisher_dim, embedding_dim=self.embedding_dim)

    def forward(self, x2):
        author_idx, year_idx, publisher_idx = x2[:,0], x2[:,1], x2[:,2]
        year_emb = self.emb_year(year_idx)
        author_emb = self.emb_author(author_idx)
        publisher_emb = self.emb_publisher(publisher_idx)
        concat_emb = torch.cat((year_emb, author_emb, publisher_emb), 1)
        return concat_emb


class DB_User_emb(torch.nn.Module):
    def __init__(self, embedding_dim, config):
        super(DB_User_emb, self).__init__()
        self.location_dim = config['n_location_db']
        self.embedding_dim = embedding_dim

        self.emb_location = torch.nn.Embedding(num_embeddings=self.location_dim, embedding_dim=self.embedding_dim)

    def forward(self, x1):
        location_idx = x1[:,0]
        location_emb = self.emb_location(location_idx)
        return location_emb

class DB_Item_emb(torch.nn.Module):
    def __init__(self, embedding_dim, config):
        super(DB_Item_emb, self).__init__()
        self.year_dim = config['n_year_db']
        self.author_dim = config['n_author_db']
        self.publisher_dim = config['n_publisher_db']
        self.embedding_dim = embedding_dim

        self.emb_year = torch.nn.Embedding(num_embeddings=self.year_dim, embedding_dim=self.embedding_dim)
        self.emb_author = torch.nn.Embedding(num_embeddings=self.author_dim, embedding_dim=self.embedding_dim)
        self.emb_publisher = torch.nn.Embedding(num_embeddings=self.publisher_dim, embedding_dim=self.embedding_dim)

    def forward(self, x2):
        author_idx, publisher_idx, year_idx  = x2[:,0], x2[:,1], x2[:,2]
        year_emb = self.emb_year(year_idx)
        author_emb = self.emb_author(author_idx)
        publisher_emb = self.emb_publisher(publisher_idx)
        concat_emb = torch.cat((year_emb, author_emb, publisher_emb), 1)
        return concat_emb



# decoder 和 model modulater 可以分开写！
# 在forward的时候，就self.decoder.layer1()->alpha,bias->self.decoder.layer2就行
class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Decoder, self).__init__()
        self.x_dim = x_dim
        # self.z_dim = z_dim
        # self.task_dim = task_dim
        self.h1_dim = self.x_dim
        self.h2_dim = int(self.x_dim / 2)
        self.h3_dim = int(self.h2_dim / 2)
        # self.h_dims = [x_dim] + h_dims
        self.y_dim = y_dim
        # self.dropout_rate = dropout_rate
        # self.dropout = nn.Dropout(self.dropout_rate)

        self.hidden_layer_1 = nn.Linear(self.x_dim, self.h1_dim)
        self.hidden_layer_2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.hidden_layer_3 = nn.Linear(self.h2_dim, self.h3_dim)

        self.final_projection = nn.Linear(self.h3_dim, self.y_dim)
        self.relu_layer = nn.LeakyReLU()
        # self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x1, x2):
        inputs = torch.cat([x1, x2], dim=1)
        hidden_1 = self.hidden_layer_1(inputs)
        hidden_2 = self.relu_layer(hidden_1)

        hidden_2 = self.hidden_layer_2(hidden_2)
        hidden_3 = self.relu_layer(hidden_2)

        hidden_3 = self.hidden_layer_3(hidden_3)
        hidden_final = self.relu_layer(hidden_3)

        y_pred = self.final_projection(hidden_final)
        # return self.softmax_layer(y_pred)
        return y_pred

    def get_param_dict(self):
        return self.state_dict(keep_vars=True)
    
    # def get_param_with_grads(self):
    #     return self.parameters()


class RecModel(nn.Module):
    def __init__(self, user_emb, item_emb, decoder) -> None:
        super().__init__()
        self.user_emb_layer = user_emb
        self.item_emb_layer = item_emb
        self.decoder = decoder

    def forward(self, x1, x2):
        x1_out, x2_out = self.user_emb_layer(x1), self.item_emb_layer(x2)
        out = self.decoder(x1_out, x2_out)
        return out

    def get_embedding(self, x1, x2):
        x1_out, x2_out = self.user_emb_layer(x1), self.item_emb_layer(x2) # return nan
        # x1_out, x2_out = self.user_deep_layer(x1_out), self.item_deep_layer(x2_out)
        return torch.cat([x1_out, x2_out], dim=1)


class RecModel_old(nn.Module):
    def __init__(self, user_emb, item_emb, user_deep, item_deep, decoder) -> None:
        super().__init__()
        self.user_emb_layer = user_emb
        self.item_emb_layer = item_emb
        self.user_deep_layer = user_deep
        self.item_deep_layer = item_deep
        self.decoder = decoder

    def forward(self, x1, x2, weights=None):

        if weights is not None:
            self.decoder.load_state_dict(weights)

        x1_out, x2_out = self.user_emb_layer(x1), self.item_emb_layer(x2)
        x1_out, x2_out = self.user_deep_layer(x1_out), self.item_deep_layer(x2_out)
        out = self.decoder(x1_out, x2_out)
        return out

    def get_embedding(self, x1, x2):
        x1_out, x2_out = self.user_emb_layer(x1), self.item_emb_layer(x2)
        x1_out, x2_out = self.user_deep_layer(x1_out), self.item_deep_layer(x2_out)
        return torch.cat([x1_out, x2_out], dim=1)
