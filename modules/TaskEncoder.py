import torch
from torch import nn


# task composition difficulty 
class TaskEncoder(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, device) -> None:
      super().__init__()
      # hyper parameter
      self.task_dim = task_dim
      self.num_class = num_class
      self.emb_size = emb_size
      self.feat_num = feat_num
      self.max_shot = max_shot
      self.device = device

      # some variable
      self.s_size = self.emb_size
      self.v_size = int(self.emb_size / 2)

      # layers
      # statistic
      self.sta_inter_layer = nn.Linear(2*self.feat_num*self.emb_size+1, self.s_size)
      self.sta_inter_layer2 = nn.Linear(self.v_size*2+2, self.task_dim)
      self.relu_layer = nn.ReLU()

      # encoder
      self.en_layer = nn.Sequential(
        nn.Linear(self.emb_size, int(self.emb_size/2)),
        nn.ReLU(),
        nn.Linear(int(self.emb_size/2), int(self.emb_size/2))
      )
    

    # Compute element-wise sample mean, var., and set cardinality
    # x:(n, es)
    def _statistic_pooling(self, x):
      var = torch.var(x, dim=0)
      mean = torch.mean(x, dim=0)
      if x.shape[0] == 1:
        var = torch.zeros_like(x)
      return var.reshape(-1), mean
    

    # Input：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def forward(self, x, y):

      # (1) Intra-class statistics
      s = []; Nc_list = []; mean_list = []
      for c in range(self.num_class):
        if c not in y:
          continue
        c_mask = torch.eq(y, c) # cuda
        c_x = x[c_mask] # 不可用masked_select
        c_y = torch.masked_select(y, c_mask)
        N_c = torch.tensor([(len(c_y)-1.)/(self.max_shot-1.)], device=self.device)
        Nc_list.append(N_c)
        # calculate `\mu_c, \sigma_c, k_c`
        c_var, c_mean = self._statistic_pooling(c_x) # 这里cvar出现的nan
        s.append(torch.cat([c_var, c_mean, N_c], 0))
        mean_list.append(c_mean) # c_y, tensor([feat_num*emb_size])
      
      # s is a list of `\mu_c, \sigma_c, k_c`
      s = torch.stack(s, 0) # n_c, 2*f*es+1
      # interaction
      s = self.relu_layer(self.sta_inter_layer(s)) 
      v = self.en_layer(s)

      # (2) Inter-class statistics
      v_var, v_mean = self._statistic_pooling(v)
      v_N = torch.mean(torch.tensor(Nc_list))
      # calculate `\pi_v`
      len_mean_list = len(mean_list)
      mean_tensor = torch.cat(mean_list, dim=0).reshape(len_mean_list, -1) # list of tensor -> tensor
      mean_norm = mean_tensor / torch.max(torch.norm(mean_tensor, p=2, dim=-1, keepdim=True), torch.tensor(1e-7).to(self.device))
      cos_sim = torch.einsum('ni,mi->nm', mean_norm, mean_norm)
      sim_sum = torch.tensor([0.], device=self.device)
      for i in range(len_mean_list):
        for j in range(i+1, len_mean_list):
          sim_sum += (j-i)*cos_sim[i][j]
      v = torch.cat([v_var,v_mean,torch.tensor([v_N], device=self.device),sim_sum])
      
      # return task-composition difficulty
      return self.relu_layer(self.sta_inter_layer2(v))

# task composition difficulty + task relevance difficulty
class TaskMemEncoder(nn.Module):
    def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device) -> None:
      super().__init__()
      # 超参数
      self.task_dim = task_dim
      self.num_class = num_class
      self.emb_size = emb_size
      self.feat_num = feat_num
      self.max_shot = max_shot
      self.device = device
      self.clusters_k = clusters_k
      self.temperature = temperature
      self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)

      # 部分变量参数
      self.e_size = self.feat_num*self.emb_size
      self.s_size = self.emb_size
      self.v_size = int(self.emb_size / 2)

      # layers
      # statistic
      self.sta_inter_layer = nn.Linear(2*self.feat_num*self.emb_size+1, self.s_size)
      self.sta_inter_layer2 = nn.Linear(self.v_size*2+2, self.task_dim)
      # self.final_emb = nn.Linear(2*self.task_dim, self.task_dim)
      self.relu_layer = nn.ReLU()

      # encoder
      self.en_layer0 = nn.Sequential(
        nn.Linear(self.e_size, self.e_size),
        nn.ReLU(),
        nn.Linear(self.e_size, self.e_size)
      )

      self.en_layer = nn.Sequential(
        nn.Linear(self.emb_size, int(self.emb_size/2)),
        nn.ReLU(),
        nn.Linear(int(self.emb_size/2), int(self.emb_size/2))
      )
    
    # Compute element-wise sample mean, var., and set cardinality
    # x:(n, es)
    def _statistic_pooling(self, x):
      var = torch.var(x, dim=0)
      mean = torch.mean(x, dim=0)
      if x.shape[0] == 1:
        var = torch.zeros_like(x)
      return var.reshape(-1), mean
    
    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
    def forward(self, x, y):
      # encoder 1
      x = self.en_layer0(x)
      
      # statistic 1
      s = []; Nc_list = []; mean_list = []
      for c in range(self.num_class):
        if c not in y:
          continue
        c_mask = torch.eq(y, c) # cuda
        c_x = x[c_mask] # 不可用masked_select
        c_y = torch.masked_select(y, c_mask)
        N_c = torch.tensor([(len(c_y)-1.)/(self.max_shot-1.)], device=self.device)
        Nc_list.append(N_c)
        c_var, c_mean = self._statistic_pooling(c_x) # 这里cvar出现的nan
        s.append(torch.cat([c_var, c_mean, N_c], 0))
        mean_list.append(c_mean) # c_y, tensor([feat_num*emb_size])
      
      s = torch.stack(s, 0) # n_c, 2*f*es+1
      s = self.relu_layer(self.sta_inter_layer(s)) # interaction

      # encoder 2
      v = self.en_layer(s)

      # statistic 2
      v_var, v_mean = self._statistic_pooling(v)
      v_N = torch.mean(torch.tensor(Nc_list))
      
      len_mean_list = len(mean_list)
      mean_tensor = torch.cat(mean_list, dim=0).reshape(len_mean_list, -1) # list of tensor -> tensor
      mean_norm = mean_tensor / torch.max(torch.norm(mean_tensor, p=2, dim=-1, keepdim=True), torch.tensor(1e-7).to(self.device))
      cos_sim = torch.einsum('ni,mi->nm', mean_norm, mean_norm)
      sim_sum = torch.tensor([0.], device=self.device)
      for i in range(len_mean_list):
        for j in range(i+1, len_mean_list):
          sim_sum += (j-i)*cos_sim[i][j]
      
      v = torch.cat([v_var,v_mean,torch.tensor([v_N], device=self.device),sim_sum])
      task_emb = self.relu_layer(self.sta_inter_layer2(v))
      task_mem_emb = self.task_mem(task_emb)
      # print(task_emb.shape, task_mem_emb.shape)
      
      # return self.relu_layer(self.final_emb(torch.cat([task_emb, task_mem_emb])))
      return torch.cat([task_emb, task_mem_emb])

# task composition difficulty (w/o Intra- and Inter- CS)
class TaskMeanEncoder(nn.Module):
  def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, device) -> None:
    super().__init__()
    self.task_dim = task_dim
    # self.num_class = num_class
    self.emb_size = emb_size
    self.feat_num = feat_num
    # self.max_shot = max_shot
    self.device = device

    self.encoder_layer = nn.Linear(self.feat_num*self.emb_size, self.task_dim)
    self.relu_layer = nn.ReLU()

  def forward(self, x, y):
    return torch.mean(self.relu_layer(self.encoder_layer(x)), dim=0)

# task composition difficulty + task relevance difficulty (w/o Intra- and Inter- CS)
class TaskMeanMemEncoder(nn.Module):
  def __init__(self, task_dim, num_class, emb_size, feat_num, max_shot, clusters_k, temperature, device) -> None:
    super().__init__()
    # 超参数
    self.task_dim = task_dim
    self.emb_size = emb_size
    self.device = device
    self.feat_num = feat_num
    self.clusters_k = clusters_k
    self.temperature = temperature
    self.task_mem = MemoryUnit(self.clusters_k, self.task_dim, self.temperature)

    # 部分变量参数
    self.encoder_layer = nn.Linear(self.feat_num*self.emb_size, self.task_dim)
    self.relu_layer = nn.ReLU()

    # 输入：x:(batch_size, feat_num*emb_size), y:(batch_size, 1)
  def forward(self, x, y):
    task_emb = torch.mean(self.relu_layer(self.encoder_layer(x)), dim=0)
    task_mem_emb = self.task_mem(task_emb)
    return torch.cat([task_emb, task_mem_emb])
  

# task relevance difficulty
class Attention(torch.nn.Module):
    def __init__(self, n_k):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.relu_layer = torch.nn.ReLU()
        self.fc_layer = torch.nn.Linear(self.n_k, self.n_k)
        self.soft_max_layer = torch.nn.Softmax()

    def forward(self, pu, mp):
        expanded_pu = pu.repeat(1, len(mp)).view(len(mp), -1)  # shape, n_k, pu_dim
        inputs = self.cos_sim(expanded_pu, mp)
        fc_layers = self.relu_layer(self.fc_layer(inputs))
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values

    def cos_sim(input1, input2):
        query_norm = torch.sqrt(torch.sum(input1**2+0.00001, 1))
        doc_norm = torch.sqrt(torch.sum(input2**2+0.00001, 1))

        prod = torch.sum(torch.mul(input1, input2), 1)
        norm_prod = torch.mul(query_norm, doc_norm)

        cos_sim_raw = torch.div(prod, norm_prod)
        return cos_sim_raw

class MemoryUnit2(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature, device):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.device = device
        self.array = nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))
        self.att_model = Attention(self.n_k).to(self.device)

    def forward(self, task_embed):
        atten = self.att_model(task_embed, self.array).to(self.device)
        # res = torch.norm(task_embed-self.array, p=2, dim=1, keepdim=True)
        # res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # # 1*k
        # C = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(atten, self.array)
        # simple add operation
        new_task_embed = torch.cat([task_embed, value], dim=1)
        # calculate target distribution
        return new_task_embed

class MemoryUnit(nn.Module):
    # clusters_k is k keys
    def __init__(self, clusters_k, emb_size, temperature):
        super(MemoryUnit, self).__init__()
        self.clusters_k = clusters_k
        self.embed_size = emb_size
        self.temperature = temperature
        self.array = nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(self.clusters_k, self.embed_size)))

    def forward(self, task_embed):
        res = torch.norm(task_embed-self.array, p=2, dim=1, keepdim=True)
        res = torch.pow((res / self.temperature) + 1, (self.temperature + 1) / -2)
        # 1*k
        C = torch.transpose(res / res.sum(), 0, 1)
        # 1*k, k*d, 1*d
        value = torch.mm(C, self.array)
        return value.view(-1)
