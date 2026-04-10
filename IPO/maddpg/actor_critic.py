import torch
import torch.nn as nn
import torch.nn.functional as F
from maddpg.seed_init import set_global_seed
# from torch_geometric.nn import GCNConv

# 设置全局随机种子
set_global_seed(122)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# define the actor network
'''把网络的输入，输出参数修改一下'''


def net_init(m,gain=None,use_relu = True):
    use_orthogonal = True # -> 1
    use_relu = use_relu

    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_fuction = ['tanh','relu', 'leaky_relu']  # relu 和 leaky_relu 的gain值一样
    gain = gain if gain is not None else  nn.init.calculate_gain(activate_fuction[use_relu]) # 根据的激活函数设置
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)



# def __init__(self, args, agent_id, state_shape=11, action_shape=2):
class Actor(nn.Module):#这个是连续动作空间对应的actor网络
    def __init__(self, args, agent_id, obs_dim=11, action_dim=2, hidden_1=128, hidden_2=128):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim)) # 与PPO.py的方法一致：对角高斯函数
        #self.log_std_layer = nn.Linear(hidden_2, action_dim) # 式2

        net_init(self.l1)
        net_init(self.l2)
        net_init(self.mean_layer, gain=0.01)   

    def forward(self, x, ):
        # if self.trick['feature_norm']:
        x = F.layer_norm(x, x.size()[1:])#这个trick可以去掉，先注释掉吧
        x = F.relu(self.l1(x))
        x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        x = F.layer_norm(x, x.size()[1:])

        mean = torch.tanh(self.mean_layer(x))  # 使得mean在-1,1之间
        log_std = self.log_std.expand_as(mean)  # 使得log_std与mean维度相同 输出log_std以确保std=exp(log_std)>0
        #log_std = self.log_std_layer(x) # 式2
        log_std = torch.clamp(log_std, -20, 2) # exp(-20) - exp(2) 等于 2e-9 - 7.4，确保std在合理范围内
        std = torch.exp(log_std)

        return mean, std
    


# def __init__(self, state_action=11, action_shape=2, n_agents=5):
class Critic(nn.Module):#列表，里面通常是 [obs_dim, act_dim] 或类似结构
    def __init__(self, obs_dim=11, hidden_1=128 , hidden_2=128):
        super(Critic, self).__init__()
        # global_obs_dim = sum(val[0] for val in dim_info.values())  
        
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        
        net_init(self.l1)
        net_init(self.l2)
        net_init(self.l3)  
        # C:\Users\A1736\Desktop\MAPPO\maddpg\actor_critic.py
    def forward(self, s): # 传入全局观测和动作
        # s = torch.cat(list(s), dim = 1)#这个地方先去掉，后期根据batch_size的维度来处理
        s = F.layer_norm(s, s.size()[1:])#这个trick先去掉吧
        q = F.relu(self.l1(s))
        q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        q = F.layer_norm(q, q.size()[1:])
        q = self.l3(q)

        return q
    



#cost critic network, cost网络
class Cost_critic(nn.Module):#列表，里面通常是 [obs_dim, act_dim] 或类似结构
    def __init__(self, obs_dim=11, hidden_1=128 , hidden_2=128):
        super(Cost_critic, self).__init__()
        
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        
        net_init(self.l1)
        net_init(self.l2)
        net_init(self.l3)  

    def forward(self, s): # 传入全局观测和动作
        s = F.layer_norm(s, s.size()[1:])#这个trick先去掉吧
        q = F.relu(self.l1(s))
        q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        q = F.layer_norm(q, q.size()[1:])
        q = self.l3(q)

        return q