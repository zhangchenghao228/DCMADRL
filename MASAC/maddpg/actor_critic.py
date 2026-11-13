import torch
import torch.nn as nn
import torch.nn.functional as F
from maddpg.seed_init import set_global_seed
# from torch_geometric.nn import GCNConv
from torch.distributions import Normal
# 设置全局随机种子
set_global_seed(1212)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, args, agent_id, state_shape=11, action_shape=2):
        super(Actor, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear(state_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc_mu = torch.nn.Linear(128, action_shape)
        self.fc_std = torch.nn.Linear(128, action_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mu = self.fc_mu(x)
        # 添加 epsilon 防止 std 太小
        std = F.softplus(self.fc_std(x)) + 1e-7  # 防止 std 为零
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # 重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)

        # 计算 tanh_normal 分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        action = action * self.max_action
        return action, log_prob



class Critic(nn.Module):
    def __init__(self, state_action=11, action_shape=2, n_agents=5):
        super(Critic, self).__init__()
        self.state_action_shape = state_action + action_shape
        self.n_agents = n_agents
        self.fc1 = nn.Linear(n_agents * (state_action + action_shape), 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 512)
        self.q_out = nn.Linear(512, 1)

    def forward(self, state, action):
        batch_size = state.shape[0]
        state_action = torch.cat([state, action], dim=2)
        # state_action维度为batch_size,num_agents,state_shape+action_shape
        state_action = state_action.reshape(batch_size, self.n_agents * self.state_action_shape)
        x = F.relu(self.fc1(state_action))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        q_value = self.q_out(x)
        return q_value
