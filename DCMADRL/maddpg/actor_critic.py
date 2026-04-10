import torch
import torch.nn as nn
import torch.nn.functional as F
from maddpg.seed_init import set_global_seed
# from torch_geometric.nn import GCNConv
from torch.distributions import Normal
from rlkit import pytorch_util as ptu 
import numpy as np
# 设置全局随机种子
set_global_seed(1212)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, args, agent_id, state_shape=11, action_shape=2):
        super(Actor, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear(state_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc_mu = torch.nn.Linear(128, action_shape)
        self.fc_std = torch.nn.Linear(128, action_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
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

class QuantileMlp(nn.Module):

    def __init__(
            self,
            hidden_sizes = [256, 256], # hidden_sizes=[M, M], 是隐藏层神经网络的形状
            # output_size,
            input_size = 65, # state_shape = 11 + action_shape = 2； agents * (state_shape + action_shape) = 5 * 13 = 65
            embedding_size=64, # Quantile fraction embedding size
            num_quantiles=32,
            layer_norm=True,
            # **kwargs,
    ):
        super().__init__()
        self.layer_norm = layer_norm
        # hidden_sizes[:-2] MLP base
        # hidden_sizes[-2] before merge
        # hidden_sizes[-1] before output

        self.base_fc = []
        last_size = input_size
        for next_size in hidden_sizes[:-1]: #这里的hidden_sizes[:-1]是对列表的切片索引操作
            self.base_fc += [
                nn.Linear(last_size, next_size),
                nn.LayerNorm(next_size) if layer_norm else nn.Identity(),
                nn.ReLU(inplace=True),#这个地方最好是
            ]
            last_size = next_size
        self.base_fc = nn.Sequential(*self.base_fc)#*是可迭代对象得解包操作
        self.num_quantiles = num_quantiles
        self.embedding_size = embedding_size
        self.tau_fc = nn.Sequential(
            nn.Linear(embedding_size, last_size),
            nn.LayerNorm(last_size) if layer_norm else nn.Identity(),
            nn.Sigmoid(),
        )
        self.merge_fc = nn.Sequential(
            nn.Linear(last_size, hidden_sizes[-1]),
            nn.LayerNorm(hidden_sizes[-1]) if layer_norm else nn.Identity(),
            nn.ReLU(inplace=True),
        )
        self.last_fc = nn.Linear(hidden_sizes[-1], 1)
        self.const_vec = ptu.from_numpy(np.arange(1, 1 + self.embedding_size))#生成一个等差数列，这里的间隔是默认值1
        #ptu.from_numpy将 NumPy 数组转为 float 类型的 PyTorch 张量，并移到 GPU 上（或默认设备）

    def forward(self, state, action, tau):

        state = torch.cat(state, dim=1)
        action = torch.cat(action, dim=1)
        """
        Calculate Quantile Value in Batch
        tau: quantile fractions, (N, T)

        state is (batch_size, state_shape) tensor
        action is (batch_size, action_shap) tensor
        tau is (batch_szie, num_quantiles) tensor
        """
        h = torch.cat([state, action], dim=1)#将state和action这两个tensor按照第二个维度进行拼接，这个地方需要action和state都要是tensor
        h = self.base_fc(h)  # (N, C)

        x = torch.cos(tau.unsqueeze(-1) * self.const_vec * np.pi)  # (N, T, E)，这个地方是是的每个tau都变为一个E维的余弦向量
        x = self.tau_fc(x)  # (N, T, C)

        h = torch.mul(x, h.unsqueeze(-2))  # (N, T, C)
        h = self.merge_fc(h)  # (N, T, C)
        output = self.last_fc(h).squeeze(-1)  # (N, T)
        return output


class Critic(nn.Module):
    def __init__(self, state_shape=11, action_shape=2, n_agents=1):
        super(Critic, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear((n_agents * (state_shape + action_shape)), 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


class Cost_critic(nn.Module):
    def __init__(self, state_shape=11, action_shape=2, n_agents=5):
        super(Cost_critic, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear((n_agents * (state_shape + action_shape)), 256)
        self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
