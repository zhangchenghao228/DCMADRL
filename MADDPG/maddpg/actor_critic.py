import torch
import torch.nn as nn
import torch.nn.functional as F
from maddpg.seed_init import set_global_seed
# from torch_geometric.nn import GCNConv

# 设置全局随机种子
set_global_seed(52)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# define the actor network
'''把网络的输入，输出参数修改一下'''


class Actor(nn.Module):
    def __init__(self, args, agent_id, state_shape=11, action_shape=2):
        super(Actor, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear(state_shape, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.action_out = nn.Linear(128, action_shape)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


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
        # state维度为batch_size,num_agents,state_shape
        # action维度为batch_size,num_agents,action_shape
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
