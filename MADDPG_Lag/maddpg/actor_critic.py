import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np


# define the actor network
'''把网络的输入，输出参数修改一下'''
# 设置全局随机种子
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(22)


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


'''这个也要修改一下'''


class Critic(nn.Module):
    def __init__(self, args, agent_id, state_shape=11, action_shape=2, n_agents=5):
        super(Critic, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear((n_agents * (state_shape + action_shape)), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        # print(state.shape)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


# class GlobalCritic(nn.Module):
#     def __init__(self, args, agent_id, state_shape=11, action_shape=2, n_agents=5):
#         super(GlobalCritic, self).__init__()
#         self.max_action = 1
#         self.fc1 = nn.Linear((n_agents * (state_shape + action_shape)), 512)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc3 = nn.Linear(512, 512)
#         # self.fc4 = nn.Linear(512, 512)
#         self.q_out = nn.Linear(512, 1)
#
#     def forward(self, state, action):
#         state = torch.cat(state, dim=1)
#         for i in range(len(action)):
#             action[i] /= self.max_action
#         action = torch.cat(action, dim=1)
#         x = torch.cat([state, action], dim=1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         # x = F.relu(self.fc4(x))
#         q_value = self.q_out(x)
#         return q_value

class GlobalCritic(nn.Module):
    def __init__(self, args, agent_id, state_shape=11, action_shape=2, n_agents=5):
        super(GlobalCritic, self).__init__()
        self.max_action = 1
        self.fc1 = nn.Linear((n_agents * (state_shape + action_shape)), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.q_out = nn.Linear(128, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        # print(state.shape)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
