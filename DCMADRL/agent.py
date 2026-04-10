import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(1212)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, agent_id, args, target_entropy):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id, target_entropy)#将MADDPG实例移动到GPU

    def select_action(self, o):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)  # 将输入移动到GPU
        pi = self.policy.actor_network(inputs)[0].squeeze(0)
        u = pi.cpu().numpy()  # 将结果移动到CPU以进行进一步处理
        return u.copy()

    def learn(self, transitions, other_agents, critic_net, cost_net, logger):
        # print(type(gat_net))
        # 使用policy网络来更新
        self.policy.train_new(transitions, other_agents, critic_net, cost_net, logger)

    def actor_state_dict(self):
        return self.policy.actor_state_dict()

    def actor_target_state_dict(self):
        return self.policy.actor_target_state_dict()
