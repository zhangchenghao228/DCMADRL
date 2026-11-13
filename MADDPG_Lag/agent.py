import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(22)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    def select_action(self, o, noise_rate, epsilon, high_action=1, action_shape=2):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-high_action, high_action, action_shape)
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)  # 将输入移动到GPU
            pi = self.policy.actor_network(inputs).squeeze(0)
            u = pi.cpu().numpy()  # 将结果移动到CPU以进行进一步处理
            # 动作的噪声取高斯噪声
            noise = noise_rate * high_action * np.random.randn(*u.shape)  # 高斯噪声
            u += noise
            # 线性插值，使动作分布在-动作空间到正的动作空间
            u = np.clip(u, -high_action, high_action)
        return u.copy()

    def learn(self, transitions, other_agents, critic_net, global_critic,  logger):
        # 使用policy网络来更新
        self.policy.train_new(transitions, other_agents, critic_net, global_critic, logger)

    def actor_state_dict(self):
        return self.policy.actor_state_dict()

    def actor_target_state_dict(self):
        return self.policy.actor_target_state_dict()
