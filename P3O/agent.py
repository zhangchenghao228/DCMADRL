import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG
from maddpg.seed_init import set_global_seed
from torch.distributions import Normal
# 设置全局随机种子
set_global_seed(22)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)#将MADDPG实例移动到GPU

    def select_action(self, o, high_action=1.0):
        inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(device)  # 将输入移动到GPU
        mean, std = self.policy.actor_network(inputs)
        dist = Normal(mean, std) #from torch.distributions import Normal, Categorical
        action = dist.sample()
        action_log_pi = dist.log_prob(action) # 1xaction_dim#计算概率比值log pai
        u = action.detach().cpu().numpy().squeeze(0)  # 将结果移动到CPU以进行进一步处理, 感觉这个地方有没有.detach()都可以
        u_buffer = u.copy()  # 复制动作以存储在buffer中
        u = np.clip(u, -high_action, high_action)
        action_log_pis= action_log_pi.detach().cpu().numpy().squeeze(0)#感觉这个地方有没有.detach()都可以
        "这个地方先对动作进行裁剪, 之后存放到buffer里面的动作不是这个动作了吧,这个地方可能需要更改啊"
        # print("-----------------")
        # print("u:", u)
        # print("action_log_pis:", action_log_pis)
        # print("-----------------")
        return u.copy(), action_log_pis.copy(), u_buffer  # 返回动作和动作的对数概率

    "actor网络这里可以考虑用重采样化技术来做"

    def learn(self, transitions, critic_net, cost_net, logger):
        # print(type(gat_net))
        # 使用policy网络来更新
        self.policy.train_new(transitions, critic_net, cost_net, logger)

    def actor_state_dict(self):
        return self.policy.actor_state_dict()

    def actor_target_state_dict(self):
        return self.policy.actor_target_state_dict()
