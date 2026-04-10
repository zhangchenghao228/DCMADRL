import torch
import os
from maddpg.actor_critic import Actor, Critic, Cost_critic, QuantileMlp
from maddpg.seed_init import set_global_seed
import numpy as np
import torch.nn.functional as F
import math
import time
from rlkit import pytorch_util as ptu 
# 设置全局随机种子
set_global_seed(1212)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, args, agent_id, target_entropy):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_q = 0
        self.actor_loss = 0


        self.tau_type = 'iqn' #这个用于选取不同的分布式强化学习的方法

        self.target_fp = None #分布式强化学习
        self.fp = None
        self.num_quantiles = 32

        self.cost_limit = 0.25

        "MATD3新增内容"
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.max_action = args.max_action
        self.actor_pointer = 0
        self.policy_update_freq = args.policy_update_freq

        # create the network
        self.actor_network = Actor(args, agent_id).to(device)
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)

        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.args.lr_alpha)
        self.target_entropy = target_entropy  # 目标熵的大小

        self.lagrangian = torch.tensor(1.0, requires_grad=True)
        self.lagrangian_optimizer = torch.optim.Adam([self.lagrangian], lr=self.args.lagrangian_lr)

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles#len(actions)是用于获取batch_size的大小
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau


    # MADDPG
    def train_new(self, transitions, other_agents, critic_net, cost_critic, logger=None, n_agents=5):
        self.actor_pointer += 1
        # 重点更新网络参数代码
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32).to(device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        c = transitions['c_%d' % self.agent_id]
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项

        for agent_id in range(n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])  # o现在
        u_next = []

        with torch.no_grad():
            index = 0
            for agent_id in range(n_agents):
                if agent_id == self.agent_id:
                    batch_u_next, batch_u_next_log_prob = self.actor_network(o_next[self.agent_id])
                    entropy = -batch_u_next_log_prob
                    u_next.append(batch_u_next)
                else:
                    u_next.append(other_agents[index].policy.actor_network(o_next[agent_id])[0])
                    index += 1
            next_tau, next_tau_hat, next_presum_tau = self.get_tau(o_next[0], u_next[0], fp=self.target_fp)#这个地方不需要梯度操作



        # with torch.no_grad():
        #     batch_u_next, batch_u_next_log_prob = self.actor_network(o_next[0])
        #     entropy = -batch_u_next_log_prob
        #     u_next.append(batch_u_next)
        #     next_tau, next_tau_hat, next_presum_tau = self.get_tau(o_next[0], u_next[0], fp=self.target_fp)#这个地方不需要梯度操作

        #critic网络更新
        critic_net.upda(o, u, o_next, u_next, r, entropy, self.log_alpha, logger)

        #cost网络更新
        cost_critic.upda(o, u, o_next, u_next, c, next_tau_hat, next_presum_tau, logger)


        u[self.agent_id], log_pros_actor = self.actor_network(o[self.agent_id])
        entropy_actor = -log_pros_actor


        with torch.no_grad():
            new_tau, new_tau_hat, new_presum_tau = self.get_tau(o[0], u[0], fp=self.fp)

        actor_critic_loss = critic_net.actor_loss(o, u, entropy_actor, self.log_alpha)


        self.actor_loss = actor_critic_loss

        cost_critic_loss = cost_critic.actor_loss(o, u, self.cost_limit, new_tau_hat, new_presum_tau)


        actor_loss = actor_critic_loss + cost_critic_loss

        # print("actor_critic_loss shape:", actor_critic_loss.shape)
        # print("cost_critic_loss shape:", cost_critic_loss.shape)
        # print("actor_loss shape:", actor_loss.shape)

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        #更新alpha
        alpha_loss = torch.mean(
            (entropy_actor - self.target_entropy).detach() * self.log_alpha.exp()) # entropy_actor = -log_pros_actor; - self.log_alpha * (entropy_actor - self.target_entropy)
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # # ---------- Lagrangian 更新 ----------
        # cost_mean = cost_critic_loss.detach()
        # lagrangian_loss = - self.lagrangian * (cost_mean - self.cost_limit)#-lagrangian * (cost_mean - self.cost_limit)是lagrangian loss

        # self.lagrangian_optimizer.zero_grad()
        # lagrangian_loss.backward()
        # self.lagrangian_optimizer.step()
        # 确保 self.lagrangian 大于等于0
        # with torch.no_grad():
        #     self.lagrangian.data.clamp_(min=0.0)


        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            # 定期保存模型参数
            self.save_model(self.train_step)

        self.train_step += 1

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        return model

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.actor_network.state_dict(), model_path + '/' + num + '_actor_params.pkl')

    def actor_state_dict(self):
        return self.actor_network.state_dict()

    def actor_target_state_dict(self):
        return self.actor_target_network.state_dict()


class CRITIC_NET:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_q = 0
        self.critic_loss1 = 0
        self.critic_loss2 = 0

        self.critic_network1 = Critic().to(device)
        self.critic_target_network1 = Critic().to(device)
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic)

        self.critic_network2 = Critic().to(device)
        self.critic_target_network2 = Critic().to(device)
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)


        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'critic_agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target_network1.parameters(), self.critic_network1.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        for target_param, param in zip(self.critic_target_network2.parameters(), self.critic_network2.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)


    def upda(self, o, u, o_next, u_next, r, entropy, log_alpha, logger):
        with torch.no_grad():
            # 使用目标网络预测下一个状态的 Q 值
            q_next_1 = self.critic_target_network1(o_next, u_next).detach()
            q_next_2 = self.critic_target_network2(o_next, u_next).detach()
            next_value = torch.min(q_next_1, q_next_2) + log_alpha.exp() * entropy
            target_q = (r.unsqueeze(1) + self.args.gamma * next_value).detach()


        # time1 = time.time()
        # 计算当前 Q 网络的 Q 值
        q_value1 = self.critic_network1(o, u)
        q_value2 = self.critic_network2(o, u)


        # 计算两个 Q 网络的损失
        critic_loss1 = torch.mean(F.mse_loss(q_value1, target_q))
        critic_loss2 = torch.mean(F.mse_loss(q_value2, target_q))


        # 反向传播和优化
        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        # 更新损失值
        self.critic_loss1 = critic_loss1.item()
        self.critic_loss2 = critic_loss2.item()

        self._soft_update_target_network()

        self.train_step += 1


    def actor_loss(self, o, u, entropy, log_alpha):
        q1_value = self.critic_network1(o, u)
        q2_value = self.critic_network2(o, u)
        actor_loss = torch.mean(-log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        return actor_loss

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'critic_agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.critic_network1.state_dict(), model_path + '/' + num + '_critic1_params.pkl')


class COST_CRITIC_NET:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_q = 0
        self.critic_loss1 = 0
        self.critic_loss2 = 0

        self.tau_type = 'iqn'
        self.num_quantiles = 32
        self.fp = None


        self.ipo_t = 10  # 内点惩罚参数

        self.critic_network1 = QuantileMlp().to(device)
        self.critic_target_network1 = QuantileMlp().to(device)
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic)

        # self.critic_network2 = Cost_critic().to(device)
        # self.critic_target_network2 = Cost_critic().to(device)
        # self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        # self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)


        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'cost_critic_agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target_network1.parameters(), self.critic_network1.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        # for target_param, param in zip(self.critic_target_network2.parameters(), self.critic_network2.parameters()):
        #     target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles#len(actions)是用于获取batch_size的大小
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau
    
    def quantile_regression_loss(self, input, target, tau, weight):
        """
        input: (N, T)
        target: (N, T)
        tau: (N, T)
        输入都是tensor
        """
        input = input.unsqueeze(-1) # (N, T, 1)
        target = target.detach().unsqueeze(-2) # (N, 1, T)
        tau = tau.detach().unsqueeze(-1) # (N, T, 1)
        weight = weight.detach().unsqueeze(-2) # (N, 1, T)
        expanded_input, expanded_target = torch.broadcast_tensors(input, target) #广播到(N,T,T)形状
        L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
        sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
        rho = torch.abs(tau - sign) * L * weight
        return rho.sum(dim=-1).mean()
        """最后一个维度是目标分位点索引j,对它求和是为了每个预测分位i聚合其与所有目标分位之间的损失,符合分位数回归的理论定义"""


    def upda(self, o, u, o_next, u_next, r, next_tau_hat, next_presum_tau, logger):
        with torch.no_grad():
            # 使用目标网络预测下一个状态的 Q 值
            # q_next_1 = self.critic_target_network1(o_next, u_next).detach()
            # q_next_2 = self.critic_target_network2(o_next, u_next).detach()
            # next_value = torch.min(q_next_1, q_next_2)
            q_next_1 = self.critic_target_network1(o_next, u_next, next_tau_hat).detach()
            z_target = (r.unsqueeze(1) + self.args.gamma * q_next_1).detach()

        # 计算当前 Q 网络的 Q 值
        # q_value1 = self.critic_network1(o, u)
        # q_value2 = self.critic_network2(o, u)

        # 计算两个 Q 网络的损失
        # critic_loss1 = torch.mean(F.mse_loss(q_value1, target_q))
        # critic_loss2 = torch.mean(F.mse_loss(q_value2, target_q))

        tau, tau_hat, presum_tau = self.get_tau(o[0], u[0], fp=self.fp)
        z1_pred = self.critic_network1(o, u, tau_hat)
        zf1_loss = self.quantile_regression_loss(z1_pred, z_target, tau_hat, next_presum_tau)#next_presum_tau是target distribution fraction tau_i+1 - tau_i


        # 反向传播和优化
        self.critic_optim1.zero_grad()
        zf1_loss.backward()
        self.critic_optim1.step()

        # self.critic_optim2.zero_grad()
        # critic_loss2.backward()
        # self.critic_optim2.step()

        # 更新损失值
        self.critic_loss1 = zf1_loss.item()
        # self.critic_loss2 = zf1_loss.item()

        self._soft_update_target_network()

        self.train_step += 1

    # def actor_loss(self, o, u):
    #     q1_value = self.critic_network1(o, u)
    #     # q2_value = self.critic_network2(o, u)
    #     actor_loss = torch.mean(q1_value)#这个是正的cost
    #     return actor_loss
    
    def actor_loss(self, o, u, cost_limit, new_tau_hat, new_presum_tau):
        # q1_value = self.critic_network1(o, u).mean()
        z1_new_actions = self.critic_network1(o, u, new_tau_hat)
        q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
        q1_value = q1_new_actions.mean()  # 计算当前动作的平均值

        temp = F.relu(q1_value - cost_limit) - 1  #这个值是  

        if temp < - 1 / (self.ipo_t ** 2):  # 如果cost偏差小于0,进行内点惩罚，还没有超出约束范围
            actor_loss = - torch.log(-temp) / self.ipo_t  # 添加一个小的常数以避免对数为零, 这个应该是负数
        else:  # 如果cost偏差大于0,则惩罚,超出约束范围进行策略惩罚，以引导策略恢复
            actor_loss = self.ipo_t * temp - math.log(1 / (self.ipo_t ** 2)) / self.ipo_t + 1 / self.ipo_t  # 添加一个小的常数以避免对数为零, 这个应该是负数

        # q2_value = self.critic_network2(o, u)
        # actor_loss = torch.mean(q1_value)#这个是正的cost
        return actor_loss

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'cost_critic_agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.critic_network1.state_dict(), model_path + '/' + num + 'cost_critic1_params.pkl')



class Quantile_net:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_q = 0
        self.critic_loss1 = 0
        self.critic_loss2 = 0


        self.tau_type = 'iqn'
        self.num_quantiles = 32
        self.fp = None

        self.critic_network1 = QuantileMlp().to(device)
        self.critic_target_network1 = QuantileMlp().to(device)
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic)

        self.critic_network2 = QuantileMlp().to(device)
        self.critic_target_network2 = QuantileMlp().to(device)
        self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)


        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'critic_agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.critic_target_network1.parameters(), self.critic_network1.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)
        for target_param, param in zip(self.critic_target_network2.parameters(), self.critic_network2.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    def get_tau(self, obs, actions, fp=None):
        if self.tau_type == 'fix':
            presum_tau = ptu.zeros(len(actions), self.num_quantiles) + 1. / self.num_quantiles#len(actions)是用于获取batch_size的大小
        elif self.tau_type == 'iqn':  # add 0.1 to prevent tau getting too close
            presum_tau = ptu.rand(len(actions), self.num_quantiles) + 0.1
            presum_tau /= presum_tau.sum(dim=-1, keepdims=True)
        elif self.tau_type == 'fqf':
            if fp is None:
                fp = self.fp
            presum_tau = fp(obs, actions)
        tau = torch.cumsum(presum_tau, dim=1)  # (N, T), note that they are tau1...tauN in the paper
        with torch.no_grad():
            tau_hat = ptu.zeros_like(tau)
            tau_hat[:, 0:1] = tau[:, 0:1] / 2.
            tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2.
        return tau, tau_hat, presum_tau


    def upda(self, o, u, o_next, u_next, r, entropy, log_alpha, next_tau_hat, next_presum_tau, logger):
        with torch.no_grad():
            # 使用目标网络预测下一个状态的 Q 值
            q_next_1 = self.critic_target_network1(o_next[0], u_next[0], next_tau_hat).detach()
            q_next_2 = self.critic_target_network2(o_next[0], u_next[0], next_tau_hat).detach()
            next_value = torch.min(q_next_1, q_next_2) + log_alpha.exp() * entropy
            z_target = (r.unsqueeze(1) + self.args.gamma * next_value).detach()


        # # time1 = time.time()
        # # 计算当前 Q 网络的 Q 值
        # q_value1 = self.critic_network1(o, u)
        # q_value2 = self.critic_network2(o, u)


        # # 计算两个 Q 网络的损失
        # critic_loss1 = torch.mean(F.mse_loss(q_value1, target_q))
        # critic_loss2 = torch.mean(F.mse_loss(q_value2, target_q))
        tau, tau_hat, presum_tau = self.get_tau(o[0], u[0], fp=self.fp)
        z1_pred = self.critic_network1(o[0], u[0], tau_hat)
        z2_pred = self.critic_network2(o[0], u[0], tau_hat)
        zf1_loss = self.quantile_regression_loss(z1_pred, z_target, tau_hat, next_presum_tau)#next_presum_tau是target distribution fraction tau_i+1 - tau_i
        zf2_loss = self.quantile_regression_loss(z2_pred, z_target, tau_hat, next_presum_tau)#next_presum_tau是target distribution fraction tau_i+1 - tau_i


        # 反向传播和优化
        self.critic_optim1.zero_grad()
        zf1_loss.backward()
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        zf2_loss.backward()
        self.critic_optim2.step()

        # 更新损失值
        self.critic_loss1 = zf1_loss.item()
        self.critic_loss2 = zf2_loss.item()

        self._soft_update_target_network()

        self.train_step += 1


    def actor_loss(self, o, u, entropy, log_alpha, new_tau_hat, new_presum_tau):
        # q1_value = self.critic_network1(o, u)
        # q2_value = self.critic_network2(o, u)
        z1_new_actions = self.critic_network1(o[0], u[0], new_tau_hat)
        z2_new_actions = self.critic_network2(o[0], u[0], new_tau_hat)

        q1_new_actions = torch.sum(new_presum_tau * z1_new_actions, dim=1, keepdims=True)
        q2_new_actions = torch.sum(new_presum_tau * z2_new_actions, dim=1, keepdims=True)

        actor_loss = torch.mean(-log_alpha.exp() * entropy -
                                torch.min(q1_new_actions, q2_new_actions))
        return actor_loss

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'critic_agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.critic_network1.state_dict(), model_path + '/' + num + '_critic1_params.pkl')

    def quantile_regression_loss(self, input, target, tau, weight):
        """
        input: (N, T)
        target: (N, T)
        tau: (N, T)
        输入都是tensor
        """
        input = input.unsqueeze(-1) # (N, T, 1)
        target = target.detach().unsqueeze(-2) # (N, 1, T)
        tau = tau.detach().unsqueeze(-1) # (N, T, 1)
        weight = weight.detach().unsqueeze(-2) # (N, 1, T)
        expanded_input, expanded_target = torch.broadcast_tensors(input, target) #广播到(N,T,T)形状
        L = F.smooth_l1_loss(expanded_input, expanded_target, reduction="none")  # (N, T, T)
        sign = torch.sign(expanded_input - expanded_target) / 2. + 0.5
        rho = torch.abs(tau - sign) * L * weight
        return rho.sum(dim=-1).mean()
        """最后一个维度是目标分位点索引j,对它求和是为了每个预测分位i聚合其与所有目标分位之间的损失,符合分位数回归的理论定义"""