import torch
import os
from maddpg.actor_critic import Actor, Critic
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(52)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_q = 0
        self.actor_loss = 0

        "MATD3新增内容"
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.max_action = args.max_action
        self.actor_pointer = 0
        self.policy_update_freq = args.policy_update_freq

        # create the network
        self.actor_network = Actor(args, agent_id).to(device)
        # build up the target network
        self.actor_target_network = Actor(args, agent_id).to(device)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)

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

    # soft update
    def _soft_update_target_network(self):
        for target_param, param in zip(self.actor_target_network.parameters(), self.actor_network.parameters()):
            target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # MADDPG
    def train_new(self, transitions, other_agents, critic_net, logger=None, n_agents=5):
        self.actor_pointer += 1
        # 重点更新网络参数代码
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32).to(device)
        r = transitions['r_%d' % self.agent_id]  # 训练时只需要自己的reward
        o, u, o_next = [], [], []  # 用来装每个agent经验中的各项

        for agent_id in range(n_agents):
            o.append(transitions['o_%d' % agent_id])
            u.append(transitions['u_%d' % agent_id])
            o_next.append(transitions['o_next_%d' % agent_id])  # o现在是一个3维的数组，维度为[agent_num,batch_size,state_shape]
        u_next = []

        with torch.no_grad():
            index = 0
            for agent_id in range(n_agents):
                if agent_id == self.agent_id:
                    batch_u_next = self.actor_target_network(o_next[self.agent_id])
                    # noise = (torch.randn_like(batch_u_next) * self.policy_noise).clamp(-self.noise_clip,
                    #                                                                    self.noise_clip)
                    # batch_u_next = (batch_u_next + noise).clamp(-self.max_action, self.max_action)
                    u_next.append(batch_u_next)
                else:
                    batch_u_next = other_agents[index].policy.actor_target_network(o_next[agent_id])
                    # noise = (torch.randn_like(batch_u_next) * self.policy_noise).clamp(-self.noise_clip,
                    #                                                                    self.noise_clip)
                    # batch_u_next = (batch_u_next + noise).clamp(-self.max_action, self.max_action)
                    u_next.append(batch_u_next)
                    index += 1

        o_aggregation0 = torch.stack((o[0], o[1], o[2], o[3], o[4]), dim=1)
        u_aggregation0 = torch.stack((u[0], u[1], u[2], u[3], u[4]), dim=1)
        # 形状为batch_size, num_agnets, features
        o_aggregation_next0 = torch.stack((o_next[0], o_next[1], o_next[2], o_next[3], o_next[4]), dim=1)
        u_aggregation_next1 = torch.stack((u_next[0], u_next[1], u_next[2], u_next[3], u_next[4]), dim=1)

        critic_net.upda(o_aggregation0, u_aggregation0, o_aggregation_next0, u_aggregation_next1, r, logger)

        # if self.actor_pointer % self.policy_update_freq == 0:
        u[self.agent_id] = self.actor_network(o[self.agent_id])
        u_aggregation1 = torch.stack((u[0], u[1], u[2], u[3], u[4]), dim=1)
        actor_loss = critic_net.actor_loss(o_aggregation0, u_aggregation1)
        self.actor_loss = actor_loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self._soft_update_target_network()

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

        # self.critic_network2 = Critic().to(device)
        # self.critic_target_network2 = Critic().to(device)
        # self.critic_target_network2.load_state_dict(self.critic_network2.state_dict())
        # self.critic_optim2 = torch.optim.Adam(self.critic_network2.parameters(), lr=self.args.lr_critic)

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
        # for target_param, param in zip(self.critic_target_network2.parameters(), self.critic_network2.parameters()):
        #     target_param.data.copy_((1 - self.args.tau) * target_param.data + self.args.tau * param.data)

    # def upda(self, o, u, o_next, u_next, r, logger):
    #     with torch.no_grad():
    #         q_next = self.critic_target_network(o_next, u_next).detach()
    #         target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()
    #     q_value = self.critic_network(o, u)
    #     critic_loss = (target_q - q_value).pow(2).mean()
    #     self.critic_loss = critic_loss
    #     self.critic_optim.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optim.step()
    #
    #     self._soft_update_target_network()
    #     if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
    #         # 定期保存模型参数
    #         self.save_model(self.train_step)
    #
    #     self.train_step += 1

    def upda(self, o, u, o_next, u_next, r, logger):
        with torch.no_grad():
            # 使用目标网络预测下一个状态的 Q 值
            q_next = self.critic_target_network1(o_next, u_next).detach()
            # q_next2 = self.critic_target_network2(o_next, u_next).detach()
            # q_next = torch.min(q_next1, q_next2)  # 使用 Clipped Double Q-Learning
            # 计算目标 Q 值
            target_q = (r.unsqueeze(1) + self.args.gamma * q_next).detach()

        # 计算当前 Q 网络的 Q 值
        q_value1 = self.critic_network1(o, u)
        # q_value2 = self.critic_network2(o, u)

        # 计算两个 Q 网络的损失
        critic_loss1 = (target_q - q_value1).pow(2).mean()
        # critic_loss2 = (target_q - q_value2).pow(2).mean()

        # 反向传播和优化
        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        # self.critic_optim2.zero_grad()
        # critic_loss2.backward()
        # self.critic_optim2.step()

        # 更新损失值
        self.critic_loss1 = critic_loss1.item()
        # self.critic_loss2 = critic_loss2.item()

        self._soft_update_target_network()

        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            # 定期保存模型参数
            self.save_model(self.train_step)

        self.train_step += 1

    def actor_loss(self, o, u):
        actor_loss = - self.critic_network1(o, u).mean()
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
        # torch.save(self.critic_network2.state_dict(), model_path + '/' + num + '_critic1_params.pkl')

    # def critic_state_dict(self):
    #     return self.critic_network1.state_dict()
