import torch
import os
import numpy as np
from torch.distributions import Normal
import torch.nn.functional as F
from maddpg.actor_critic import Actor, Critic, Cost_critic
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(122)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_q = 0
        self.actor_loss = 0


        self.cost_limit = 0
        self.alpha = 0.02

        self.ipo_t = 20 #  # 内点惩罚的温度参数,这个参数可以调节内点惩罚的强度
        self.rev = 1 #用于当约束被违反的时候，引导策略恢复


        self.cost_limit_update = self.cost_limit  # 用于更新cost limit的值, 这是一种可能

        "PPO算法新增内容"
        self.horizon_size = args.horizon_size
        self.K_epochs = args.K_epochs
        self.batch_size = args.batch_size
        self.clip_param = args.clip_param
        self.entropy_coefficient = args.entropy_coefficient
        self.huber_delta = args.huber_delta

        # create the network
        self.actor_network = Actor(args, agent_id).to(device)
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor, eps=1e-5)
        #这个eps是为了增加训练的稳定性

        # "safe ppo算法新增内容"
        # self.lagrangian = torch.tensor(1.0, requires_grad=True)
        # self.lagrangian_optimizer = torch.optim.Adam([self.lagrangian], lr=self.args.lagrangian_lr)

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


    def _huber_loss(self, e, d):
        a = (abs(e) <= d).float()
        b = (abs(e) > d).float()
        return a*e**2/2 + b*d*(abs(e)-d/2)

    def train_new(self, transitions, critic_net, cost_net, logger=None):

        #首先把所有的transitions转换为tensor
        for key in transitions.keys():
            transitions[key] = torch.as_tensor(transitions[key], dtype=torch.float32).to(device)

        o = transitions['o_%d' % self.agent_id]  # 当前agent的观测 [horizon_size × state_shape]
        o_next = transitions['o_next_%d' % self.agent_id]  # 当前agent的下一个观测 [horizon_size × state_shape]
        action = transitions['u_%d' % self.agent_id]  # 当前agent的动作 [horizon_size × action_shape]
        action_log_pi_old = transitions['action_log_pi_%d' % self.agent_id]  # 当前agent的动作对数概率 [horizon_size × action_shape]

        reward =  transitions['reward_%d' % self.agent_id].unsqueeze(1) # 当前agent的reward [horizon_size]通过unsqueeze操作变为 [horizon_size × 1]
        cost =  transitions['cost_%d' % self.agent_id].unsqueeze(1) # 当前agent的reward [horizon_size]通过unsqueeze操作变为 [horizon_size × 1]
        
        done = transitions['done_%d' % self.agent_id].unsqueeze(1) # 当前agent的done标志 [horizon_size]通过unsqueeze操作变为 [horizon_size × 1]
        

        #计算reward对应的advantage和目标值v_target
        with torch.no_grad():  # adv and v_target have no gradient
            adv = torch.zeros(self.horizon_size, 1, device=device)
            gae = 0

            vs = critic_net.critic_network1(o) # critic的值的维度为 [horizon_size × 1]
            vs_next = critic_net.critic_network1(o_next) # 下一时刻的状态对应的V值的形状为 [horizon_size × 1]

            td_delta = reward + self.args.gamma * vs_next * (1 - done) - vs  # 计算TD误差 [horizon_size × 1]

            # 计算GAE
            for i in reversed(range(self.horizon_size)):
                gae = td_delta[i] + self.args.gamma * self.args.lmbda * gae * (1.0 - done[i]) #dim=N,返回shape为N的一维tensor
                adv[i] = gae #dim=N,返回shape为N的一维tensor

            v_target = adv + vs  # [horizon_size, 1]
            #advantage 标准化（advantage normalization）
            adv = (adv - adv.mean()) / (adv.std() + 1e-8) #[horizon_size, 1]


        #计算cost对应的advantage和目标值v_target
        with torch.no_grad():  # adv and v_target have no gradient
            adv_cost = torch.zeros(self.horizon_size, 1, device=device)
            gae_cost = 0

            vs_cost = cost_net.cost_network(o) # critic的值的维度为 [horizon_size × 1]
            vs_next_cost = cost_net.cost_network(o_next) # 下一时刻的状态对应的V值的形状为 [horizon_size × 1]

            td_delta_cost = cost + self.args.gamma * vs_next_cost * (1 - done) - vs_cost  # 计算TD误差 [horizon_size × 1]

            # 计算GAE
            for i in reversed(range(self.horizon_size)):
                gae_cost = td_delta_cost[i] + self.args.gamma * self.args.lmbda * gae_cost * (1.0 - done[i]) #dim=N,返回shape为N的一维tensor
                adv_cost[i] = gae_cost #dim=N,返回shape为N的一维tensor

            v_target_cost = adv_cost + vs_cost  # [horizon_size, 1]
            #advantage 标准化（advantage normalization）
            adv_cost = (adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-8) #[horizon_size, 1]
            
            
            #对cost的偏差进行归一化处理
            cost_mean = cost.mean().item() # 输出是一个 Python float

            # self.cost_limit_update = max(self.cost_limit, cost_mean + self.alpha * self.cost_limit )


            # Jc = (((cost_mean - self.cost_limit_update) * (1 - self.args.gamma)) +  adv_cost.mean()) / (adv_cost.std() + 1e-8) # 计算当前的cost偏差,这是一个标量

            
            # self.cost_limit = cost_limit
        "上方是PPO算法中关于目标值的计算部分, 这部分的tensor不能有梯度"


        # # cost_mean = adv_cost.detach().mean()#这个地方用cost网络的adv值进行计算，可能有点不合适啊
        # with torch.no_grad():
        #     cost_mean = cost.mean().item() # 输出是一个 Python float
        # lagrangian_loss = -self.lagrangian * (cost_mean - self.cost_limit)

        # self.lagrangian_optimizer.zero_grad()
        # lagrangian_loss.backward()
        # self.lagrangian_optimizer.step()
        # # 确保 self.lagrangian 大于等于0
        # with torch.no_grad():
        #     self.lagrangian.data.clamp_(min=0.0)  # ✅ 安全保留梯度追踪





        for _ in range(self.K_epochs): 
            # 随机打乱样本 并 生成小批量
            shuffled_indices = np.random.permutation(self.horizon_size)
            indexes = [shuffled_indices[i:i + self.batch_size] for i in range(0, self.horizon_size, self.batch_size)]
            # 步骤 | 描述
            # 第 1 行 | 打乱所有样本的索引顺序
            # 第 2 行 | 将打乱后的索引按 minibatch_size 分组
            # 输出 | 一个列表，每个元素是一个小批量的 sample 索引，后续用于 SGD，这个indexes其实是一个列表，里面的每个元素都是一个小批量的索引
            for index in indexes:
                mean, std = self.actor_network(o[index])  # mini_batch_size x state_shape 转换为 mini_batch_size x action_dim
                dist_now = Normal(mean, std)
                dist_entropy = dist_now.entropy().sum(dim = 1, keepdim=True)  # mini_batch_size x action_dim -> mini_batch_size x 1
                action_log_pi_now = dist_now.log_prob(action[index]) # mini_batch_size x action_dim


                ratios = torch.exp(action_log_pi_now.sum(dim = 1, keepdim=True) - action_log_pi_old[index].sum(dim = 1, keepdim=True))  # shape(mini_batch_size X 1)


                surr1 = ratios * (adv[index].detach()) #shape(mini_batch_size X 1)
                surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * (adv[index].detach()) #shape(mini_batch_size X 1)

                #reward对应的actor损失函数
                actor_loss_reward = -torch.min(surr1, surr2).mean() - self.entropy_coefficient * dist_entropy.mean()


                #cost对应的actor损失函数
                # actor_loss_cost = ratios * (adv_cost[index].detach()) # shape(mini_batch_size X 1)
                # surr_cost1 = ratios * (adv_cost[index].detach())  # shape(mini_batch_size X 1)
                # surr_cost2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * (adv_cost[index].detach())
                # surr_cadv = torch.max(surr_cost1, surr_cost2).mean()  # shape(mini_batch_size X 1)
                
                # temp = surr_cadv + Jc


                # Jc = (((cost_mean - self.cost_limit_update) * (1 - self.args.gamma)) +  adv_cost.mean()) / (adv_cost.std() + 1e-8) # 计算当前的cost偏差,这是一个标量
                temp = self.cost_limit - (cost_mean + (adv_cost[index].mean()) / (1 - self.args.gamma))  # 计算当前的cost偏差,这是一个标量

                # actor_loss_cost = - torch.log(self.self.cost_limit - (cost_mean + (adv_cost[index].mean()) / (1 - self.args.gamma))) / self.ipo_t

                if (temp + 1) < 0:  # 如果cost偏差小于0,进行内点惩罚，还没有超出约束范围
                    actor_loss_cost = - torch.log(-temp + 1) / self.ipo_t  # 添加一个小的常数以避免对数为零, 这个应该是负数
                else:  # 如果cost偏差大于0,则惩罚,超出约束范围进行策略惩罚，以引导策略恢复
                    surr_cost1 = ratios * (adv_cost[index].detach())  # shape(mini_batch_size X 1)
                    surr_cost2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * (adv_cost[index].detach())
                    surr_cadv = torch.max(surr_cost1, surr_cost2).mean()  # shape(mini_batch_size X 1)
                    actor_loss_cost = self.rev * surr_cadv

                actor_loss = actor_loss_reward + actor_loss_cost


                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_network.parameters(), 0.5)#梯度裁剪操作
                self.actor_optim.step()

                v_target_min_batch = v_target[index].detach()  # mini_batch_size x 1
                vs_min_batch = critic_net.critic_network1(o[index])  # mini_batch_size x 1
                "计算critic的损失函数, 这个地方或许可以进行更改"
                v_target_clip = torch.clamp(v_target_min_batch, vs_min_batch - self.clip_param, vs_min_batch + self.clip_param)
                # if self.trick['huber_loss']:
                critic_loss_clip = self._huber_loss(v_target_clip - vs_min_batch, self.huber_delta).mean()#裁剪过后的目标值
                critic_loss_original = self._huber_loss(v_target_min_batch - vs_min_batch, self.huber_delta).mean()#未裁剪的目标值
                critic_loss = torch.max(critic_loss_original, critic_loss_clip)
                critic_net.critic_optim1.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_net.critic_network1.parameters(), 0.5)  # 梯度裁剪操作
                critic_net.critic_optim1.step()  # 更新critic网络


                cost_target_min_batch = v_target_cost[index].detach()  # mini_batch_size x 1
                cost_min_batch = cost_net.cost_network(o[index])  # mini_batch_size x 1
                "计算critic的损失函数, 这个地方或许可以进行更改"
                cost_target_clip = torch.clamp(cost_target_min_batch, cost_min_batch - self.clip_param, cost_min_batch + self.clip_param)
                # if self.trick['huber_loss']:
                cost_loss_clip = self._huber_loss(cost_target_clip - cost_min_batch, self.huber_delta).mean()#裁剪过后的目标值
                cost_loss_original = self._huber_loss(cost_target_min_batch - cost_min_batch, self.huber_delta).mean()#未裁剪的目标值
                cost_loss = torch.max(cost_loss_original, cost_loss_clip)
                cost_net.cost_optim.zero_grad()
                cost_loss.backward()
                torch.nn.utils.clip_grad_norm_(cost_net.cost_network.parameters(), 0.5)  # 梯度裁剪操作
                cost_net.cost_optim.step()  # 更新cost网络

        # # with torch.no_grad():
        # cost_mean = adv_cost.detach().mean()#这个地方用cost网络的adv值进行计算，可能有点不合适啊
        # lagrangian_loss = -self.lagrangian * (cost_mean - self.cost_limit)

        # self.lagrangian_optimizer.zero_grad()
        # lagrangian_loss.backward()
        # self.lagrangian_optimizer.step()
        # # 确保 self.lagrangian 大于等于0
        # with torch.no_grad():
        #     self.lagrangian.data.clamp_(min=0.0)  # ✅ 安全保留梯度追踪


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

        self.critic_network1 = Critic().to(device)
        self.critic_target_network1 = Critic().to(device)
        self.critic_target_network1.load_state_dict(self.critic_network1.state_dict())
        self.critic_optim1 = torch.optim.Adam(self.critic_network1.parameters(), lr=self.args.lr_critic, eps=1e-5)
        #eps=1e-5是为了增加训练的稳定性

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

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'critic_agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.critic_network1.state_dict(), model_path + '/' + num + '_critic1_params.pkl')


class COST_NET:
    def __init__(self, args, agent_id):  # 因为不同的agent的obs、act维度可能不一样，所以神经网络不同,需要agent_id来区分
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0
        self.last_c = 0
        self.cost_loss = 0

        self.cost_network = Cost_critic().to(device)
        self.cost_target_network = Cost_critic().to(device)
        self.cost_target_network.load_state_dict(self.cost_network.state_dict())
        self.cost_optim = torch.optim.Adam(self.cost_network.parameters(), lr=self.args.lr_critic, eps=1e-5)
        #eps=1e-5是为了增加训练的稳定性

        # create the dict for store the model
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # path to save the model
        self.model_path = self.args.save_dir + '/' + self.args.scenario_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.model_path = self.model_path + '/' + 'cost_agent_%d' % agent_id
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        model_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, 'cost_agent_%d' % self.agent_id)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.critic_network1.state_dict(), model_path + '/' + num + '_cost_params.pkl')
