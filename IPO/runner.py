from tqdm import tqdm
from agent import Agent
from maddpg.maddpg import CRITIC_NET, COST_NET
# from maddpg.maddpg import GAT_NET
from common.replay_buffer import Buffer
import torch
import copy
import os
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
# from normalization import Normalization
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(122)
# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练的类
class Runner:
    def __init__(self, args, env):
        self.args = args
        # self.noise = args.noise_rate
        # self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env

        self.n_agents = 5
        self.max_reward = -10000
        self.agents = self._init_agents()
        self.critic_net = self._init_critic()
        self.cost_net = self._init_cost()

        # #实例化每个agent的归一化类
        # self.obs_norm = self._init_obs_norm()

        self.buffer = Buffer(args)
        self.save_path = os.path.join(self.args.save_dir, self.args.scenario_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _init_agents(self):
        agents = []
        for i in range(self.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def _init_critic(self):
        agents = []
        for i in range(self.n_agents):
            agent = CRITIC_NET(self.args, i)
            agents.append(agent)
        return agents

    def _init_cost(self):
        agents = []
        for i in range(self.n_agents):
            agent = COST_NET(self.args, i)
            agents.append(agent)
        return agents
    

    def run_tensor(self):
        decline = 1
        model_dir = Path('./models')
        if not model_dir.exists():
            run_num = 1
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            run_num = max(exst_run_nums, default=0) + 1

        curr_run = 'run%i' % run_num
        run_dir = model_dir / curr_run
        log_dir = run_dir / 'logs'
        os.makedirs(log_dir)
        logger = SummaryWriter(str(log_dir))

        # 开始训练
        reward_hist = []
        hist = []
        hist_local = []
        hist_global = []
        hist_violate = []
        hist_violate_count = []
        hist_total_cost = []
        hist_total_cost_violate_sum_average = []
        hist_total_cost_count_sum_probability = []
        st = time.time()
        done = 0

        for episode in tqdm(range(self.args.max_episode)):
            print(episode)
            reward_list = []
            agent_reward = [0] * self.n_agents
            agent_reward_local = [0] * self.n_agents
            agent_reward_global = [0] * self.n_agents
            agent_cost_violate_sum = [0] * self.n_agents
            agent_cost_count_sum = [0] * self.n_agents
            agent_total_cost = [0]
            agent_total_cost_violate_sum_average = [0]
            agent_total_cost_count_sum_probability = [0]
            done_flag_runner = 0
            s, week, hour, PPD = self.env.reset()
            # s = [self.obs_norm[i](s[i]) for i in range(self.n_agents)]
            time_step = 0

            occupy_count_1 = 0
            occupy_count_2 = 0
            occupy_count_3 = 0
            occupy_count_4 = 0
            occupy_count_5 = 0

            while not done_flag_runner:
                time_step += 1
                u = []
                u_buffer = []
                actions = []
                action_log_pi_list = []

                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action, action_log_pi, action_buffer = agent.select_action(s[agent_id])
                        u.append(action)
                        action_log_pi_list.append(action_log_pi)
                        actions.append(action)
                        u_buffer.append(action_buffer)

                if s[0][9] != 0:
                    occupy_count_1 += 1
                if s[1][9] != 0:
                    occupy_count_2 += 1
                if s[2][9] != 0:
                    occupy_count_3 += 1
                if s[3][9] != 0:
                    occupy_count_4 += 1
                if s[4][9] != 0:
                    occupy_count_5 += 1
                        
                # actions = [a.squeeze(0) for a in actions]
                # print("actions:", actions)
                # s_next, r, done_flag, week, hour, PPD, r_local, r_global = self.env.step(actions)
                s_next, r, r_local, r_global, done_flag, PPD, cost_violate_sum, cost_violate_count  = self.env.step(actions)

                done_flag_runner = done_flag[0]  # Assuming all agents share the same done flag

                self.buffer.store_episode(s[:self.n_agents], u_buffer, r[:self.n_agents], s_next[:self.n_agents], action_log_pi_list, done_flag[:self.n_agents]
                                            , r_local[:self.n_agents], r_global[:self.n_agents])#存储每个agent的经验
                s = s_next
                reward = np.array(r)

                for i in range(self.n_agents):
                    agent_reward[i] += r[i]
                for i in range(self.n_agents):
                    agent_reward_local[i] += r_local[i]
                for i in range(self.n_agents):
                    agent_reward_global[i] += r_global[i]


                for i in range(self.n_agents):
                    agent_cost_violate_sum[i] += cost_violate_sum[i]
                for i in range(self.n_agents):
                    agent_cost_count_sum[i] += cost_violate_count[i]


                agent_total_cost[0] += np.sum(r_local)
                agent_total_cost_violate_sum_average[0] += np.sum(cost_violate_sum)
                agent_total_cost_count_sum_probability[0] += np.sum(cost_violate_count)

                reward_list.append(np.sum(reward))

                
                if time_step % self.args.horizon_size == 0:
                    start_time = time.time()
                    transitions = self.buffer.sample_all()#把buffer里面的所有数据都取出来
                    for agent_id, agent in enumerate(self.agents):
                        # other_agents = self.agents.copy()
                        # other_agents.remove(agent)
                        id = agent_id
                        agent.learn(transitions, self.critic_net[id], self.cost_net[id], logger)
                    self.buffer.clear()  # 清空buffer
                    end_time = time.time()
                    print("Time taken for learning step: %.2f seconds" % (end_time - start_time))
            reward = np.sum(reward_list) / self.episode_limit * decline
            reward_hist.append(reward)
            if reward > self.max_reward:
                self.max_reward = reward

            ave_r = [0] * 5
            for i in range(5):
                ave_r[i] = agent_reward[i]
            print("Episode:%d r1:%.2f r2:%.2f r3:%.2f r4:%.2f r5:%.2f  av_r:%.3f max_av_r:%.3f Time: %.1fs" %
                  (episode, agent_reward[0], agent_reward[1], agent_reward[2], agent_reward[3], agent_reward[4],
                   reward,
                   self.max_reward, time.time() - st))
            st = time.time()
            agent_cost_violate_sum[0] = agent_cost_violate_sum[0] / occupy_count_1
            agent_cost_violate_sum[1] = agent_cost_violate_sum[1] / occupy_count_2
            agent_cost_violate_sum[2] = agent_cost_violate_sum[2] / occupy_count_3
            agent_cost_violate_sum[3] = agent_cost_violate_sum[3] / occupy_count_4
            agent_cost_violate_sum[4] = agent_cost_violate_sum[4] / occupy_count_5


            hist.append(agent_reward)
            hist_local.append(agent_reward_local)
            hist_global.append(agent_reward_global)
            hist_violate.append(agent_cost_violate_sum)
            hist_violate_count.append(agent_cost_count_sum)
            hist_total_cost.append(agent_total_cost)
            agent_total_cost_violate_sum_average[0] = agent_total_cost_violate_sum_average[0] / (occupy_count_1 + occupy_count_2 + occupy_count_3 + occupy_count_4 + occupy_count_5)
            hist_total_cost_violate_sum_average.append(agent_total_cost_violate_sum_average)
            agent_total_cost_count_sum_probability[0] = agent_total_cost_count_sum_probability[0] / (occupy_count_1 + occupy_count_2 + occupy_count_3 + occupy_count_4 + occupy_count_5)
            hist_total_cost_count_sum_probability.append(agent_total_cost_count_sum_probability)
            
            total_reward = agent_reward_global[0] + agent_reward_global[1] + agent_reward_global[2] + agent_reward_global[3] + agent_reward_global[4]
            total_cost = agent_reward_local[0] + agent_reward_local[1] + agent_reward_local[2] + agent_reward_local[3] + agent_reward_local[4]
            total_cost_violate_sum_average = (agent_cost_violate_sum[0] + agent_cost_violate_sum[1] + agent_cost_violate_sum[2] + agent_cost_violate_sum[3] + agent_cost_violate_sum[4]) / 5
            total_cost_count_sum_probability = (agent_cost_count_sum[0] + agent_cost_count_sum[1] + agent_cost_count_sum[2] + agent_cost_count_sum[3] + agent_cost_count_sum[4]) / (occupy_count_1 + occupy_count_2 + occupy_count_3 + occupy_count_4 + occupy_count_5)
            to_log1 = {
                "total_reward": total_reward,
                "total_cost": total_cost,
                "total_cost_violate_sum_average": total_cost_violate_sum_average,
                "total_cost_count_sum_probability": total_cost_count_sum_probability,
                "agent_reward_1": agent_reward_global[0],
                "agent_reward_2": agent_reward_global[1],
                "agent_reward_3": agent_reward_global[2],
                "agent_reward_4": agent_reward_global[3],
                "agent_reward_5": agent_reward_global[4],
                "agent_cost_1": agent_reward_local[0],
                "agent_cost_2": agent_reward_local[1],
                "agent_cost_3": agent_reward_local[2],
                "agent_cost_4": agent_reward_local[3],
                "agent_cost_5": agent_reward_local[4],
                "agent_cost_violate_sum_1_average": agent_cost_violate_sum[0],
                "agent_cost_violate_sum_2_average": agent_cost_violate_sum[1],
                "agent_cost_violate_sum_3_average": agent_cost_violate_sum[2],
                "agent_cost_violate_sum_4_average": agent_cost_violate_sum[3],
                "agent_cost_violate_sum_5_average": agent_cost_violate_sum[4],
                "agent_violate_count_1_probability": agent_cost_count_sum[0] / occupy_count_1,
                "agent_violate_count_2_probability": agent_cost_count_sum[1] / occupy_count_2,
                "agent_violate_count_3_probability": agent_cost_count_sum[2] / occupy_count_3,
                "agent_violate_count_4_probability": agent_cost_count_sum[3] / occupy_count_4,
                "agent_violate_count_5_probability": agent_cost_count_sum[4] / occupy_count_5,
                "episode:": episode,
            }

            self.env.swanlab.log(to_log1, step=episode)
            if episode % 10 == 0:
                # df2 = pd.DataFrame(hist)
                df3 = pd.DataFrame(hist_local)
                df4 = pd.DataFrame(hist_global)
                df5 = pd.DataFrame(hist_violate)
                df6 = pd.DataFrame(hist_violate_count)
                df7 = pd.DataFrame(hist_total_cost)
                df8 = pd.DataFrame(hist_total_cost_violate_sum_average)
                df9 = pd.DataFrame(hist_total_cost_count_sum_probability)
                # df2.to_excel('reward.xlsx', index=False)
                df3.to_excel('reward_local.xlsx', index=False)
                df4.to_excel('reward_global.xlsx', index=False)
                df5.to_excel('PPD_violate_average.xlsx', index=False)
                df6.to_excel('PPD_violate_count.xlsx', index=False)
                df7.to_excel('total_cost.xlsx', index=False)
                df8.to_excel('hist_total_cost_violate_sum_average.xlsx', index=False)
                df9.to_excel('hist_total_cost_count_sum_probability.xlsx', index=False)

        # df2 = pd.DataFrame(hist)
        df3 = pd.DataFrame(hist_local)
        df4 = pd.DataFrame(hist_global)
        df5 = pd.DataFrame(hist_violate)
        df6 = pd.DataFrame(hist_violate_count)
        df7 = pd.DataFrame(hist_total_cost)
        df8 = pd.DataFrame(hist_total_cost_violate_sum_average)
        df9 = pd.DataFrame(hist_total_cost_count_sum_probability)
        # df2.to_excel('reward.xlsx', index=False)
        df3.to_excel('reward_local.xlsx', index=False)
        df4.to_excel('reward_global.xlsx', index=False)
        df5.to_excel('PPD_violate_average.xlsx', index=False)
        df6.to_excel('PPD_violate_count.xlsx', index=False)
        df7.to_excel('total_cost.xlsx', index=False)
        df8.to_excel('hist_total_cost_violate_sum_average.xlsx', index=False)
        df9.to_excel('hist_total_cost_count_sum_probability.xlsx', index=False)

        plt.plot(reward_hist)
        plt.show()
        time.sleep(1)
        logger.close()

