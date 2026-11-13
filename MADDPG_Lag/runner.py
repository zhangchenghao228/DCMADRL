from tqdm import tqdm
from agent import Agent
from maddpg.maddpg import CRITIC_NET, CRITIC_NET_GLOBAL
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
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(22)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 训练的类
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env

        self.n_agents = 5
        self.max_reward = -10000
        self.agents = self._init_agents()
        self.critic_net = self._init_critic()
        self.global_critic = self._init_global_critic()  # 这个是对应的全局奖励的critic网络

        self.buffer = Buffer(args)
        self.save_path = self.args.save_dir + '/' + self.args.scenario_name
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

    def _init_global_critic(self):
        agents = []
        for i in range(self.n_agents):
            agent = CRITIC_NET_GLOBAL(self.args, i)
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
        hist_total_violate_count = []
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
            agent_total_violate_count = [0]
            done_flag = 0
            s, PPD = self.env.reset()
            time_step = 0

            occupy_count_1 = 0
            occupy_count_2 = 0
            occupy_count_3 = 0
            occupy_count_4 = 0
            occupy_count_5 = 0

            while not done_flag:
                time_step += 1
                u = []
                actions = []

                with torch.no_grad():
                    for agent_id, agent in enumerate(self.agents):
                        action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                        u.append(action)
                        actions.append(action)

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

                s_next, r, r_local, r_global, done_flag, PPD, cost_violate_sum, cost_violate_count = self.env.step(actions)
                self.buffer.store_episode(s[:self.n_agents], u, r_local[:self.n_agents], r_global[:self.n_agents],
                                          s_next[:self.n_agents])
                s = s_next
                r_local_array = np.array(r_local)
                r_global_array = np.array(r_global)
                r = r_local_array + r_global_array

                reward = r

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
                agent_total_violate_count[0] += np.sum(cost_violate_count)



                reward_list.append(np.sum(reward))
                start_time = time.time()
                if (time_step + 1) % 4 == 0 and self.buffer.current_size >= 256:
                    for agent_id, agent in enumerate(self.agents):
                        transitions = self.buffer.sample(
                            self.args.batch_size)
                        other_agents = self.agents.copy()
                        other_agents.remove(agent)
                        id = agent_id
                        agent.learn(transitions, other_agents, self.critic_net[id],
                                    self.global_critic[id], logger)
                end_time = time.time()
                print('--------------------------------------------------------')
                print('learn_time: ', end_time - start_time)
                print('--------------------------------------------------------')
                # to_log3 = {"global_critic1_loss1": self.global_critic[0].global_loss1,
                #            "global_critic1_loss2": self.global_critic[1].global_loss1,
                #            "global_critic1_loss3": self.global_critic[2].global_loss1,
                #            "global_critic1_loss4": self.global_critic[3].global_loss1,
                #            "global_critic1_loss5": self.global_critic[4].global_loss1,
                #            # "global_critic1_loss6": self.global_critic[5].global_loss1,
                #            "global_critic2_loss1": self.global_critic[0].global_loss2,
                #            "global_critic2_loss2": self.global_critic[1].global_loss2,
                #            "global_critic2_loss3": self.global_critic[2].global_loss2,
                #            "global_critic2_loss4": self.global_critic[3].global_loss2,
                #            "global_critic2_loss5": self.global_critic[4].global_loss2,
                #            # "global_critic2_loss6": self.global_critic[5].global_loss2,
                #            "local_critic1_loss_1": self.critic_net[0].local_loss1,
                #            "local_critic1_loss_2": self.critic_net[1].local_loss1,
                #            "local_critic1_loss_3": self.critic_net[2].local_loss1,
                #            "local_critic1_loss_4": self.critic_net[3].local_loss1,
                #            "local_critic1_loss_5": self.critic_net[4].local_loss1,
                #            "local_critic2_loss_1": self.critic_net[0].local_loss2,
                #            "local_critic2_loss_2": self.critic_net[1].local_loss2,
                #            "local_critic2_loss_3": self.critic_net[2].local_loss2,
                #            "local_critic2_loss_4": self.critic_net[3].local_loss2,
                #            "local_critic2_loss_5": self.critic_net[4].local_loss2,
                #            "global_actor_loss1": self.agents[0].policy.global_actor_loss,
                #            "global_actor_loss2": self.agents[1].policy.global_actor_loss,
                #            "global_actor_loss3": self.agents[2].policy.global_actor_loss,
                #            "global_actor_loss4": self.agents[3].policy.global_actor_loss,
                #            "global_actor_loss5": self.agents[4].policy.global_actor_loss,
                #            "local_actor_loss1": self.agents[0].policy.local_actor_loss,
                #            "local_actor_loss2": self.agents[1].policy.local_actor_loss,
                #            "local_actor_loss3": self.agents[2].policy.local_actor_loss,
                #            "local_actor_loss4": self.agents[3].policy.local_actor_loss,
                #            "local_actor_loss5": self.agents[4].policy.local_actor_loss,
                #            }
                # self.env.wandb.log(to_log3, step=time_step + episode * 8736)
            self.noise = max(0.05, self.noise - 0.0000005)
            self.epsilon = max(0.05, self.epsilon - 0.0000005)

            reward = np.sum(reward_list) / self.episode_limit * decline
            reward_hist.append(reward)
            if reward > self.max_reward:
                self.max_reward = reward

            ave_r = [0] * self.n_agents
            for i in range(self.n_agents):
                ave_r[i] = agent_reward[i]

            print("Episode:%d r1:%.2f r2:%.2f r3:%.2f r4:%.2f r5:%.2f av_r:%.3f max_av_r:%.3f Time: %.1fs" %
                  (episode, agent_reward[0], agent_reward[1], agent_reward[2], agent_reward[3], agent_reward[4], reward,
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
            agent_total_cost_violate_sum_average[0] = agent_total_cost_violate_sum_average[0] / (
                        occupy_count_1 + occupy_count_2 + occupy_count_3 + occupy_count_4 + occupy_count_5)
            hist_total_cost_violate_sum_average.append(agent_total_cost_violate_sum_average)
            agent_total_cost_count_sum_probability = agent_total_cost_count_sum_probability[0] / (
                        occupy_count_1 + occupy_count_2 + occupy_count_3 + occupy_count_4 + occupy_count_5)
            hist_total_cost_count_sum_probability.append(agent_total_cost_count_sum_probability)
            hist_total_violate_count.append(agent_total_violate_count)

            total_reward = agent_reward_global[0] + agent_reward_global[1] + agent_reward_global[2] + \
                           agent_reward_global[3] + agent_reward_global[4]
            total_cost = agent_reward_local[0] + agent_reward_local[1] + agent_reward_local[2] + agent_reward_local[3] + \
                         agent_reward_local[4]
            total_cost_violate_sum_average = (agent_cost_violate_sum[0] + agent_cost_violate_sum[1] +
                                              agent_cost_violate_sum[2] + agent_cost_violate_sum[3] +
                                              agent_cost_violate_sum[4]) / 5
            total_cost_count_sum_probability = (agent_cost_count_sum[0] + agent_cost_count_sum[1] +
                                                agent_cost_count_sum[2] + agent_cost_count_sum[3] +
                                                agent_cost_count_sum[4]) / (
                                                           occupy_count_1 + occupy_count_2 + occupy_count_3 + occupy_count_4 + occupy_count_5)
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
                # "agent_violate_sum_1": agent_cost_violate_sum[0] / occupy_count_1,
                "episode:": episode,
                # "total_reward": total_reward,
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
                df10 = pd.DataFrame(hist_total_violate_count)
                # df2.to_excel('reward.xlsx', index=False)
                df3.to_excel('reward_local.xlsx', index=False)
                df4.to_excel('reward_global.xlsx', index=False)
                df5.to_excel('PPD_violate_average.xlsx', index=False)
                df6.to_excel('PPD_violate_count.xlsx', index=False)
                df7.to_excel('total_cost.xlsx', index=False)
                df8.to_excel('hist_total_cost_violate_sum_average.xlsx', index=False)
                df9.to_excel('hist_total_cost_count_sum_probability.xlsx', index=False)
                df10.to_excel('total_violate_count.xlsx', index=False)

            # df2 = pd.DataFrame(hist)
        df3 = pd.DataFrame(hist_local)
        df4 = pd.DataFrame(hist_global)
        df5 = pd.DataFrame(hist_violate)
        df6 = pd.DataFrame(hist_violate_count)
        df7 = pd.DataFrame(hist_total_cost)
        df8 = pd.DataFrame(hist_total_cost_violate_sum_average)
        df9 = pd.DataFrame(hist_total_cost_count_sum_probability)
        df10 = pd.DataFrame(hist_total_violate_count)
        # df2.to_excel('reward.xlsx', index=False)
        df3.to_excel('reward_local.xlsx', index=False)
        df4.to_excel('reward_global.xlsx', index=False)
        df5.to_excel('PPD_violate_average.xlsx', index=False)
        df6.to_excel('PPD_violate_count.xlsx', index=False)
        df7.to_excel('total_cost.xlsx', index=False)
        df8.to_excel('hist_total_cost_violate_sum_average.xlsx', index=False)
        df9.to_excel('hist_total_cost_count_sum_probability.xlsx', index=False)
        df10.to_excel('total_violate_count.xlsx', index=False)

        plt.plot(reward_hist)
        plt.show()
        time.sleep(1)
        logger.close()

