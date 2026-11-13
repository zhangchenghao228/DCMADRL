import Energyplus
from queue import Queue, Full, Empty
import numpy as np
import itertools
import matplotlib.pyplot as plt
import wandb
import time
import os
from distutils.util import strtobool
import argparse
import pandas as pd
import math
import swanlab

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    # 用户可以通过它来控制是否使用Weights and Biases（一个实验跟踪工具）来跟踪实验
    parser.add_argument("--wandb-project-name", type=str, default="DDPG",
                        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default="shandong111",
                        help="the entity (team) of wandb's project")
    args = parser.parse_args()
    # fmt: on
    return args


name = "EnergyPlus_MADDPG"


class EnergyPlusEnvironment:
    def __init__(self) -> None:
        self.last_obs_copy = {}
        self.count = 0  # 这个用来记录时间
        self.T_MIN = 18
        self.T_MAX = 24
        args = parse_args()
        run_name = f"Circle__{args.exp_name}__{args.seed}__{int(time.time())}"
        self.swanlab = swanlab.init(
            project="P3O",
            workspace="ZCH_SDU",
            tags=["P3O"],
            experiment_name =  "P3O",  # 添加实验名字
            # 跟踪超参数和运行元数据
            config={
                "architecture": "P3O1",
                "notes": "P3O1",
            },
        )
        self.episode = -1
        self.timestep = 0
        self.obs_copy = {}
        self.last_obs = {}  # 这是一个空字典，因为energyplus返回的obs是一个字典
        self.obs_queue: Queue = None  # this queue and the energyplus's queue is the same obj,其实下面这个函数传递的是一个队列
        self.act_queue: Queue = None  # this queue and the energyplus's queue is the same obj，这个注释是什么意思
        self.energyplus: Energyplus.EnergyPlus = Energyplus.EnergyPlus(None, None)

        self.observation_space_size = len(self.energyplus.variables) + len(self.energyplus.meters)
        self.num_agents = 5#智能体的数量
        self.temps_name = ["zone_air_temp_" + str(i + 1) for i in range(self.num_agents)]
        self.occups_name = ["people_" + str(i + 1) for i in range(self.num_agents)]
        self.Relative_Humidity_name = ["zone_air_Relative_Humidity_" + str(i + 1) for i in range(self.num_agents)]
        self.PPD_name = ["PPD_" + str(i + 1) for i in range(self.num_agents)]
        self.heating_setpoint_name = ["zone_heating_setpoint_" + str(i + 1) for i in range(self.num_agents)]
        self.cooling_setpoint_name = ["zone_cooling_setpoint_" + str(i + 1) for i in range(self.num_agents)]
        self.total_energy = 0
        self.total_temp_penalty = [0] * self.num_agents
        self.total_reward = [0] * self.num_agents
        self.total_ppd = [0] * self.num_agents

        # get the indoor/outdoor temperature series
        self.indoor_temps = []
        self.outdoor_temp = []
        # get the setpoint series
        self.setpoints = []
        # get the energy series
        self.energy = []
        # get the occupancy situation
        self.occup_count = []
        self.relative_humidity = []
        self.humditys = []
        self.windspeed = []
        self.winddirection = []
        self.Direct_Solar_Radiation = []
        self.Diffuse_Solar_Radiation = []
        self.PPD = []
        self.heatingpoint = []
        self.coolingpoint = []

        '''仿真时间信息'''
        self.week = 1
        self.day_hour = 0
        '''仿真时间信息'''

        '''VAV能耗信息'''
        self.VAV_energy = 0
        self.VAV_count = 0
        self.total_energy_copy = 0
        '''VAV能耗信息'''
        self.day_count = 0

    # return the first observation
    def reset(self, file_suffix="defalut"):
        '''因为程序要进行多段episode，energyplus会重复运行多次，所以要将下面的变量置为0'''

        self.day_count = 0
        self.day_count = self.day_count + 1
        self.VAV_energy = 0
        self.VAV_count = 0

        self.total_temp_penalty = [0] * self.num_agents
        self.total_energy = 0
        self.total_reward = [0] * self.num_agents
        self.indoor_temps.clear()
        self.outdoor_temp.clear()
        self.setpoints.clear()
        self.energy.clear()
        self.occup_count.clear()

        self.relative_humidity.clear()
        self.humditys.clear()
        self.windspeed.clear()
        self.winddirection.clear()
        self.Direct_Solar_Radiation.clear()
        self.Diffuse_Solar_Radiation.clear()
        self.PPD.clear()
        self.heatingpoint.clear()
        self.coolingpoint.clear()

        self.energyplus.stop()
        self.episode += 1
        '''因为程序要进行多段episode，energyplus会重复运行多次，所以要将下面的变量置为0'''

        if self.energyplus is not None:
            self.energyplus.stop()

        self.obs_queue = Queue(maxsize=1)  # 这里是一个队列，可以理解为在传递一个地址
        self.act_queue = Queue(maxsize=1)  # 这里是一个队列，可以理解为在传递一个地址

        self.energyplus = Energyplus.EnergyPlus(
            obs_queue=self.obs_queue,
            act_queue=self.act_queue,
            # action_space=self.action_space,
            # get_action_func=get_action_f
        )

        self.energyplus.start(file_suffix)

        obs = self.obs_queue.get()  # obs是一个字典

        self.last_obs = obs
        self.last_obs_copy = self.last_obs
        # print(obs)

        self.VAV_energy = self.VAV_energy + (self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000
        self.VAV_count = self.VAV_count + 1

        '''其实这个地方的值没有什么用处，可能这里只是为了画图，从而存储这些值'''
        self.indoor_temps.append([obs[x] for x in self.temps_name])
        self.occup_count.append([obs[x] for x in self.occups_name])
        self.relative_humidity.append([obs[x] for x in self.Relative_Humidity_name])
        self.outdoor_temp.append(obs["outdoor_air_drybulb_temperature"])

        '''获取仿真的时间信息'''
        self.week, self.day_hour = self.energyplus.get_time_information()
        '''获取仿真的时间信息'''

        self.count = self.count + 1
        day_count = self.day_count % 96
        zone1_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_1"] - 15) / (30 - 15), (obs["zone_heating_setpoint_1"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_1"] - 15) / (32 - 15),
                     obs["people_1"] / 11, obs["zone_air_Relative_Humidity_1"] / 100
                     ]
        zone2_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_2"] - 15) / (30 - 15), (obs["zone_heating_setpoint_2"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_2"] - 15) / (32 - 15),
                     obs["people_2"] / 5,
                     obs["zone_air_Relative_Humidity_2"] / 100
                     ]
        zone3_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_3"] - 15) / (30 - 15), (obs["zone_heating_setpoint_3"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_3"] - 15) / (32 - 15),
                     obs["people_3"] / 11,
                     obs["zone_air_Relative_Humidity_3"] / 100
                     ]
        zone4_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"]  - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_4"] - 15) / (30 - 15), (obs["zone_heating_setpoint_4"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_4"] - 15) / (32 - 15),
                     obs["people_4"] / 5,
                     obs["zone_air_Relative_Humidity_4"] / 100
                     ]
        zone5_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_5"] - 15) / (30 - 15), (obs["zone_heating_setpoint_5"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_5"] - 15) / (32 - 15),
                     obs["people_5"] / 20,
                     obs["zone_air_Relative_Humidity_5"] / 100
                     ]


        obs_test = np.array([zone1_obs, zone2_obs, zone3_obs, zone4_obs, zone5_obs])
        PPD = np.array(
            [obs["PPD_1"] / 100, obs["PPD_2"] / 100, obs["PPD_3"] / 100, obs["PPD_4"] / 100, obs["PPD_5"] / 100])
        return obs_test, self.week, self.day_hour, PPD

    # predict next observation
    def step(self, action):
        self.day_count = self.day_count + 1
        self.timestep += 1  # 这个为什么要加1
        done = False
        if self.energyplus.failed():
            raise RuntimeError(f"E+ failed {self.energyplus.sim_results['exit_code']}")

        if self.energyplus.simulation_complete:
            done = True
            obs = self.last_obs
        else:
            timeout = 3
            try:
                self.VAV_count = self.VAV_count + 1
                self.VAV_energy = self.VAV_energy + (self.last_obs["elec_hvac"] + self.last_obs["elec_heat"]) / 3600000
                if self.VAV_count == 8926:
                    self.total_energy_copy = self.VAV_energy
                # self.week, self.day_hour = self.energyplus.get_time_information()
                keys_order = [
                    "zone_cooling_setpoint_1",
                    "zone_heating_setpoint_1",
                    "zone_cooling_setpoint_2",
                    "zone_heating_setpoint_2",
                    "zone_cooling_setpoint_3",
                    "zone_heating_setpoint_3",
                    "zone_cooling_setpoint_4",
                    "zone_heating_setpoint_4",
                    "zone_cooling_setpoint_5",
                    "zone_heating_setpoint_5",
                    # "zone_cooling_setpoint_6",
                    # "zone_heating_setpoint_6",
                ]
                zone_setpoint = []
                for key in keys_order:
                    zone_setpoint.append(self.last_obs[key])
                zone_setpoint_array = np.array(zone_setpoint)

                one_d_list = list(itertools.chain(*action))
                one_d_list = np.array(one_d_list)

                action_map = one_d_list * 5  # 将-1至1的值映射到-5到5之间
                action_result = action_map + zone_setpoint_array
                # 将神经网络的输出值映射到15-30
                # action_result = 7.5 * (one_d_list + 1) + 15
                # 这里需要对这个1维numpy数组里的元素进行判断，使得制冷温度设定值小于制热温度设定值，以及制冷温度设定值小于制热温度设定值均要在15-30这个范围之间
                '''这里需要对这个1维numpy数组里的元素进行判断，使得制冷温度设定值小于制热温度设定值，以及制冷温度设定值小于制热温度设定值均要在15-30这个范围之间'''

                for i in range(0, len(keys_order), 2):
                    heating_index = i + 1
                    cooling_index = i

                    heating_value = action_result[heating_index]
                    cooling_value = action_result[cooling_index]

                    # 确保制热温度小于制冷温度
                    if heating_value >= cooling_value:
                        heating_value = cooling_value - 1

                    # 确保制热温度在 15-30 之间
                    if heating_value < 15:
                        heating_value = 15
                    elif heating_value > 30:
                        heating_value = 30

                    # 确保制冷温度在 15-30 之间
                    if cooling_value < 15:
                        cooling_value = 15
                    elif cooling_value > 30:
                        cooling_value = 30

                    if heating_value == cooling_value:
                        cooling_value = cooling_value + 2

                    # if heating_value == 30:
                    #     heating_value = 29
                    #     cooling_value = 30

                    # 更新数组中的值
                    action_result[heating_index] = heating_value
                    action_result[cooling_index] = cooling_value

                action_result = action_result.tolist()
                self.setpoints.append(action_result)  # 将神经网络输出的-1至1的数值转换为19-24之间的数值
                '''这里相当于是在传递神经网络输出的索引值'''
                # start_time = time.time()
                self.act_queue.put(action_result, timeout=timeout)  # timeout指定此操作等待的时间，这个接收的是一个1维的numpy数组
                self.last_obs_copy = self.last_obs
                self.obs_copy = self.last_obs
                self.last_obs = obs = self.obs_queue.get(timeout=timeout)
                # end_time = time.time()
                # print('env_time: ', end_time - start_time)
            except(Full, Empty):
                done = True
                self.obs_copy = self.last_obs
                obs = self.last_obs
                self.last_obs_copy = self.last_obs
            '''上面这个函数用于捕获异常'''

        reward, reward_local, reward_global, cost_violate_sum, cost_count = self.get_reward  # 这是一个标量
        to_log = {
            "VAV_ENERGY": self.total_energy_copy
        }
        self.count = self.count + 1
        self.swanlab.log(to_log, step=self.count)
        obs_vec = np.array(list(obs.values()))  # 这是一个列表
        self.week, self.day_hour = self.energyplus.get_time_information()
        day_count = self.day_count % 96
        zone1_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_1"] - 15) / (30 - 15), (obs["zone_heating_setpoint_1"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_1"] - 15) / (32 - 15),
                     obs["people_1"] / 11, obs["zone_air_Relative_Humidity_1"] / 100
                     ]
        zone2_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_2"] - 15) / (30 - 15), (obs["zone_heating_setpoint_2"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_2"] - 15) / (32 - 15),
                     obs["people_2"] / 5,
                     obs["zone_air_Relative_Humidity_2"] / 100
                     ]
        zone3_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_3"] - 15) / (30 - 15), (obs["zone_heating_setpoint_3"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_3"] - 15) / (32 - 15),
                     obs["people_3"] / 11,
                     obs["zone_air_Relative_Humidity_3"] / 100
                     ]
        zone4_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_4"] - 15) / (30 - 15), (obs["zone_heating_setpoint_4"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_4"] - 15) / (32 - 15),
                     obs["people_4"] / 5,
                     obs["zone_air_Relative_Humidity_4"] / 100
                     ]
        zone5_obs = [day_count / 96, self.day_hour / 24, self.week / 7, (obs["outdoor_air_drybulb_temperature"] - 10) / (40 - 10),
                     obs["Outdoor_Air_Relative_Humidity"] / 100, (obs["elec_hvac"] + obs["elec_heat"]) / 15000000,
                     (obs["zone_air_temp_5"] - 15) / (30 - 15), (obs["zone_heating_setpoint_5"] - 15) / (30 - 15),
                     (obs["zone_cooling_setpoint_5"] - 15) / (32 - 15),
                     obs["people_5"] / 20,
                     obs["zone_air_Relative_Humidity_5"] / 100
                     ]


        obs_test = np.array([zone1_obs, zone2_obs, zone3_obs, zone4_obs, zone5_obs])
        PPD = np.array(
            [obs["PPD_1"] / 100, obs["PPD_2"] / 100, obs["PPD_3"] / 100, obs["PPD_4"] / 100, obs["PPD_5"] / 100])
        done_flag = np.array([done, done, done, done, done])
        return obs_test, reward, reward_local, reward_global, done_flag, PPD, cost_violate_sum, cost_count

    '''下面这个@property不能删除'''

    @property
    def get_reward(self):
        PPD_thres = 0.1
        w_e = 0.6
        w_c = 0.4
        cof = 10
        reward = []  # 存放5个agent的奖励
        reward_local = []
        reward_global = []

        cost_sum = []
        cost_count = []  # 存放5个agent的代价
        # according to the meters and variables to compute
        obs = self.last_obs  # 这个是在获取状态，这个状态是一个字典
        '''这个函数用于判断每个区域是否有人'''
        occups_vals = []
        for occup in self.occups_name:
            occups_vals.append(obs[occup])
        '''这个函数取得每个区域的PPD值'''
        PPD_vals = []
        for PPD in self.PPD_name:
            PPD_vals.append(obs[PPD] / 100)
        '''这个函数取得每个区域的PPD值'''
        '''计算c(t)值'''
        c_result = []
        cost_PPD_violate_sum = []  # 存放5个agent的代价
        cost_PPD_count_sum = []  # 存放5个agent的代价

        for PPD_copy in PPD_vals:
            if PPD_copy > PPD_thres:
                c_result.append(PPD_copy)
                cost_PPD_violate_sum.append(PPD_copy - PPD_thres)
                cost_PPD_count_sum.append(1)
            else:
                # ppd_temp = self.calculate_reward(PPD_copy)
                c_result.append(0)
                cost_PPD_violate_sum.append(0)
                cost_PPD_count_sum.append(0)
        '''这个值对于5个智能体都是一样的，是不是应该寻找一个替代值'''
        # TODO find a good function to evaluate the temperature reward
        energy = (obs["elec_hvac"] + obs["elec_heat"]) /  20000000 / 2.5 # 将电能消耗量从瓦特秒转换为千瓦时,这个值可能是在1以下
        for o, c, cost_PPD_violate, cost_PPD_count in zip(occups_vals, c_result, cost_PPD_violate_sum, cost_PPD_count_sum):
            if o == 0:
                r = -w_e * energy
                reward_local_temp = 0
                reward_global_temp = - energy
                cost_PPD_violate_temp = 0
                cost_count_temp = 0
            else:
                r = -w_e * energy - w_c * c
                reward_local_temp =  cof * c
                reward_global_temp = - energy
                cost_PPD_violate_temp = cost_PPD_violate
                cost_count_temp = cost_PPD_count
            reward.append(r)
            reward_local.append(reward_local_temp)
            reward_global.append(reward_global_temp)
            cost_sum.append(cost_PPD_violate_temp)
            cost_count.append(cost_count_temp)
        return reward, reward_local, reward_global, cost_sum, cost_count

    def close(self):
        if self.energyplus is not None:
            self.energyplus.stop()

    def render(self):
        # get the indoor/outdoor temperature series
        zone_temp = []
        for i in range(5):
            zone_temp.append(np.array(self.indoor_temps)[:, i])

        # get occupancy
        zone_occupy = []
        for i in range(5):
            zone_occupy.append(np.array(self.occup_count)[:, i])
        # get the setpoint series
        sp_series = []
        for i in range(0, 10, 2):
            sp_series.append(np.array(self.setpoints)[:, i])
        # get the energy series
        x = range(len(self.setpoints))

        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("temperature (℃)")
            plt.plot(x, zone_temp[i], label=f"zone_{i + 1}_temperature")
        plt.legend()
        plt.show()

        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("setpoint (℃)")
            plt.plot(x, sp_series[i], label=f"zone_{i + 1}_setpoint")
        plt.legend()
        plt.show()
        for i in range(5):
            plt.xlabel("timestep")
            plt.ylabel("occupancy")
            plt.plot(x, zone_occupy[i], label=f"zone_{i + 1}_people_occupant_count ")
        plt.legend()
        plt.show()

        plt.plot(x, self.energy)
        plt.title("energy cost")
        plt.xlabel("timestep")
        plt.ylabel("energy cost (kwh)")
        plt.show()

        plt.plot(x, self.outdoor_temp)
        plt.title("outdoor temperature")
        plt.xlabel("timestep")
        plt.ylabel("temperature (℃)")
        plt.show()
