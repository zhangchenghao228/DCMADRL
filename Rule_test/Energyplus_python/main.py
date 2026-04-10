import Energyplus_Env
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # 获取参数
    num_agent = 5
    Energyplus_env = Energyplus_Env.EnergyPlusEnvironment()
    action = [[30, 15], [30, 15], [30, 15], [30, 15], [30, 15]]
    s, week, hour, PPD, Day_count = Energyplus_env.reset()
    reward_hist = []
    reward_list = []
    reward_hist_copy = []
    reward_list_copy = []
    agent_reward = [0] * num_agent
    agent_reward_local = [0] * num_agent
    agent_reward_global = [0] * num_agent
    max_reward = 100000
    hist = []
    PPD1 = []
    PPD2 = []
    PPD3 = []
    PPD4 = []
    PPD5 = []
    # PPD6 = []
    PPD1_10_violate_count = 0
    PPD2_10_violate_count = 0
    PPD3_10_violate_count = 0
    PPD4_10_violate_count = 0
    PPD5_10_violate_count = 0

    PPD1_20_violate_count = 0
    PPD2_20_violate_count = 0
    PPD3_20_violate_count = 0
    PPD4_20_violate_count = 0
    PPD5_20_violate_count = 0

    PPD1_10_violate_sum = 0
    PPD2_10_violate_sum = 0
    PPD3_10_violate_sum = 0
    PPD4_10_violate_sum = 0
    PPD5_10_violate_sum = 0

    occupy_count = 0
    agent_reward_copy = [0] * num_agent
    max_reward_copy = 100000
    hist_copy = []
    done_flag = 0
    while not done_flag:
        if week == 7 or week == 1:
            action = [[30.0, 15.0], [30.0, 15.0], [30.0, 15.0], [30.0, 15.0], [30.0, 15.0]]
        else:
            if 31 <= Day_count <= 83:
                action1 = [26, 24]
            else:
                action1 = [30.0, 15.0]
            if 31 <= Day_count <= 83:
                action2 = [26, 24]
            else:
                action2 = [30.0, 15.0]
            if 31 <= Day_count <= 83:
                action3 = [26, 24]
            else:
                action3 = [30.0, 15.0]
            if 31 <= Day_count <= 83:
                action4 = [26, 24]
            else:
                action4 = [30.0, 15.0]
            if 31 <= Day_count <= 83:
                action5 = [26, 24]
            else:
                action5 = [30.0, 15.0]
            action = [action1, action2, action3, action4, action5]

        s_next, r_local, r_global, done_flag, week, hour, PPD, Day_count, total_energy = Energyplus_env.step(action)

        s = s_next
        r_local_array = np.array(r_local)
        r_global_array = np.array(r_global)
        r = r_local_array + r_global_array

        reward = r

        for i in range(num_agent):
            agent_reward[i] += r[i]
            agent_reward_local[i] += r_local_array[i]
            agent_reward_global[i] += r_global_array[i]

        if s[0][8] != 0:
            occupy_count = occupy_count + 1
            PPD1.append(PPD[0].astype(np.float64))
        if s[1][8] != 0:
            PPD2.append(PPD[1].astype(np.float64))
        if s[2][8] != 0:
            PPD3.append(PPD[2].astype(np.float64))
        if s[3][8] != 0:
            PPD4.append(PPD[3].astype(np.float64))
        if s[4][8] != 0:
            PPD5.append(PPD[4].astype(np.float64))
        if s[0][8] != 0 and PPD[0] > 0.1:
            PPD1_10_violate_sum = PPD1_10_violate_sum + (PPD[0] - 0.1)
            PPD1_10_violate_count = PPD1_10_violate_count + 1
        if s[0][8] != 0 and PPD[0] > 0.2:
            PPD1_20_violate_count = PPD1_20_violate_count + 1
        if s[1][8] != 0 and PPD[1] > 0.1:
            PPD2_10_violate_sum = PPD2_10_violate_sum + (PPD[1] - 0.1)
            PPD2_10_violate_count = PPD2_10_violate_count + 1
        if s[1][8] != 0 and PPD[1] > 0.2:
            PPD2_20_violate_count = PPD2_20_violate_count + 1
        if s[2][8] != 0 and PPD[2] > 0.1:
            PPD3_10_violate_sum = PPD3_10_violate_sum + (PPD[2] - 0.1)
            PPD3_10_violate_count = PPD3_10_violate_count + 1
        if s[2][8] != 0 and PPD[2] > 0.2:
            PPD3_20_violate_count = PPD3_20_violate_count + 1
        if s[3][8] != 0 and PPD[3] > 0.1:
            PPD4_10_violate_sum = PPD4_10_violate_sum + (PPD[3] - 0.1)
            PPD4_10_violate_count = PPD4_10_violate_count + 1
        if s[3][8] != 0 and PPD[3] > 0.2:
            PPD4_20_violate_count = PPD4_20_violate_count + 1
        if s[4][8] != 0 and PPD[4] > 0.1:
            PPD5_10_violate_sum = PPD5_10_violate_sum + (PPD[4] - 0.1)
            PPD5_10_violate_count = PPD5_10_violate_count + 1
        if s[4][8] != 0 and PPD[4] > 0.2:
            PPD5_20_violate_count = PPD5_20_violate_count + 1
        reward = np.sum(reward_list) / 96 * 1
        reward_hist.append(reward)

    ave_r = [0] * num_agent
    for i in range(num_agent):
        ave_r[i] = agent_reward[i]
    print("r1:%.2f r2:%.2f r3:%.2f r4:%.2f r5:%.2f " %
          (agent_reward[0], agent_reward[1], agent_reward[2], agent_reward[3], agent_reward[4]))
    print("r_local_1:%.2f r_local_2:%.2f r_local_3:%.2f r_local_4:%.2f r_local_5:%.2f" %
          (agent_reward_local[0], agent_reward_local[1], agent_reward_local[2], agent_reward_local[3],
           agent_reward_local[4]))
    print("r_global_1:%.2f r_global_2:%.2f r_global_3:%.2f r_global_4:%.2f r_global_5:%.2f " %
          (agent_reward_global[0], agent_reward_global[1], agent_reward_global[2], agent_reward_global[3],
           agent_reward_global[4]))

    st = time.time()
    hist.append(agent_reward)
    total_reward = (agent_reward[0] + agent_reward[1] + agent_reward[2] + agent_reward[3] +
                    agent_reward[4])
    total_reward_local = (
            agent_reward_local[0] + agent_reward_local[1] + agent_reward_local[2] + agent_reward_local[3] +
            agent_reward_local[4])
    total_reward_global = (
            agent_reward_global[0] + agent_reward_global[1] + agent_reward_global[2] + agent_reward_global[3] +
            agent_reward_global[4])
    PPD1_reward = sum(PPD1)
    PPD2_reward = sum(PPD2)
    PPD3_reward = sum(PPD3)
    PPD4_reward = sum(PPD4)
    PPD5_reward = sum(PPD5)

    print("**************************************************************")
    print("PPD1_average:", PPD1_reward / occupy_count)
    print("PPD2_average:", PPD2_reward / occupy_count)
    print("PPD3_average:", PPD3_reward / occupy_count)
    print("PPD4_average:", PPD4_reward / occupy_count)
    print("PPD5_average:", PPD5_reward / occupy_count)
    print("5_Zone_PPD_average", (PPD1_reward / occupy_count + PPD2_reward / occupy_count + PPD3_reward / occupy_count +
                                 PPD4_reward / occupy_count + PPD5_reward / occupy_count) / 5 * 100)
    print("occupy_count: ", occupy_count)
    print("**************************************************************")
    # print("PPD6_reward_average:", PPD6_reward / occupy_count)
    print("PPD1_10_violate_count", PPD1_10_violate_count)
    print("PPD2_10_violate_count", PPD2_10_violate_count)
    print("PPD3_10_violate_count", PPD3_10_violate_count)
    print("PPD4_10_violate_count", PPD4_10_violate_count)
    print("PPD5_10_violate_count", PPD5_10_violate_count)

    print("occupy_count: ", occupy_count)
    print("**************************************************************")

    time.sleep(0.5)
    print("PPD1_10_violate_sum", PPD1_10_violate_sum)
    print("PPD2_10_violate_sum", PPD2_10_violate_sum)
    print("PPD3_10_violate_sum", PPD3_10_violate_sum)
    print("PPD4_10_violate_sum", PPD4_10_violate_sum)
    print("PPD5_10_violate_sum", PPD5_10_violate_sum)

    time.sleep(0.5)
    print("PPD1_10_average_violate", PPD1_10_violate_sum * 100 / occupy_count)
    print("PPD2_10_average_violate", PPD2_10_violate_sum * 100 / occupy_count)
    print("PPD3_10_average_violate", PPD3_10_violate_sum * 100 / occupy_count)
    print("PPD4_10_average_violate", PPD4_10_violate_sum * 100 / occupy_count)
    print("PPD5_10_average_violate", PPD5_10_violate_sum * 100 / occupy_count)

    time.sleep(0.5)

    print("PPD1_10_violate_probability", 100 *  PPD1_10_violate_count / occupy_count)
    print("PPD2_10_violate_probability", 100 *  PPD2_10_violate_count / occupy_count)
    print("PPD3_10_violate_probability", 100 *  PPD3_10_violate_count / occupy_count)
    print("PPD4_10_violate_probability", 100 *  PPD4_10_violate_count / occupy_count)
    print("PPD5_10_violate_probability", 100 *  PPD5_10_violate_count / occupy_count)
    time.sleep(0.5)

    PPD_avg_rewards = [
                100 * PPD1_reward / occupy_count,
                100 * PPD2_reward / occupy_count,
                100 * PPD3_reward / occupy_count,
                100 * PPD4_reward / occupy_count,
                100 * PPD5_reward / occupy_count
                ]
            
    PPD_average_5_zones_average = sum(PPD_avg_rewards) / len(PPD_avg_rewards)
            

            
    PPD_10_violate_sums = [
                PPD1_10_violate_sum,
                PPD2_10_violate_sum,
                PPD3_10_violate_sum,
                PPD4_10_violate_sum,
                PPD5_10_violate_sum
                ]
            
    PPD_10_violate_sums_5_zones_average = sum(PPD_10_violate_sums) / len(PPD_10_violate_sums)
            
    PPD_10_avg_violates = [
                PPD1_10_violate_sum * 100 / occupy_count,
                PPD2_10_violate_sum * 100 / occupy_count,
                PPD3_10_violate_sum * 100 / occupy_count,
                PPD4_10_violate_sum * 100 / occupy_count,
                PPD5_10_violate_sum * 100 / occupy_count
                ]
            
    PPD_10_avg_violates_5_zones_average = sum(PPD_10_avg_violates) / len(PPD_10_avg_violates)

    PPD_10_violate_counts = [
                PPD1_10_violate_count,
                PPD2_10_violate_count,
                PPD3_10_violate_count,
                PPD4_10_violate_count,
                PPD5_10_violate_count
                ]
            
    PPD_10_violate_counts_5_zones_average = sum(PPD_10_violate_counts) / len(PPD_10_violate_counts)

    PPD_10_violate_probs = [
                PPD1_10_violate_count * 100 / occupy_count,
                PPD2_10_violate_count * 100 / occupy_count,
                PPD3_10_violate_count * 100 / occupy_count,
                PPD4_10_violate_count * 100 / occupy_count,
                PPD5_10_violate_count * 100 / occupy_count
            ]

    PPD_10_violate_probs_5_zones_average = sum(PPD_10_violate_probs) / len(PPD_10_violate_probs)

    total_violate_count = sum(PPD_10_violate_counts)
    total_violate_prob = 100 * total_violate_count / (5 * occupy_count)

    # 构建DataFrame
    df = pd.DataFrame({
                "Zone": [f"PPD{i+1}" for i in range(5)],
                "Average PPD": PPD_avg_rewards,
                "10% Violation Sum": PPD_10_violate_sums,
                "10% Average Violation (%)": PPD_10_avg_violates,
                "10% Violation Count": PPD_10_violate_counts,
                "10% Violation Probability (%)": PPD_10_violate_probs
                })
            
    # 添加 5 区域平均值行
    df.loc[len(df)] = [
                "5-Zone Average",
                PPD_average_5_zones_average,
                PPD_10_violate_sums_5_zones_average,
                PPD_10_avg_violates_5_zones_average,
                PPD_10_violate_counts_5_zones_average,
                PPD_10_violate_probs_5_zones_average
                ]



    # 添加总计行
    df.loc[len(df)] = [
                "Total",
                "-", 
                sum(PPD_10_violate_sums),
                "-", 
                total_violate_count,
                total_violate_prob
            ]

    # 添加 total_energy 行
    df.loc[len(df)] = [
                "Total Energy",
                total_energy,
                "-", "-", "-", "-"
            ]

    # 保存为Excel
    save_path = "RULE_PPD_statistics.xlsx"
    df.to_excel(save_path, index=False)
    print(f"\033[92mExcel file saved to {save_path}\033[0m")



    














    


