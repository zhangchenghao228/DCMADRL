import argparse

"""
Here are the param for the training
"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=96, help="maximum episode length")#这个是一个episode的步数
    parser.add_argument("--max-episode", type=int, default=300, help="number of episodes")#40
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")#这个学习率可能有点问题
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--lagrangian_lr", type=float, default=1e-4, help="learning rate of lagrangian")
    # parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    # parser.add_argument("--noise_rate", type=float, default=0.1,
                        # help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    # parser.add_argument("--tau", type=float, default=0.001, help="parameter for updating the target network")
    # parser.add_argument("--buffer-size", type=int, default=int(1024),#这个值再on policy的PPO算法里面是buffer的大小，每次存满之后进行训练，训练完成之后会把buffer进行清空处理
    #                     help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # --------------------------------------MATD3--------------------------------------------------------------------
    # parser.add_argument("--policy_noise", type=float, default=0.1, help="Target policy smoothing")
    # parser.add_argument("--noise_clip", type=float, default=0.1, help="Clip noise")
    parser.add_argument("--max_action", type=float, default=1, help="max action")
    # parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")

    # --------------------------------------IPPO--------------------------------------------------------------------
    parser.add_argument("--horizon-size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--K_epochs", type=int, default=5, help="number of episodes to optimize at the same time")#PPO算法新增内容

    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda for Generalized Advantage Estimation")#PPO算法新增内容

    parser.add_argument("--clip_param", type=float, default=0.2, help="GAE lambda for Generalized Advantage Estimation")#PPO算法新增内容

    parser.add_argument("--entropy_coefficient", type=float, default=0.01, help="GAE lambda for Generalized Advantage Estimation")#PPO算法新增内容
    
    parser.add_argument("--huber_delta", type=float, default=10.0) # huber_loss参数

    args = parser.parse_args()

    return args

# D:\code\safe_RL\MAPPO\common\arguments.py

# C:\Users\A1736\Desktop\MAPPO\common\arguments.py