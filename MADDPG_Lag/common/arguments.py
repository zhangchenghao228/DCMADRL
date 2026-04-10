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
    parser.add_argument("--lr-actor", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")
    # lagrangian_lr
    parser.add_argument("--lagrangian_lr", type=float, default=1e-3, help="learning rate of lagrangian")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1,
                        help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.001, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1000000),
                        help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=5000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # --------------------------------------MATD3--------------------------------------------------------------------
    parser.add_argument("--policy_noise", type=float, default=0.05, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.1, help="Clip noise")
    parser.add_argument("--max_action", type=float, default=1, help="max action")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")
    # # fedrate_rate
    # parser.add_argument("--federate-rate", type=int, default=100,
    #                     help="federate in training rate")

    # Evaluate
    # parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    # parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    # parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    # parser.add_argument("--evaluate-rate", type=int, default=400, help="how often to evaluate model")

    # parser.add_argument("--obs-shape", type=int, default=1000, help="how often to evaluate model")
    args = parser.parse_args()

    return args
