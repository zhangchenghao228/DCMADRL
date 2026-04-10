from runner import Runner
from common.arguments import get_args
import EnergyplusEnv
from maddpg.seed_init import set_global_seed


if __name__ == '__main__':

    # 设置全局随机种子
    set_global_seed(122)
    env = EnergyplusEnv.EnergyPlusEnvironment()
    # 获取环境
    args = get_args()
    # 初始化
    runner = Runner(args, env)
    # 若选择训练/测试
    runner.run_tensor()
