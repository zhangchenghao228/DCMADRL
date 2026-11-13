from runner import Runner
from common.arguments import get_args
import EnergyplusEnv
from maddpg.seed_init import set_global_seed
from rlkit import pytorch_util as ptu 


if __name__ == '__main__':

    # 设置全局随机种子
    set_global_seed(82)
    ptu.set_gpu_mode(True, 0)#这个地方是为了将pytorch_util中的device设置为gpu模式
    env = EnergyplusEnv.EnergyPlusEnvironment()
    # 获取环境
    args = get_args()
    # 初始化
    runner = Runner(args, env)
    # 若选择训练/测试
    runner.run_tensor()



