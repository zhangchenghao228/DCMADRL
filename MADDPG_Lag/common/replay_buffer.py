import threading
import numpy as np
# 设置全局随机种子
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(22)

class Buffer:
    '''这个地方需要修改，state_shape和action_shape都要加1'''
    def __init__(self, args, n_agents=5, state_shape=11, action_shape=2):
        self.size = args.buffer_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, state_shape])
            self.buffer['u_%d' % i] = np.empty([self.size, action_shape])
            self.buffer['r_local_%d' % i] = np.empty([self.size])
            self.buffer['r_global_%d' % i] = np.empty([self.size])#添加的全局奖励相关项
            self.buffer['o_next_%d' % i] = np.empty([self.size, state_shape])
            # self.buffer['occ_%d' % i] = np.empty([self.size])#添加的全局奖励相关项
        # thread lock
        # self.weather_buffer = np.zeros((self.size, 12, 2), dtype=np.float32)#这个是存放天气数据的weather buffer
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r_local, r_global, o_next, n_agents=5):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_local_%d' % i][idxs] = r_local[i]
                self.buffer['r_global_%d' % i][idxs] = r_global[i]#添加的全局奖励相关项
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
                # self.buffer['occ_%d' % i][idxs] = occ[i]

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffer = {}
        # weather_buffer = []
        idx = np.random.randint(0, self.current_size, batch_size)
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][idx]
        # weather_buffer = self.weather_buffer[idx]
        # 将生成的索引数组的每个元素加 1
        # idx_next = idx + 1
        # weather_buffer_next = self.weather_buffer[idx_next]
        return temp_buffer

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx