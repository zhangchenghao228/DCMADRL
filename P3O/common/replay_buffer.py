import threading
import numpy as np
from maddpg.seed_init import set_global_seed

# 设置全局随机种子
set_global_seed(22)


class Buffer:
    # '''这个地方需要修改，state_shape和action_shape都要加1'''
    def __init__(self, args, n_agents=5, state_shape=11, action_shape=2):
        self.size = args.horizon_size
        self.args = args
        # memory management
        self.current_size = 0
        # create the buffer to store info
        self.buffer = dict()
        for i in range(n_agents):
            self.buffer['o_%d' % i] = np.empty([self.size, state_shape])
            self.buffer['u_%d' % i] = np.empty([self.size, action_shape])
            self.buffer['r_%d' % i] = np.empty([self.size])
            self.buffer['o_next_%d' % i] = np.empty([self.size, state_shape])
            self.buffer['action_log_pi_%d' % i] = np.empty([self.size, action_shape])#action_log_pi
            self.buffer['done_%d' % i] = np.empty([self.size])#done
            self.buffer['cost_%d' % i] = np.empty([self.size])
            self.buffer['reward_%d' % i] = np.empty([self.size])
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, o, u, r, o_next, action_log_pi, done_flag, cost, reward, n_agents=5):
        idxs = self._get_storage_idx(inc=1)  # 以transition的形式存，每次只存一条经验
        for i in range(n_agents):
            with self.lock:
                self.buffer['o_%d' % i][idxs] = o[i]
                self.buffer['u_%d' % i][idxs] = u[i]
                self.buffer['r_%d' % i][idxs] = r[i]
                self.buffer['o_next_%d' % i][idxs] = o_next[i]
                self.buffer['action_log_pi_%d' % i][idxs] = action_log_pi[i]
                self.buffer['done_%d' % i][idxs] = done_flag[i]
                self.buffer['cost_%d' % i][idxs] = cost[i]
                self.buffer['reward_%d' % i][idxs] = reward[i]

    # # sample the data from the replay buffer
    # def sample(self, batch_size):
    #     temp_buffer = {}
    #     # weather_buffer = []
    #     idx = np.random.randint(0, self.current_size, batch_size)
    #     for key in self.buffer.keys():
    #         temp_buffer[key] = self.buffer[key][idx]
    #     # weather_buffer = self.weather_buffer[idx]
    #     # 将生成的索引数组的每个元素加 1
    #     # idx_next = idx + 1
    #     # weather_buffer_next = self.weather_buffer[idx_next]
    #     return temp_buffer
    
    def sample_all(self):
        temp_buffer = {}
        for key in self.buffer.keys():
            temp_buffer[key] = self.buffer[key][:self.current_size]#这个self.current_size是当前存储的大小， 在PPO算法里面应该是horizon_size
        return temp_buffer
    
    # 清空整个buffer
    # 这个函数会将所有的buffer清空，并将每个数组元素置为0
    # def clear(self):
    #     with self.lock:
    #         for key in self.buffer.keys():
    #             self.buffer[key].fill(0)  # 将每个数组元素置为0，这个地方可以删除
    #     self.current_size = 0  # 重置当前存储大小

    def clear(self):
        with self.lock:
            self.current_size = 0


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
