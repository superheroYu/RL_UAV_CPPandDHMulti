import random
from collections import deque
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        self.memory = deque(maxlen=size) # 分配内存
        self.size = size # 容量

    def store(self, experience):
        """ 存储经验(state, action, reward, next_state, done)"""
        self.memory.append(experience)

    def sample(self, batch_size):
        """ 采样batch_size大小的经验 """
        # random_idx = np.random.choice(len(self.memory), size=batch_size, replace=False)
        # sampled_experiences = [self.memory[i] for i in random_idx]
        sampled_experiences = random.sample(self.memory, batch_size)
        states = [x[0] for x in sampled_experiences]
        actions = [[x[1]] for x in sampled_experiences]
        rewards = [[x[2]] for x in sampled_experiences]
        next_states = [x[3] for x in sampled_experiences]
        done = [[x[4]] for x in sampled_experiences]
        states = [np.array(list(x)) for x in zip(*states)]
        next_states = [np.array(list(x)) for x in zip(*next_states)]
        return [states, actions, rewards, next_states, done]  # Transpose list of experiences

    def get(self, start, length):
        """ 获取start开始length条经验 """
        return list(map(list, zip(*list(self.memory)[start:start + length])))  # Transpose sublist of experiences

    def get_max_size(self):
        """ 获取最大容量 """
        return self.size

    def reset(self):
        """ 重置ReplayBuffer """
        self.memory.clear()

    def shuffle(self):
        """ 打乱经验 """
        shuffled_memory = self.sample(self.get_size())
        self.memory = deque(shuffled_memory, maxlen=self.size)
        
    def __len__(self):
        """ 获取已存储的经验量 """
        return len(self.memory)
    
    @property
    def full(self):
        """ 检查是否满了 """
        return len(self.memory) == self.size
