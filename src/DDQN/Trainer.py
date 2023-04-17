from typing import Optional
from .Agent import DDQNAgent
from .ReplayBuffer import ReplayBuffer
import tqdm


class DDQNTrainerParams:
    def __init__(self):
        self.batch_size: int = 128 # 设置批处理大小
        self.num_steps: int = 1e6 # 设置训练的总步数
        self.rm_pre_fill_ratio: float = 0.5 # 设置预填充 Replay Buffer 的比例
        self.rm_pre_fill_random: bool = True # 设置是否随机预填充 Replay Buffer
        self.eval_period: int = 5 # 设置评估周期
        self.rm_size: int = 50000 # 设置 Replay Memory 的大小
        self.load_model: Optional[str]  = "" # 设置加载的模型路径，默认为空字符串


class DDQNTrainer:
    def __init__(self, params: DDQNTrainerParams, agent: DDQNAgent):
        self.params = params # 存储参数对象
        self.replay_buffer = ReplayBuffer(size=params.rm_size) # 根据参数创建一个 Replay Buffer 对象
        self.agent = agent # 存储智能体对象

        if self.params.load_model != "": # 如果有指定模型路径，则加载模型
            print("正在为智能体加载模型：", self.params.load_model, "……")
            self.agent.load_weights(self.params.load_model)

        self.prefill_bar = None # 初始化一个进度条变量，用于显示预填充 Replay Buffer 的进度

    def add_experience(self, *experiences):
        """ 添加经验, experiences=(state, action, reward, next_state, done) """
        self.replay_buffer.store(experience=experiences)

    def train_agent(self):
        """ 训练智能体 """
        if self.params.batch_size > len(self.replay_buffer): # 如果batch size大于经验回放池里经验数，则不训练
            return
        mini_batch = self.replay_buffer.sample(self.params.batch_size) # 从经验回放池中采样一批数据

        self.agent.train(mini_batch) # 智能体进行训练

    def should_fill_replay_buffer(self):
        target_size = self.replay_buffer.get_max_size() * self.params.rm_pre_fill_ratio # 预填充的大小
        if len(self.replay_buffer) >= target_size or self.replay_buffer.full: # 如果经验回放池里的经验数大于与填充大小或者经验回放池已满
            if self.prefill_bar: # 如果有进度条则更新进度条
                self.prefill_bar.update(target_size - self.prefill_bar.n)
                self.prefill_bar.close()
            return False

        if self.prefill_bar is None: # 如果没有进度条
            print("正在填充经验回放池")
            self.prefill_bar = tqdm.tqdm(total=target_size) # 创建一个进度条对象

        self.prefill_bar.update(len(self.replay_buffer) - self.prefill_bar.n) # 更新填充进度

        return True
