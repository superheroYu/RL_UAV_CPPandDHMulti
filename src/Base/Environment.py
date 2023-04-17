import tqdm
from ..ModelStats import ModelStatsParams, ModelStats
from .Display import Display
from .GridActions import GridActions
import numpy as np
from typing import Type, Union
from ..DDQN.Agent import DDQNAgent
from ..DDQN.Trainer import DDQNTrainer
from .Rewards import GridRewards
from .GridPhysics import GridPhysics
from ..ModelStats import ModelStats
from .State import State
from .Grid import Grid


class EnvironmentParams(object):
    def __init__(self,
                 ActionsEnum = GridActions):
        self.model_stats_params = ModelStatsParams()
        self.step_count = 0
        self.episode_count = 0

class Environment(object):
    def __init__(self, params: EnvironmentParams, display: Display):
        super(Environment, self).__init__()
        
        self.episode_count = params.episode_count # 记录训练回合数，用于tensorboard对齐
        self.step_count = params.step_count # 记录训练步数

        ### 初始化其他环境变量
        self.trainer: DDQNTrainer = None  # 训练器
        self.agent: DDQNAgent = None  # 智能体
        self.grid: Grid = None  # 网格环境
        self.rewards: GridRewards = None  # 奖励函数
        self.physics: GridPhysics = None  # 物理引擎
        self.display = display  # 显示函数
        self.stats = ModelStats(params.model_stats_params, display=display)

    def reset(self, init_state: State = None) -> State:
        """ 重置环境, 初始化回合 """
        if init_state:
            state = self.grid.init_scenario(init_state)  # 使用指定初始化状态初始化回合
        else:
            state = self.grid.init_episode()  # 使用默认初始化状态初始回合

        self.rewards.reset()  # 重置奖励函数
        self.physics.reset(state)  # 重置物理引擎
        return state

    def step(self, action, pre_state):
        """ 采取行动 """
        action = GridActions(action)
        state = self.physics.step(action)  # 在物理引擎中进行状态转移
        reward = self.rewards.calculate_reward(pre_state, action, state)  # 计算奖励
        done = state.terminal  # 判断是否结束回合
        return state, reward, done

    def fill_replay_buffer(self):
        "填充经验回放缓存"
        while self.trainer.should_fill_replay_buffer():  # 当需要经验回放填充时，执行以下操作
            state = self.reset()  # 初始化一个新状态
            while not state.is_terminal():
                action = self.agent.get_random_action()  # 智能体随机获取动作
                state_ = state.copy()
                next_state, reward, done = self.step(
                    action=action, pre_state=state_)  # 执行动作并新状态、奖励、终止标志和信息
                self.trainer.add_experience(self.decode_state(state_), action, reward, self.decode_state(next_state),
                                            done)  # 将状态转移、奖励和终止标志添加到经验回放缓存中
                state = next_state

    def run(self):
        """ 进行训练 """
        self.fill_replay_buffer()  # 填充经验回放缓存
        print("正在执行：", self.stats.params.log_file_name)
        bar = tqdm.tqdm(total=int(self.trainer.params.num_steps))
        last_step = 0
        while self.step_count < self.trainer.params.num_steps:  # 指定步数训练
            bar.update(self.step_count - last_step)
            last_step = self.step_count
            self.train_episode()  # 训练一个回合
            if self.episode_count % self.trainer.params.eval_period == 0:  # 每隔一定回合测试一次智能体模型
                self.test_episode()  # 测试智能体模型的性能
            self.stats.save_if_best()  # 如果当前代理模型是最好的，则保存它
        self.stats.training_ended()  # 在统计器中记录训练结束

    def train_episode(self):
        """ 根据经验回放的在线学习方法训练一个回合 """
        state = self.reset()  # 初始化回合
        self.stats.on_episode_begin() # 统计器记录回合开始
        while not state.is_terminal():
            decode_state = self.decode_state(state)
            action = self.agent.act(decode_state)  # 智能体根据状态选择一个动作（探索性）
            state_ = state.copy()
            next_state, reward, done = self.step(action, pre_state=state_)  # 执行动作并获取新状态、奖励等
            decode_next_state = self.decode_state(next_state)
            self.trainer.add_experience(decode_state, action, reward, decode_next_state,
                                        done)  # 将状态转移、奖励和终止标志添加到经验回放缓存中
            self.stats.add_experience((state_, action, reward, next_state.copy()))  # 将状态转移、奖励添加到统计器中
            self.trainer.train_agent()  # 训练智能体
            state = next_state
            self.step_count += 1 # 训练步数加1
        # self.stats.on_episode_end(self.episode_count) # 统计器记录回合结束 
        self.stats.log_training_data(step=self.step_count)  # 记录训练数据
        self.episode_count += 1  # 增加已完成的回合数

    def test_episode(self, init_state=None):
        """ 测试代理模型在一个回合中的性能 """
        state = self.reset(init_state)  # 初始化回合
        self.stats.on_episode_begin() # 统计器记录回合开始
        while not state.is_terminal():
            action = self.agent.get_exploitation_action_target(
                self.decode_state(state))  # 智能体根据状态选择一个动作（利用性）
            state_ = state.copy()
            next_state, reward, done = self.step(action, pre_state=state_)  # 执行动作并获取新状态、奖励等
            self.stats.add_experience((state_, action, reward, next_state.copy()))  # 将状态转移、奖励和终止标志添加到统计器中
            state = next_state
        # self.stats.on_episode_end(self.episode_count) # 统计器记录回合结束
        self.stats.log_testing_data(step=self.step_count)  # 记录测试数据
        
    def gen_model_graph(self):
        """ 用于生成模型图 """
        state = self.reset()  # 初始化回合
        self.agent.get_exploitation_action(self.decode_state(state))  # 智能体根据状态选择一个动作（探索性）
        
    def decode_state(self, state: State):
        """ 解码state """
        return state.get_boolean_map(),  state.get_float_map(), np.array(state.get_scalars(), dtype=np.single)

    def eval(self, episodes_num: int, init_state=None, show=False):
        """ 评估测试 """
        for i, _ in enumerate(tqdm.tqdm(range(episodes_num))):
            self.test_episode(init_state) # 采集一条轨迹
            self.render(i, show=show)

    def render(self, i, mode="human", show=True):
        """渲染环境"""
        self.display.display_episode(self.grid.map_image, self.stats.trajectory, plot=show, save_path="eval/" + f"{i}" + '.png')

    def close(self):
        """ 关闭环境 """
        pass
