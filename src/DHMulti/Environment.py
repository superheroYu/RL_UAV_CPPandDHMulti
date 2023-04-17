from ..DDQN.Agent import DDQNAgentParams, DDQNAgent
from ..DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from .Physics import DHPhysicsParams
from .Rewards import DHRewardParams, DHRewards
from .State import DHMultiState
from .Display import DHMultiDisplay
from .Grid import DHMultiGrid, DHMultiGridParams
from .Physics import DHPhysics
from ..Base.Environment import Environment, EnvironmentParams
from ..Base.GridActions import GridActions

class DHMultiEnvironmentParams(EnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = DHMultiGridParams()
        self.reward_params = DHRewardParams()
        self.trainer_params = DDQNTrainerParams()
        self.agent_params = DDQNAgentParams()
        self.physics_params = DHPhysicsParams()


class DHMultiEnvironment(Environment):
    def __init__(self, params: DHMultiEnvironmentParams):
        self.display = DHMultiDisplay()
        super().__init__(params, self.display)

        self.grid = DHMultiGrid(params.grid_params, stats=self.stats)
        self.rewards = DHRewards(params.reward_params, stats=self.stats)
        self.physics = DHPhysics(params=params.physics_params,
                                      stats=self.stats)
        self.agent = DDQNAgent(num_actions=len(GridActions.__members__), params=params.agent_params, stats=self.stats)
        self.trainer = DDQNTrainer(params.trainer_params, agent=self.agent)

        self.display.set_channel(self.physics.channel)

    def fill_replay_buffer(self):
        "填充经验回放缓存"
        while self.trainer.should_fill_replay_buffer():  # 当需要经验回放填充时，执行以下操作
            state: DHMultiState = self.reset()  # 初始化一个新状态
            while not state.all_terminal:
                for state.active_agent in range(state.num_agents):
                    if state.terminal:
                        continue
                    action = self.agent.get_random_action()  # 智能体随机获取动作
                    state_decode = self.decode_state(state)
                    next_state, reward, done = self.step(
                        action=action, pre_state=state.copy())  # 执行动作并新状态、奖励、终止标志和信息
                    self.trainer.add_experience(
                        state_decode, action, reward,
                        self.decode_state(next_state),
                        done)  # 将状态转移、奖励和终止标志添加到经验回放缓存中
                    state = next_state

    def train_episode(self):
        """ 根据经验回放的在线学习方法训练一个回合 """
        state: DHMultiState = self.reset()  # 初始化回合
        self.stats.on_episode_begin()  # 统计器记录回合开始
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                decode_state = self.decode_state(state)
                action = self.agent.act(decode_state)  # 智能体根据状态选择一个动作（探索性）
                state_ = state.copy()
                next_state, reward, done = self.step(action, state_)
                decode_next_state = self.decode_state(next_state)
                self.trainer.add_experience(decode_state, action, reward,
                                            decode_next_state,
                                            done)  # 将状态转移、奖励和终止标志添加到经验回放缓存中
                self.stats.add_experience(
                    (state_, action, reward, next_state.copy()))  # 将状态转移、奖励添加到统计器中
                state = next_state
            self.trainer.train_agent()  # 训练智能体
            self.step_count += 1  # 训练步数加1
        self.episode_count += 1  # 增加已完成的回合数
        # self.stats.on_episode_end(self.episode_count) # 统计器记录回合结束
        self.stats.log_training_data(step=self.step_count)  # 记录训练数据
        self.episode_count += 1  # 增加已完成的回合数
        
    def test_episode(self, init_state=None):
        """ 测试一个回合 """
        state: DHMultiState = self.reset(init_state)
        self.stats.on_episode_begin()
        while not state.all_terminal:
            for state.active_agent in range(state.num_agents):
                if state.terminal:
                    continue
                action = self.agent.get_exploitation_action_target(self.decode_state(state))
                state_ = state.copy()
                next_state, reward, done = self.step(action, state_)
                self.stats.add_experience((state_, action, reward, next_state.copy()))
                state = next_state
        self.stats.log_testing_data(step=self.step_count)
