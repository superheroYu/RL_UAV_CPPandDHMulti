from ..Base.GridActions import GridActions
from ..Base.Rewards import GridRewards, GridRewardParams
from .State import DHMultiState

# DHRewardParams类定义了奖励参数
class DHRewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()
        # 数据收集奖励系数，默认为1.0
        self.data_multiplier = 1.0

# DHRewards类负责跟踪奖励，继承自GridRewards类
class DHRewards(GridRewards):
    # 累计奖励
    cumulative_reward: float = 0.0

    # 初始化函数
    def __init__(self, reward_params: DHRewardParams, stats):
        super().__init__(stats)
        self.params = reward_params
        self.reset()

    # 计算奖励
    def calculate_reward(self, state: DHMultiState, action: GridActions, next_state: DHMultiState):
        # 计算运动奖励
        reward = self.calculate_motion_rewards(state, action, next_state)

        # 计算数据收集奖励
        reward += self.params.data_multiplier * (state.get_remaining_data() - next_state.get_remaining_data())

        # 更新累计奖励
        self.cumulative_reward += reward

        return reward
