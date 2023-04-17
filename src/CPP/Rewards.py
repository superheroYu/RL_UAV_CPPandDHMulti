from .State import CPPState
from ..Base.GridActions import GridActions
from ..Base.Rewards import GridRewardParams, GridRewards


class CPPRewardParams(GridRewardParams):
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        self.cell_multiplier = 0.4  # 定义每个单元格的奖励系数


# 用于跟踪奖励的类
class CPPRewards(GridRewards):
    def __init__(self, reward_params: CPPRewardParams, stats):
        super().__init__(stats)  # 调用父类构造函数
        self.params = reward_params  # 保存奖励参数实例
        self.reset()  # 重置累计奖励和其他统计信息

    def calculate_reward(self, state: CPPState, action: GridActions,
                         next_state: CPPState):
        reward = self.calculate_motion_rewards(state, action, next_state)  # 计算运动奖励
        # 根据收集到的数据计算奖励
        reward += self.params.cell_multiplier * (state.get_remaining_cells() - next_state.get_remaining_cells())

        # 累计奖励
        self.cumulative_reward += reward

        return reward  # 返回当前计算的奖励
