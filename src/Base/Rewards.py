from .GridActions import GridActions
from ..ModelStats import ModelStats
from .State import State

class GridRewardParams:
    def __init__(self):
        # 碰到边界的惩罚
        self.boundary_penalty = 1.0
        # 电量用尽的惩罚
        self.empty_battery_penalty = 150.0
        # 移动惩罚
        self.movement_penalty = 0.2


class GridRewards:
    def __init__(self, stats:ModelStats, AcitonsEnum = GridActions):
        # 初始化奖励对象
        self.params = GridRewardParams()
        # 累计奖励值
        self.cumulative_reward: float = 0.0
        # 枚举动作
        self.ActionsEnum = AcitonsEnum
        # 将累计奖励值添加到统计对象中
        stats.add_log_data_callback('cumulative_reward', self.get_cumulative_reward)

    def get_cumulative_reward(self):
        # 返回累计奖励值
        return self.cumulative_reward

    def calculate_motion_rewards(self, state:State, action: GridActions, next_state):
        # 初始化奖励值
        reward = 0.0
        if not next_state.landed:
            # 如果下一状态没有降落，则减去电量消耗的惩罚
            reward -= self.params.movement_penalty

        if state.position == next_state.position and not next_state.landed and not action == self.ActionsEnum.HOVER:
            # 如果下一状态和当前状态相同且不是降落或悬停动作，则减去碰到边界的惩罚
            reward -= self.params.boundary_penalty

        if next_state.movement_budget == 0 and not next_state.landed:
            # 如果下一状态的电量用尽且没有降落，则减去电量用尽的惩罚
            reward -= self.params.empty_battery_penalty

        # 更新累计奖励值
        self.cumulative_reward += reward
        return reward

    def reset(self):
        # 将累计奖励值重置为 0
        self.cumulative_reward = 0
        
    def calculate_reward(self, state, action, nextstate) -> float:
        pass
    
if __name__ == "__main__":
    print(GridActions(1))