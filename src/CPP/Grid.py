import numpy as np

from ..UAVMap.Map import load_target
from .State import CPPState, CPPScenario
from .RandomTargetGenerator import RandomTargetGenerator, RandomTargetGeneratorParams
from ..Base.Grid import Grid, GridParams


class CPPGridParams(GridParams):
    def __init__(self):
        super().__init__() # 调用父类构造函数
        self.generator_params = RandomTargetGeneratorParams() # 初始化随机目标生成器参数

class CPPGrid(Grid):

    def __init__(self, params: CPPGridParams, stats):
        super().__init__(params, stats)  # 调用父类构造函数
        self.params = params  # 保存参数实例

        # 创建随机目标生成器，并根据地图尺寸生成目标区域
        self.generator = RandomTargetGenerator(params.generator_params, self.map_image.size())
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

    def init_episode(self):
        # 为每个新的episode生成新的目标区域
        self.target_zone = self.generator.generate_target(self.map_image.obstacles)

        state = CPPState(self.map_image)  # 初始化状态
        state.reset_target(self.target_zone)  # 重置目标

        idx = np.random.randint(0, len(self.starting_vector))  # 随机选择起始位置
        state.position = self.starting_vector[idx]

        # 随机设定运动预算
        state.movement_budget = np.random.randint(low=self.params.movement_range[0],
                                                high=self.params.movement_range[1] + 1)

        state.initial_movement_budget = state.movement_budget
        state.landed = False
        state.terminal = False

        return state  # 返回初始化后的状态

    def create_scenario(self, scenario: CPPScenario):
        state = CPPState(self.map_image)
        target = load_target(scenario.target_path, self.map_image.obstacles)
        state.reset_target(target)
        state.position = self.starting_vector[scenario.position_idx]
        state.movement_budget = scenario.movement_budget
        state.initial_movement_budget = scenario.movement_budget
        return state

    def init_scenario(self, state: CPPState):
        self.target_zone = state.target  # 设置目标区域
        return state

    def get_target_zone(self):
        return self.target_zone  # 返回目标区域

