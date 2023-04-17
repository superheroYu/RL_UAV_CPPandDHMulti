import numpy as np

from ..UAVMap.Map import UAVMap
from ..StateUtils import pad_centered
from ..Base.State import State


class CPPScenario:
    def __init__(self):
        self.target_path = "" # 初始化目标路径为空字符串
        self.position_idx = 0 # 初始化位置索引为0
        self.movement_budget = 100 # 初始化移动预算为100

class CPPState(State):
    def __init__(self, map_init: UAVMap):
        super().__init__(map_init) # 调用父类构造函数
        self.target = None # 初始化目标为空
        self.position = None # 初始化位置为空
        self.movement_budget = None # 初始化移动预算为空
        self.landed = False # 初始化着陆状态为False
        self.terminal = False # 初始化终止状态为False

        self.initial_movement_budget = None  # 初始化初始移动预算为空
        self.initial_target_cell_count = 0  # 初始化初始目标单元格数量为0
        self.coverage = None  # 初始化覆盖区域为空

    def reset_target(self, target):
        self.target = target  # 重置目标
        self.initial_target_cell_count = np.sum(target)  # 计算初始目标单元格数量
        self.coverage = np.zeros(self.target.shape, dtype=bool)  # 初始化覆盖区域为全零布尔数组

    def get_remaining_cells(self):
        return np.sum(self.target)  # 获取剩余未探索的单元格数量

    def get_total_cells(self):
        return self.initial_target_cell_count # 获取初始目标单元格数量

    def get_coverage_ratio(self):
        return 1.0 - float(np.sum(self.get_remaining_cells())) / float(self.initial_target_cell_count)  # 计算覆盖率

    def get_scalars(self):
        return np.array([self.movement_budget])  # 获取移动预算数组

    def get_num_scalars(self):
        return 1  # 获取标量数量

    def get_boolean_map(self):
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        padded_rest = pad_centered(self, np.concatenate([np.expand_dims(self.landing_zone, -1),
                                                        np.expand_dims(self.target, -1)], axis=-1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)  # 获取布尔地图

    def get_boolean_map_shape(self):
        return self.get_boolean_map().shape  # 获取布尔地图形状

    def get_float_map(self):
        shape = list(self.get_boolean_map().shape)
        shape[2] = 0
        float_map = np.zeros(tuple(shape), dtype=float)  # 获取浮点地图
        return float_map

    def get_float_map_shape(self):
        return self.get_float_map().shape  # 获取浮点地图形状

    def is_in_landing_zone(self):
        return self.landing_zone[self.position[1]][self.position[0]]  # 判断是否在着陆区域内

    def is_in_no_fly_zone(self):
        # Out of bounds is implicitly nfz
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.no_fly_zone[self.position[1], self.position[0]]  # 判断是否在禁飞区内
        return True

    def add_explored(self, view):
        self.target &= ~view  # 更新未探索区域
        self.coverage |= view  # 更新覆盖区域

    def set_terminal(self, terminal):
        self.terminal = terminal  # 设置终止状态

    def set_landed(self, landed):
        self.landed = landed  # 设置着陆状态

    def set_position(self, position):
        self.position = position  # 设置位置

    def decrement_movement_budget(self):
        self.movement_budget -= 1  # 减少移动预算

    def is_terminal(self):
        return self.terminal