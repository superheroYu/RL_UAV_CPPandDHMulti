import numpy as np

from .DeviceManager import DeviceManagerParams, DeviceManager
from ..Base.Grid import Grid, GridParams  # 导入 DHGrid 和 DHGridParams 类
from .State import DHMultiState, DHScenario  # 导入 DHMultiState 类

class DHMultiGridParams(GridParams):  # 定义一个新类 DHMultiGridParams，继承自 DHGridParams
    def __init__(self):
        super().__init__()
        self.device_manager = DeviceManagerParams()
        self.num_agents_range = [1, 3]  # 为 num_agents_range 设置默认值 [1, 3]
        self.multi_agent = False
        self.fixed_starting_idcs = False
        self.starting_idcs = [1, 2, 3]

class DHMultiGrid(Grid):  # 定义一个新类 DHMultiGrid，继承自 DHGrid

    def __init__(self, params: DHMultiGridParams, stats):
        super().__init__(params, stats)  # 调用父类的初始化方法，传入参数和统计对象
        self.params = params
        if params.multi_agent:
            self.num_agents = params.num_agents_range[0]
        else:
            self.num_agents = 1
        
         # 初始化设备列表和设备管理器
        self.device_list = None
        self.device_manager = DeviceManager(self.params.device_manager)

        # 计算可用空间并获取设备位置
        free_space = np.logical_not(
            np.logical_or(self.map_image.obstacles, self.map_image.start_land_zone))
        free_idcs = np.where(free_space)
        self.device_positions = list(zip(free_idcs[1], free_idcs[0]))
        
    # 获取通信障碍物
    def get_comm_obstacles(self):
        return self.map_image.obstacles

    # 获取数据地图
    def get_data_map(self):
        return self.device_list.get_data_map(self.shape)

    # 获取已收集数据的地图
    def get_collected_map(self):
        return self.device_list.get_collected_map(self.shape)

    # 获取设备列表
    def get_device_list(self):
        return self.device_list

    # 获取网格参数
    def get_grid_params(self):
        return self.params

    # 创建场景
    def create_scenario(self, scenario: DHScenario):
        state = DHMultiState(self.map_image)
        state.position = self.starting_vector[scenario.position_idx]
        state.movement_budget = scenario.movement_budget
        state.initial_movement_budget = scenario.movement_budget

        positions = [self.device_positions[idx] for idx in scenario.device_idcs]
        state.reset_devices(self.device_manager.generate_device_list_from_args(len(positions), positions,
                                                                               scenario.device_data))
        return state

    # 初始化场景
    def init_scenario(self, state: DHMultiState):
        self.device_list = state.device_list

        return state

    def init_episode(self):  # 重写 init_episode 方法，适用于多无人机情况
        self.device_list = self.device_manager.generate_device_list(self.device_positions)

        self.num_agents = int(np.random.randint(low=self.params.num_agents_range[0],
                                                high=self.params.num_agents_range[1] + 1, size=1))
        state = DHMultiState(self.map_image, self.num_agents)
        state.reset_devices(self.device_list)

        idx = np.random.choice(len(self.starting_vector), size=self.num_agents, replace=False)
        state.positions = [self.starting_vector[i] for i in idx]

        state.movement_budgets = np.random.randint(low=self.params.movement_range[0],
                                                   high=self.params.movement_range[1] + 1, size=self.num_agents)

        state.initial_movement_budgets = state.movement_budgets.copy()

        return state

    def init_scenario(self, scenario):  # 重写 init_scenario 方法，适用于多无人机情况
        self.device_list = scenario.device_list
        self.num_agents = scenario.init_state.num_agents

        return scenario.init_state
