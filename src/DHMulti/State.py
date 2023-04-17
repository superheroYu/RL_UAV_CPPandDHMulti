import numpy as np  # 导入numpy库，进行矩阵和数组操作
from ..UAVMap.Map import UAVMap  # 导入UAVMap类，处理地图信息
from src.StateUtils import pad_centered  # 导入pad_centered函数，进行地图数据填充
from ..Base.State import State  # 导入State基类，定义环境状态

# 定义了一个描述设备和数据相关的场景类
class DHScenario:
    def __init__(self):
        self.device_idcs = []  # 设备索引列表
        self.device_data = []  # 设备数据列表
        self.position_idx = 0  # 位置索引
        self.movement_budget = 100  # 移动预算

class DHMultiState(State):  # 定义一个名为DHMultiState的类，继承自State类
    def __init__(self, map_init: UAVMap, num_agents: int):  # 初始化方法，传入UAVMap对象和代理数量
        super().__init__(map_init)  # 调用父类的初始化方法
        self.device_list = None  # 初始化设备列表为空
        self.device_map = None  # 初始化设备地图为空

        self.active_agent = 0  # 初始化当前活动的代理索引为0
        self.num_agents = num_agents  # 设置代理数量

        self.positions = [[0, 0]] * num_agents  # 初始化代理位置列表
        self.movement_budgets = [0] * num_agents  # 初始化代理移动预算列表
        self.landeds = [False] * num_agents  # 初始化代理着陆状态列表
        self.terminals = [False] * num_agents  # 初始化代理终止状态列表
        self.device_coms = [-1] * num_agents  # 初始化代理设备通信列表

        self.initial_movement_budgets = [0] * num_agents  # 初始化代理初始移动预算列表
        self.initial_total_data = 0  # 初始化总数据量为0
        self.collected = None  # 初始化已收集数据矩阵为空

    @property
    def position(self):
        """ 获取当前活动代理的位置 """
        return self.positions[self.active_agent]

    @property
    def movement_budget(self):  
        """ 获取当前活动代理的移动预算 """
        return self.movement_budgets[self.active_agent]

    @property
    def initial_movement_budget(self):
        """ 获取当前活动代理的初始移动预算 """
        return self.initial_movement_budgets[self.active_agent]

    @property
    def landed(self):
        """ 获取当前活动代理的着陆状态 """
        return self.landeds[self.active_agent]

    @property
    def terminal(self):
        """ 获取当前活动代理的终止状态 """
        return self.terminals[self.active_agent]

    @property
    def all_landed(self):
        """ 判断所有代理是否都已着陆 """
        return all(self.landeds)

    @property
    def all_terminal(self):
        """ 判断所有代理是否都已到达终止状态 """
        return all(self.terminals)

    def is_terminal(self):
        """ 判断整个环境是否处于终止状态 """
        return self.all_terminal

    def set_landed(self, landed):
        """ 设置当前活动代理的着陆状态 """
        self.landeds[self.active_agent] = landed

    def set_position(self, position):
        """ 设置当前活动代理的位置 """
        self.positions[self.active_agent] = position
        
    def decrement_movement_budget(self): 
        """ 当前活动代理的移动预算减1 """
        self.movement_budgets[self.active_agent] -= 1

    def set_terminal(self, terminal): 
        """ 设置当前活动代理的终止状态 """
        self.terminals[self.active_agent] = terminal

    def set_device_com(self, device_com):  
        """ 设置当前活动代理的设备通信 """
        self.device_coms[self.active_agent] = device_com

    def get_active_agent(self): 
        """ 获取当前活动代理的索引 """
        return self.active_agent

    def get_remaining_data(self):  
        """ 获取剩余数据量 """
        return np.sum(self.device_map)

    def get_total_data(self):  
        """ 获取初始总数据量 """
        return self.initial_total_data

    def get_scalars(self): 
        """ 获取标量值，不包括位置 """
        return np.array([self.movement_budget])

    def get_num_scalars(self): 
        """ 获取标量值的数量 """
        return len(self.get_scalars())

    def get_boolean_map(self):
        """ 获取布尔地图 """
        padded_red = pad_centered(self, np.concatenate([np.expand_dims(self.no_fly_zone, -1),
                                                        np.expand_dims(self.obstacles, -1)], axis=-1), 1)
        padded_rest = pad_centered(self,
                                np.concatenate(
                                    [np.expand_dims(self.landing_zone, -1), self.get_agent_bool_maps()],
                                    axis=-1), 0)
        return np.concatenate([padded_red, padded_rest], axis=-1)

    def get_boolean_map_shape(self): 
        """ 获取布尔地图的形状 """
        return self.get_boolean_map().shape

    def get_float_map(self): 
        """ 获取浮点地图 """
        return pad_centered(self, np.concatenate([np.expand_dims(self.device_map, -1),
                                                self.get_agent_float_maps()], axis=-1), 0)

    def get_float_map_shape(self):
        """ 获取浮点地图的形状 """
        return self.get_float_map().shape

    def is_in_landing_zone(self):
        """ 判断当前活动代理是否在着陆区域内 """
        return self.landing_zone[self.position[1]][self.position[0]]

    def is_in_no_fly_zone(self): 
        """ 判断当前活动代理是否在禁飞区域内 """
        if 0 <= self.position[1] < self.no_fly_zone.shape[0] and 0 <= self.position[0] < self.no_fly_zone.shape[1]:
            return self.no_fly_zone[self.position[1], self.position[0]] or self.is_occupied()
        return True

    def is_occupied(self):  
        """ 判断当前活动代理的位置是否被其他代理占据 """
        for i, pos in enumerate(self.positions):
            if self.terminals[i]:
                continue
            if i == self.active_agent:
                continue
            if pos == self.position:
                return True
        return False

    def get_collection_ratio(self): 
        """ 获取已收集数据与初始总数据量的比例 """
        return np.sum(self.collected) / self.initial_total_data

    def get_collected_data(self):  
        """ 获取已收集的数据量 """
        return np.sum(self.collected)

    def reset_devices(self, device_list): 
        """ 重置设备列表 """
        self.device_map = device_list.get_data_map(self.no_fly_zone.shape)  # 根据设备列表重置设备地图
        self.collected = np.zeros(self.no_fly_zone.shape, dtype=float)  # 初始化已收集数据矩阵
        self.initial_total_data = device_list.get_total_data()  # 获取初始总数据量
        self.device_list = device_list  # 设置设备列表

    def get_agent_bool_maps(self): 
        """ 获取代理布尔地图 """
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=bool)  # 初始化代理地图
        for agent in range(self.num_agents):  # 遍历代理
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = not self.terminals[agent]
        return agent_map

    def get_agent_float_maps(self):  
        """ 获取代理浮点地图 """
        agent_map = np.zeros(self.no_fly_zone.shape + (1,), dtype=float)  # 初始化代理地图
        for agent in range(self.num_agents):  # 遍历代理
            agent_map[self.positions[agent][1], self.positions[agent][0]][0] = self.movement_budgets[agent]
        return agent_map

    def get_device_scalars(self, max_num_devices, relative):
        devices = np.zeros(3 * max_num_devices, dtype=np.float32)
        if relative:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0] - self.position[0]
                devices[k * 3 + 1] = dev.position[1] - self.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data
        else:
            for k, dev in enumerate(self.device_list.devices):
                devices[k * 3] = dev.position[0]
                devices[k * 3 + 1] = dev.position[1]
                devices[k * 3 + 2] = dev.data - dev.collected_data
        return devices

    def get_uav_scalars(self, max_num_uavs, relative):
        uavs = np.zeros(4 * max_num_uavs, dtype=np.float32)
        if relative:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0] - self.position[0]
                uavs[k * 4 + 1] = self.positions[k][1] - self.position[1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        else:
            for k in range(max_num_uavs):
                if k >= self.num_agents:
                    break
                uavs[k * 4] = self.positions[k][0]
                uavs[k * 4 + 1] = self.positions[k][1]
                uavs[k * 4 + 2] = self.movement_budgets[k]
                uavs[k * 4 + 3] = not self.terminals[k]
        return uavs

