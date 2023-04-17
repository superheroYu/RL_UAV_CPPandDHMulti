import numpy as np
from .Channel import ChannelParams, Channel
from .State import DHMultiState
from ..ModelStats import ModelStats
from ..Base.GridActions import GridActions
from ..Base.GridPhysics import GridPhysics


# DHPhysicsParams 类定义了物理参数，包括信道参数和通信步数
class DHPhysicsParams:
    def __init__(self):
        self.channel_params = ChannelParams()  # 信道参数
        self.comm_steps = 4  # 通信步数

# DHPhysics 类实现了无人机在离散网格中的物理模型，继承自 GridPhysics 类
class DHPhysics(GridPhysics):
    # 初始化函数，传入物理参数和统计模型
    def __init__(self, params: DHPhysicsParams, stats: ModelStats):
        super().__init__()
        self.channel = Channel(params.channel_params)  # 初始化信道
        self.params = params  # 保存物理参数
        self.register_functions(stats)  # 注册统计和评估函数

    # register_functions 函数用于将回调函数注册到统计模型中
    def register_functions(self, stats: ModelStats):
        # 注册评估值回调
        stats.set_evaluation_value_callback(self.get_cral)
        # 注册日志数据回调
        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_collection_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)

    # 重置状态，传入一个 DHState 类型的对象
    def reset(self, state: DHMultiState):
        GridPhysics.reset(self, state)  # 调用父类的重置函数
        self.channel.reset(self.state.shape[0])  # 重置信道

    # 根据动作更新状态，传入一个 GridActions 类型的对象
    def step(self, action: GridActions):
        old_position = self.state.position  # 保存当前位置
        self.movement_step(action)  # 执行移动步骤
        if not self.state.terminal:  # 如果状态不是终止状态，执行通信步骤
            self.comm_step(old_position)
        return self.state

    # 通信步骤，传入上一时刻的位置
    def comm_step(self, old_position):
        # 计算从当前位置到上一时刻位置之间的所有中间位置
        positions = list(reversed(np.linspace(self.state.position, old_position, num=self.params.comm_steps, endpoint=False)))
        indices = []  # 用于存储设备索引
        device_list = self.state.device_list  # 获取设备列表
        for position in positions:  # 对于每个中间位置
            data_rate, idx = device_list.get_best_data_rate(position, self.channel)  # 计算最佳数据速率和对应设备的索引
            device_list.collect_data(data_rate, idx)  # 收集数据
            indices.append(idx)  # 保存设备索引

        # 更新状态中的已收集数据地图和设备数据地图
        self.state.collected = device_list.get_collected_map(self.state.shape)
        self.state.device_map = device_list.get_data_map(self.state.shape)

        # 找到收集数据次数最多的设备索引，并将其设为当前与无人机通信的设备
        idx = max(set(indices), key=indices.count)
        self.state.set_device_com(idx)
        return idx

    def is_in_landing_zone(self):
        """ 检查无人机当前位置是否在降落区域内 """
        return self.state.is_in_landing_zone()

    def get_collection_ratio(self):
        """ 获取采集率，即已收集数据量与总数据量之比 """
        return self.state.get_collection_ratio()

    def get_max_rate(self):
        """ 获取信道的最大速率 """
        return self.channel.get_max_rate()

    def get_average_data_rate(self):
        """ 获取平均数据速率，即已收集数据量与已使用移动预算之比 """
        return self.state.get_collected_data() / self.get_movement_budget_used()

    def get_boundary_counter(self):
        """ 获取边界计数器，用于统计无人机越界的次数 """
        return self.boundary_counter

    def get_landing_attempts(self):
        """ 获取降落尝试次数 """
        return self.landing_attempts

    def get_movement_budget_used(self):  # 重写 get_movement_budget_used 方法，适用于多无人机情况
        return sum(self.state.initial_movement_budgets) - sum(self.state.movement_budgets)

    def get_cral(self):  # 重写 get_cral 方法，适用于多无人机情况
        """ 获取采集率与降落状态乘积的指标 """
        return self.get_collection_ratio() * self.state.all_landed

    def get_movement_ratio(self):  # 重写 get_movement_ratio 方法，适用于多无人机情况
        """ 获取移动比率，即已使用移动预算与初始移动预算之比 """
        return float(self.get_movement_budget_used()) / float(sum(self.state.initial_movement_budgets))

    def has_landed(self):  # 重写 has_landed 方法，适用于多无人机情况
        """ 检查无人机是否已降落 """
        return self.state.all_landed
