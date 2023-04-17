from .State import CPPState
from .SimpleSquareCamera import SimpleSquareCameraParams, SimpleSquareCamera
from ..ModelStats import ModelStats
from ..Base.GridActions import GridActions
from ..Base.GridPhysics import GridPhysics


class CPPPhysicsParams:
    def __init__(self):
        self.camera_params = SimpleSquareCameraParams() # 初始化相机参数

class CPPPhysics(GridPhysics):
    def __init__(self, params: CPPPhysicsParams, stats: ModelStats):
        super().__init__() # 调用父类构造函数
        self.landed = False # 初始化着陆状态为False

        self.camera = None  # 初始化相机为空

        self.params = params  # 存储物理参数

        self.register_functions(stats)  # 注册统计相关函数

    def register_functions(self, stats: ModelStats):
        stats.set_evaluation_value_callback(self.get_cral)  # 设置评估值回调函数

        # 添加各种日志数据回调函数
        stats.add_log_data_callback('cral', self.get_cral)
        stats.add_log_data_callback('cr', self.get_coverage_ratio)
        stats.add_log_data_callback('successful_landing', self.has_landed)
        stats.add_log_data_callback('boundary_counter', self.get_boundary_counter)
        stats.add_log_data_callback('landing_attempts', self.get_landing_attempts)
        stats.add_log_data_callback('movement_ratio', self.get_movement_ratio)

    def reset(self, state: CPPState):
        GridPhysics.reset(self, state)  # 重置父类状态
        self.landed = False  # 重置着陆状态为False

        self.camera = SimpleSquareCamera(self.params.camera_params)  # 初始化相机对象

    def step(self, action: GridActions):
        self.movement_step(action)  # 执行动作步骤
        if not self.state.terminal:  # 如果状态不是终止状态
            self.vision_step()  # 执行视觉步骤

        if self.state.landed:  # 如果状态是已着陆
            self.landed = True  # 更新着陆状态为True

        return self.state.copy()  # 返回状态

    def vision_step(self):
        view = self.camera.computeView(self.state.position, 0)  # 计算视野
        self.state.add_explored(view)  # 添加已探索区域

    def is_in_landing_zone(self):
        """ 是否在着陆区域内 """
        return self.state.is_in_landing_zone() 

    def get_coverage_ratio(self):
        """ 获取覆盖率 """
        return self.state.get_coverage_ratio() 

    def get_movement_budget_used(self):
        """ 获取已使用的移动预算 """
        return self.state.initial_movement_budget - self.state.movement_budget

    def get_cral(self):
        """ 获取覆盖率乘以着陆状态 """
        return self.get_coverage_ratio() * self.landed 

    def get_boundary_counter(self):
        """ 获取边界碰撞次数 """
        return self.boundary_counter

    def get_landing_attempts(self):
        """ 获取着陆尝试次数 """
        return self.landing_attempts

    def get_movement_ratio(self):
        """ 获取已使用移动预算占初始移动预算的比例 """
        return float(self.get_movement_budget_used()) / float(self.state.initial_movement_budget)

    def has_landed(self):
        """ 判断是否已着陆 """
        return self.landed
