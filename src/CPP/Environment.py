from ..DDQN.Agent import DDQNAgentParams, DDQNAgent
from .Display import CPPDisplay
from .Grid import CPPGrid, CPPGridParams
from .Physics import CPPPhysics, CPPPhysicsParams
from .Rewards import CPPRewardParams, CPPRewards

from ..DDQN.Trainer import DDQNTrainerParams, DDQNTrainer
from ..Base.Environment import Environment, EnvironmentParams
from ..Base.GridActions import GridActions


class CPPEnvironmentParams(EnvironmentParams):
    def __init__(self):
        super().__init__()
        self.grid_params = CPPGridParams() # 网格环境参数
        self.reward_params = CPPRewardParams() # 奖励设置参数
        self.trainer_params = DDQNTrainerParams() # 训练器设置参数
        self.agent_params = DDQNAgentParams() # 智能体设置参数
        self.physics_params = CPPPhysicsParams() # 物理引擎设置参数


class CPPEnvironment(Environment):

    def __init__(self, params: CPPEnvironmentParams):
        self.display = CPPDisplay() # 实例化展示器对象
        super().__init__(params, self.display) # 继承父类
        self.grid = CPPGrid(params.grid_params, self.stats) # 实例化网格游戏对象 
        self.rewards = CPPRewards(params.reward_params, stats=self.stats) # 实例化奖励器对象
        self.physics = CPPPhysics(params=params.physics_params, stats=self.stats) # 实例化物理引擎对象
        self.agent = DDQNAgent(num_actions=len(GridActions.__members__), params=params.agent_params, stats=self.stats) # 实例化DDQN智能体
        self.trainer = DDQNTrainer(params=params.trainer_params, agent=self.agent) # 实例化训练器