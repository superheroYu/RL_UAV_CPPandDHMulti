import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from .MapProcess import Blind_Model, Scalars_Model, Relative_Scalars_Model, Total_Map_Model, Global_Map_Model, Local_Map_Model
from ..ModelStats import ModelStats
import torch.nn.functional as F


class DDQNAgentParams:
    def __init__(self):
        self.scales_len = 3

        # 卷积部分配置
        self.in_channels = 4
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # 全连接层配置
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # 训练参数
        self.learning_rate = 3e-5  # 学习率
        self.tau = 0.005  # 软更新系数
        self.gamma = 0.95  # 折扣系数

        # 探索策略
        self.soft_max_scaling = 0.1

        # 全局-局部地图
        self.global_map_scaling = 3
        self.local_map_size = 17

        #
        self.use_scalar_input = False
        self.relative_scalars = False
        self.blind_agent = False
        self.max_uavs = 3
        self.max_devices = 10

        # 输出
        self.print_summary = False
        
        # 恢复模型的位置
        self.resume = None


class DDQNAgent(object):
    def __init__(self,
                 num_actions: int,
                 params: DDQNAgentParams,
                 stats: ModelStats = None):  #example_state, example_action,
        self.num_actions = num_actions  # 动作空间大小
        self.params = params  # 提前设置的参数

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        # 定义模型
        self.q_network = self.build_model(if_q_net=True)  # Q网络
        self.target_network = self.build_model()  #目标网络
        self.hard_update()  # 硬更新target网络

        # 定义优化器
        self.optimizer = optim.Adam(self.q_network.parameters(),
                                    lr=self.params.learning_rate)
        if stats:
            stats.set_model(self.target_network)

        # 添加模型实例化
        # 全局地图模型
        self.global_map_model = Global_Map_Model(
            self.params.in_channels, self.params.conv_kernels,
            self.params.conv_kernel_size, self.params.conv_layers,
            self.params.global_map_scaling).to(self.device)
        # 局部地图模型
        self.local_map_model = Local_Map_Model(
            self.params.in_channels, self.params.conv_kernels,
            self.params.conv_kernel_size, self.params.conv_layers,
            self.params.local_map_size).to(self.device)
        # 总地图模型
        self.total_map_model = Total_Map_Model().to(self.device)

    def build_model(self, if_q_net=False):
        """ 构建模型 """
        if self.params.blind_agent:
            return Blind_Model(self.params.scales_len,
                               self.params.hidden_layer_size,
                               self.params.hidden_layer_num,
                               self.num_actions).to(self.device)
        elif self.params.use_scalar_input:
            return Scalars_Model(
                self.params.scales_len + 3 * self.params.max_devices +
                4 * self.params.max_uavs, self.params.hidden_layer_size,
                self.params.hidden_layer_num, self.num_actions).to(self.device)
        else:
            if if_q_net:
                return Relative_Scalars_Model(self.num_actions, self.params, self.device, agent=self).to(self.device)
            else:
                return Relative_Scalars_Model(self.num_actions, self.params, self.device).to(self.device)     

    @torch.no_grad()
    def predict(self, state):
        # 预测Q值
        output = self.q_network.forward(*state)
        return output

    def train(self, experiences):
        # 训练网络
        states, actions, rewards, next_states, dones = experiences  # 将经验分为state、action、reward、next_state、done
        # 预处理状态和下一时刻状态
        states = self.process_states(state=states)
        next_states = self.process_states(state=next_states)
        actions, rewards, dones = self.process_batch_data(
            actions, rewards, dones)
        loss = self.get_q_loss(states=states,
                               actions=actions,
                               rewards=rewards,
                               next_states=next_states,
                               dones=dones)
        # 清除优化器的梯度
        self.optimizer.zero_grad()
        # 使用损失（loss）进行反向传播以计算梯度
        loss.backward()
        # 使用优化器更新网络权重
        self.optimizer.step()
        # 软更新目标网络
        self.soft_update(self.params.tau)

    def act(self, state):
        """ 使用模型选择动作 """
        return self.get_soft_max_exploration(state)

    def get_random_action(self):
        """ 随机选择动作 """
        return np.random.randint(0, self.num_actions)

    @torch.no_grad()
    def get_exploitation_action(self, state):
        """ 获取根据网络输出最优动作 """
        state = self.process_state(state)
        q_values = self.q_network(*state)
        return torch.argmax(q_values).item()

    @torch.no_grad()
    def get_soft_max_exploration(self, state):
        """ 根据输入状态和温度，返回softmax策略 """
        state = self.process_state(state)
        q_values = self.q_network(*state)
        p = torch.softmax(q_values / self.params.soft_max_scaling,
                          dim=1).cpu().numpy()[0]  # 每个动作的抽样概率
        return np.random.choice(range(self.num_actions), size=1, p=p)[0]

    @torch.no_grad()
    def get_exploitation_action_target(self, state):
        """ 根据输入状态，使用目标网络返回具有最高Q值的动作 """
        state = self.process_state(state)
        q_values = self.target_network(*state)
        return torch.argmax(q_values).item()

    def get_q_star(self, states):
        """ 计算目标网络的最大动作值（采用DDQN的方法，用Q网络来选择最大Q值动作 """
        q_values = self.q_network(*states)
        q_target_values = self.target_network(*states)
        max_action = torch.argmax(q_values, dim=1)
        q_star = torch.gather(q_target_values, 1, max_action.unsqueeze(1))
        return q_star

    def get_q_loss(self, states, actions, rewards, next_states, dones):
        """ Q_new(s, a) = Q_old(s, a) + α * (reward + γ * max_a' Q_target(s', a') - Q_old(s, a)) 
            or Q_new(s, a) = reward + γ * max_a' Q_target(s', a') * (1 - done)
        """
        actions = actions.long()
        q_values = self.q_network(*states)  # 使用 Q 网络计算给定状态下每个动作的 Q 值
        q_star = self.get_q_star(
            next_states)  # 计算给定状态下目标网络的最大动作值，优化器中不包含目标网络参数所以不需要截断
        gamma_terminated = (1 -
                            dones) * self.params.gamma  # 计算 gamma 的折扣因子，考虑终止状态
        q_update = rewards + q_star * gamma_terminated  # 计算更新值（即：reward + 折扣因子 * q_star）
        q_values_selected = torch.gather(q_values, 1,
                                         actions.view(-1, 1))  # 选定动作的q_value值
        q_loss = F.mse_loss(q_update, q_values_selected,
                            reduction='mean')  # 计算选定动作的Q值和更新Q值之间的均方误差
        return q_loss

    def hard_update(self):
        """ 将目标网络的权重设置为与当前网络相同 """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def soft_update(self, tau):
        """ 使用给定的tau值，将目标网络的权重逐步更新为与当前网络相同 """
        for target_param, param in zip(self.target_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) +
                                    param.data * tau)

    def save_weights(self, path):
        """ 将模型的权重保存到指定路径 """
        torch.save(self.target_network.state_dict(),
                   os.path.join(path, 'model_weights.pth'))

    def load_weights(self, path):
        """ 从指定路径加载模型的权重 """
        self.q_network.load_state_dict(torch.load(path))
        self.hard_update()  # 将q_network的参数复制到target_network
        

    def process_data(self, *data):
        """ 将一条数据转移到device上 """
        for d in data:
            yield torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(
                self.device)

    def process_state(self, state):
        """ 预处理一条状态 """
        state = list(self.process_data(*state))
        for i in range(2):
            state[i] = state[i].permute(0, 3, 1, 2)
        return state

    def process_batch_data(self, *data):
        """ 将批数据转移到device上 """
        for d in data:
            yield torch.tensor(d, dtype=torch.float32).to(self.device)

    def process_states(self, state):
        """ 预处理一批状态 """
        state = list(self.process_batch_data(*state))
        for i in range(2):
            state[i] = state[i].permute(0, 3, 1, 2)
        return state

    def get_global_map(self, state):
        """ 获取全局地图 """
        boolean_map_in, float_map_in = self.process_data(
            state.get_boolean_map(), state.get_float_map())
        return self.global_map_model([boolean_map_in, float_map_in
                                      ]).detach().permute(0, 2, 3, 1).numpy()

    def get_local_map(self, state):
        """ 获取局部地图 """
        boolean_map_in, float_map_in = self.process_data(
            state.get_boolean_map(), state.get_float_map())
        return self.local_map_model([boolean_map_in, float_map_in
                                     ]).detach().permute(0, 2, 3, 1).numpy()

    def get_total_map(self, state):
        """ 获取总地图 """
        boolean_map_in, float_map_in = self.process_data(
            state.get_boolean_map(), state.get_float_map())
        return self.total_map_model([boolean_map_in, float_map_in
                                     ]).detach().permute(0, 2, 3, 1).numpy()
        
    def set_optim(self):
        # 定义优化器
        self.optimizer = optim.Adam(self.q_network.parameters(),lr=self.params.learning_rate)
        
    def resume_model(self):
        # 恢复模型
        if self.params.resume != None:
            self.load_weights(self.params.resume)
            print("模型已恢复！")
