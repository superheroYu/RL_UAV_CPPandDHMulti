"""  
    这个脚本的作用是模拟无线通信信道的传输特性。脚本中定义了两个类：ChannelParams和Channel。

    ChannelParams类：用于定义信道相关的参数，如单元边缘信噪比、视距内/外路径损耗指数、无人机高度、单元格大小以及视距内/外阴影方差等。

    Channel类：根据给定的信道参数（ChannelParams对象）模拟信道特性。主要功能如下：

    初始化时，根据指定的地图路径加载或创建阴影图。
    reset方法用于计算归一化距离、视距内归一化因子以及视距内/外阴影标准差。
    get_max_rate方法计算最大速率，即在给定无人机高度的情况下，信噪比最大时的速率。
    compute_rate方法计算无人机与设备之间的通信速率。首先计算无人机与设备之间的距离，然后根据阴影图判断无人机与设备之间是否存在阴影，最后根据信噪比计算速率。
    总之，这个脚本主要用于模拟无线通信信道，帮助研究无人机与设备之间的通信特性，如速率、路径损耗和阴影效应等。
"""

import numpy as np  # 导入NumPy库，用于数组操作和数值运算
from ..UAVMap.Shadowing import load_create_shadowing  # 从Shadowing模块导入load_or_create_shadowing函数，用于加载或创建阴影图


class ChannelParams:
    def __init__(self):
        self.cell_edge_snr = -25  # 单元边缘信噪比，单位为分贝 (dB)
        self.los_path_loss_exp = 2.27  # 视距内路径损耗指数
        self.nlos_path_loss_exp = 3.64  # 视距外路径损耗指数
        self.uav_altitude = 10.0  # 无人机高度，单位为米 (m)
        self.cell_size = 10.0  # 单元格大小，单位为米 (m)
        self.los_shadowing_variance = 2.0  # 视距内阴影方差
        self.nlos_shadowing_variance = 5.0  # 视距外阴影方差
        self.map_path = "res/downtown.png"  # 地图文件路径


class Channel:
    def __init__(self, params: ChannelParams):
        self.params = params  # 通道参数
        self._norm_distance = None  # 归一化距离
        self.los_norm_factor = None  # 视距内归一化因子
        self.los_shadowing_sigma = None  # 视距内阴影标准差
        self.nlos_shadowing_sigma = None  # 视距外阴影标准差
        self.total_shadow_map = load_create_shadowing(
            self.params.map_path)  # 加载或创建总阴影图

    def reset(self, area_size):
        self._norm_distance = np.sqrt(
            2) * 0.5 * area_size * self.params.cell_size  # 计算归一化距离
        self.los_norm_factor = 10**(self.params.cell_edge_snr / 10) / (
            self._norm_distance**(-self.params.los_path_loss_exp)
        )  # 计算视距内归一化因子
        self.los_shadowing_sigma = np.sqrt(
            self.params.los_shadowing_variance)  # 计算视距内阴影标准差
        self.nlos_shadowing_sigma = np.sqrt(
            self.params.nlos_shadowing_variance)  # 计算视距外阴影标准差

    def get_max_rate(self):
        dist = self.params.uav_altitude  # 获取无人机高度

        snr = self.los_norm_factor * dist**(-self.params.los_path_loss_exp
                                            )  # 计算信噪比

        rate = np.log2(1 + snr)  # 计算最大速率

        return rate  # 返回最大速率

    def compute_rate(self, uav_pos, device_pos):
        # 计算无人机与设备之间的距离
        dist = np.sqrt((
            (device_pos[0] - uav_pos[0]) * self.params.cell_size)**2 + (
                (device_pos[1] - uav_pos[1]) * self.params.cell_size)**2 +
                       self.params.uav_altitude**2)

        # 判断无人机与设备之间是否存在阴影
        if self.total_shadow_map[int(round(device_pos[1])),
                                 int(round(device_pos[0])),
                                 int(round(uav_pos[1])),
                                 int(round(uav_pos[0]))]:
            # 如果存在阴影，计算信噪比（考虑视距外路径损耗和阴影）
            snr = self.los_norm_factor * dist**(
                -self.params.nlos_path_loss_exp) * 10**(
                    np.random.normal(0., self.nlos_shadowing_sigma) / 10)
        else:
            # 如果不存在阴影，计算信噪比（考虑视距内路径损耗和阴影）
            snr = self.los_norm_factor * dist**(
                -self.params.los_path_loss_exp) * 10**(
                    np.random.normal(0., self.los_shadowing_sigma) / 10)

        rate = np.log2(1 + snr)  # 根据信噪比计算速率

        return rate  # 返回速率