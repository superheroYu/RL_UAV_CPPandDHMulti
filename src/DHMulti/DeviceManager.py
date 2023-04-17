import numpy as np

from .IoTDevice import IoTDeviceParams, DeviceList

# 定义一个颜色列表，用于为设备分配颜色
ColorMap = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]


class DeviceManagerParams:
    def __init__(self):
        self.device_count_range = (2, 5)  # 设备数量范围
        self.data_range = (5.0, 20.0)  # 设备数据范围
        self.fixed_devices = False  # 是否使用固定设备列表
        self.devices = IoTDeviceParams()  # 设备参数


class DeviceManager:
    """
    The DeviceManager is used to generate DeviceLists according to DeviceManagerParams
    """

    def __init__(self, params: DeviceManagerParams):
        self.params = params  # 初始化设备管理器参数

    def generate_device_list(self, positions_vector):
        if self.params.fixed_devices:
            return DeviceList(self.params.devices)  # 如果使用固定设备列表，则返回固定设备列表

        # 随机生成设备数量
        device_count = np.random.randint(self.params.device_count_range[0], self.params.device_count_range[1] + 1)

        # 随机生成设备位置
        position_idcs = np.random.choice(range(len(positions_vector)), device_count, replace=False)
        positions = [positions_vector[idx] for idx in position_idcs]

        # 随机生成设备数据
        datas = np.random.uniform(self.params.data_range[0], self.params.data_range[1], device_count)

        return self.generate_device_list_from_args(device_count, positions, datas)  # 生成设备列表

    def generate_device_list_from_args(self, device_count, positions, datas):
        # 获取颜色
        colors = ColorMap[0:max(device_count, len(ColorMap))]

        # 创建IoTDeviceParams对象列表
        params = [IoTDeviceParams(position=positions[k],
                                  data=datas[k],
                                  color=colors[k % len(ColorMap)])
                  for k in range(device_count)]

        return DeviceList(params)  # 返回包含IoTDeviceParams对象的设备列表
