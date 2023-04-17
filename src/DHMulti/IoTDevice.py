import numpy as np

from .Channel import Channel

# 定义IoT设备参数的类
class IoTDeviceParams:
    def __init__(self, position=(0, 0), color='blue', data=15.0):
        self.position = position  # 设备位置 (x, y)
        self.data = data  # 设备数据，表示设备拥有的总数据量
        self.color = color  # 设备颜色，用于可视化时的设备表示颜色


# 定义IoT设备的类
class IoTDevice:
    def __init__(self, params: IoTDeviceParams):
        self.params = params

        self.position = params.position  # 设备固定位置 (x, y)
        self.color = params.color  # 设备颜色

        self.data = params.data  # 设备数据，表示设备拥有的总数据量
        self.collected_data = 0  # 已收集的数据量

    # 收集设备数据的方法
    def collect_data(self, collect):
        if collect == 0:
            return 1
        c = min(collect, self.data - self.collected_data)  # 计算可以收集的数据量
        self.collected_data += c  # 更新已收集的数据量

        # 返回收集比率，即用于通信的时间百分比
        return c / collect

    # 判断设备是否耗尽数据的方法
    @property
    def depleted(self):
        return self.data <= self.collected_data

    # 计算设备的数据传输速率的方法
    def get_data_rate(self, pos, channel: Channel):
        rate = channel.compute_rate(uav_pos=pos, device_pos=self.position)
        return rate


# 定义设备列表的类
class DeviceList:

    def __init__(self, params):
        self.devices = [IoTDevice(device) for device in params]  # 根据设备参数列表创建IoTDevice对象列表

    # 获取设备列表中数据分布的方法
    def get_data_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.data - device.collected_data

        return data_map

    # 获取设备列表中已收集数据分布的方法
    def get_collected_map(self, shape):
        data_map = np.zeros(shape, dtype=float)

        for device in self.devices:
            data_map[device.position[1], device.position[0]] = device.collected_data

        return data_map

    # 获取最佳数据速率以及对应设备索引的方法
    def get_best_data_rate(self, pos, channel: Channel):
        data_rates = np.array(
            [device.get_data_rate(pos, channel) if not device.depleted else 0 for device in self.devices])
        idx = np.argmax(data_rates) if data_rates.any() else -1
        return data_rates[idx], idx

    # 收集特定设备数据的方法
    def collect_data(self, collect, idx):
        ratio = 1
        if idx != -1:
            ratio = self.devices[idx].collect_data(collect)

        return ratio

    # 获取设备列表的方法
    def get_devices(self):
        return self.devices
    
    # 获取特定设备的方法
    def get_device(self, idx):
        return self.devices[idx]

    # 获取设备列表中设备的总数据量的方法
    def get_total_data(self):
        return sum(list([device.data for device in self.devices]))

    # 获取设备列表中已收集的总数据量的方法
    def get_collected_data(self):
        return sum(list([device.collected_data for device in self.devices]))

    # 获取设备列表中设备的数量的方法
    @property
    def num_devices(self):
        return len(self.devices)


   
