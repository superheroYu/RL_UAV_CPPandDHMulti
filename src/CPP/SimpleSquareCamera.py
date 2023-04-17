import numpy as np
from ..UAVMap.Shadowing import load_create_shadowing
from ..UAVMap.Map import load_uavMap


class SimpleSquareCameraParams:
    def __init__(self):
        self.half_length = 2 # 定义相机视野的一半长度
        self.map_path = "res/downtown.png" # 定义地图文件路径

        
class SimpleSquareCamera:

    def __init__(self, params: SimpleSquareCameraParams):
        self.params = params  # 保存参数实例
        total_map = load_uavMap(self.params.map_path)  # 加载地图
        self.obstacles = total_map.obstacles  # 获取地图中的障碍物
        self.size = self.obstacles.shape[:2]  # 获取地图尺寸
        self.obstruction_map = load_create_shadowing(self.params.map_path)  # 加载地图的遮挡图

    def computeView(self, position, attitude):
        view = np.zeros(self.size, dtype=bool)  # 创建一个与地图尺寸相同的空视图
        camera_width = self.params.half_length  # 获取相机视野的一半长度
        x_pos, y_pos = position[0], position[1]  # 获取当前位置的x和y坐标
        x_size, y_size = self.size[0], self.size[1]  # 获取地图的尺寸

        # 根据位置和相机视野的一半长度计算视图范围，并将该范围内的元素设为True
        view[max(0, y_pos - camera_width):min(y_size, y_pos + camera_width + 1),
            max(0, x_pos - camera_width):min(x_size, x_pos + camera_width + 1)] = True

        view &= ~self.obstacles  # 将视图中的障碍物设为False
        view &= ~self.obstruction_map[y_pos, x_pos]  # 将视图中的遮挡物设为False
        return view  # 返回计算后的视图