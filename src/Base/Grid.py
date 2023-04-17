from ..UAVMap.Map import load_uavMap
from ..ModelStats import ModelStats


class GridParams:
    def __init__(self):
        self.movement_range = (100, 200) # 定义运动范围
        self.map_path = 'res/downtown.png' # 定义地图文件路径

class Grid:
    def __init__(self, params: GridParams, stats: ModelStats):
        self.map_image = load_uavMap(params.map_path) # 加载地图
        self.shape = self.map_image.start_land_zone.shape # 获取起始着陆区域的形状
        self.starting_vector = self.map_image.get_slz_vector() # 获取起始着陆区域的向量
        stats.set_env_map_callback(self.get_map_image) # 设置环境地图回调函数

    def get_map_image(self):
        return self.map_image  # 返回地图图像

    def get_grid_size(self):
        return self.shape  # 返回网格尺寸

    def get_no_fly(self):
        return self.map_image.nfz  # 返回禁飞区域

    def get_landing_zone(self):
        return self.map_image.start_land_zone  # 返回起始着陆区域