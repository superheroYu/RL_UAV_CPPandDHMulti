from typing import Union
import numpy as np
from skimage import io


class UAVMap(object):
    """ UAV任务的地图类 """
    def __init__(self, map_data: np.ndarray):
        self.start_land_zone = map_data[:, :, 2].astype(bool)  # 起飞区
        self.nfz = map_data[:, :, 0].astype(bool)  # 禁飞区
        self.obstacles = map_data[:, :, 1].astype(bool)  # 障碍物

    def get_slz_vector(self):
        """ 得到起飞区的位置向量 """
        slz = np.where(self.start_land_zone)
        return list(zip(slz[1], slz[0]))

    def get_free_space_vector(self):
        """ 得到飞行区的位置向量 """
        free_space = np.logical_not(
            np.logical_or(self.obstacles, self.start_land_zone))
        fs = np.where(free_space)
        return list(zip(fs[1], fs[0]))

    def size(self):
        return self.start_land_zone.shape[:2]


def load_image(path: str) -> np.ndarray:
    """ 加载图片 """
    return np.array(io.imread(path, as_gray=True))


def save_image(path: str, image: np.ndarray):
    """ 保存图片 """
    if image.dtype == bool:
        io.imsave(path, image * np.uint8(255))
    else:
        io.imsave(path, image)


def load_uavMap(path: str) -> UAVMap:
    """ 加载UAV地图 """
    return UAVMap(io.imread(path, as_gray=False))


def load_target(path: str, obstacles: Union[np.ndarray, None]) -> np.ndarray:
    """ 加载目标区域 """
    target = np.array(io.imread(path, as_gray=True), dtype=bool)
    if obstacles is not None:
        target = target & ~obstacles
    return target


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    path = "uavmap_figure\manhattan32.png"
    fig = load_image(path)
    plt.imshow(fig)
    plt.show()