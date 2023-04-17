import numpy as np
import os
import tqdm
from .Map import load_uavMap


class Bresenham:
    """ 布莱森汉姆算法，获取两点之间直线的点 """
    @staticmethod
    def bresenham1(x_start: int, y_start: int, x_end: int, y_end: int,
                   obstacles: np.ndarray, shadow_map: np.ndarray):
        dx, dy = abs(x_end - x_start), abs(y_end - y_start)  # 获取两点X、Y方向的距离
        # 获取方向
        sx = 1 if x_end > x_start else -1
        sy = 1 if y_end > y_start else -1

        err = dx if dx > dy else -dy  # 初始化误差项
        double_dx = dx << 1  # 2 * dx
        double_dy = dy << 1  # 2 * dy

        shadow_map[y_start, x_start] = False  # 设置初始位置不在阴影区域
        while x_start != x_end or y_start != y_end:
            tmp = err
            if tmp > -double_dx:
                err -= double_dy
                x_start += sx
            if tmp < dy:
                err += double_dx
                y_start += sy

            if obstacles[y_start, x_start]:
                return

            shadow_map[y_start, x_start] = False

    @staticmethod
    def bresenham2(x_start: int, y_start: int, x_end: int, y_end: int,
                   obstacles: np.ndarray, shadow_map: np.ndarray):
        dx, dy = abs(x_end - x_start), abs(y_end - y_start)  # 获取两点X、Y方向的距离
        # 获取方向
        sx = 1 if x_end > x_start else -1
        sy = 1 if y_end > y_start else -1

        flag = False  # 标志位，标记是否dx>dy
        if dx < dy:  # 判断是否dx<dy
            dx, dy = dy, dx  # 若是则交互坐标
            flag = True

        deltaY = (dy << 1)  # 2 * dx * dy/dx
        middle = dx  # 0.5 * 2 * dx

        shadow_map[y_start, x_start] = False  # 设置初始位置不在阴影区域
        while x_start != x_end:
            if not flag:
                x_start += sx
            else:
                y_start += sy
            if deltaY > middle:
                if not flag:
                    y_start += sy
                else:
                    x_start += sx
                middle += (dx << 1)
            deltaY += (dy << 1)

            if obstacles[y_start, x_start]:
                return

            shadow_map[y_start, x_start] = False

    @staticmethod
    def bresenham3(x_start: int, y_start: int, x_end: int, y_end: int,
                   obstacles: np.ndarray, shadow_map: np.ndarray):
        dx = abs(x_end - x_start)
        dy = -abs(y_end - y_start)
        sx = 1 if x_end > x_start else -1
        sy = 1 if y_end > y_start else -1

        err = dx + dy

        shadow_map[y_start, x_start] = False
        while x_start != x_end or y_start != y_end:
            if 2 * err - dy > dx - 2 * err:
                err += dy
                x_start += sx
            else:
                err += dx
                y_start += sy

            if obstacles[y_start, x_start]:
                return

            shadow_map[y_start, x_start] = False


def cal_showing(map_path: str, save_path: str):
    print("正在计算阴影地图……")
    total_map = load_uavMap(map_path)
    obstacales = total_map.obstacles
    size = total_map.obstacles.shape[0]
    total = size**2

    total_shadow_map = np.ones((size, size, size, size), dtype=bool)
    with tqdm.tqdm(total=total) as pbar:
        for i, j in np.ndindex(total_map.obstacles.shape):
            shadow_map = np.ones((size, size), dtype=bool)

            ### 进行阴影填充
            for x in range(size):
                Bresenham.bresenham1(i, j, x, 0, obstacales, shadow_map)
                Bresenham.bresenham1(i, j, x, size - 1, obstacales, shadow_map)
                Bresenham.bresenham1(i, j, 0, x, obstacales, shadow_map)
                Bresenham.bresenham1(i, j, size - 1, x, obstacales, shadow_map)

            total_shadow_map[j, i] = shadow_map
            pbar.update(1)

    np.save(save_path, total_shadow_map)
    return total_shadow_map


def load_create_shadowing(map_path: str):
    """ 加载或创建阴影 """
    s = os.path.splitext(map_path)[0] + "_shadowing.npy"
    if os.path.exists(s):
        return np.load(s)
    else:
        return cal_showing(map_path, s)


if __name__ == "__main__":
    path = "uavmap_figure\manhattan32.png"
    save_name = "./test"
    cal_showing(path, save_name)