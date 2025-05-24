import numpy as np
from skimage.draw import random_shapes
import logging


class RandomTargetGeneratorParams:
    def __init__(self):
        self.coverage_range = (0.2, 0.8) # 设定覆盖范围参数
        self.shape_range = (1, 5) # 设定形状数量范围参数

class RandomTargetGenerator:

    def __init__(self, params: RandomTargetGeneratorParams, shape):
        self.params = params  # 初始化参数
        self.shape = shape  # 初始化形状

    def generate_target(self, obstacles):
        area = np.product(self.shape)  # 计算形状的总面积

        # 生成随机形状区域
        target = self.__generate_random_shapes_area(
            self.params.shape_range[0],
            self.params.shape_range[1],
            area * self.params.coverage_range[0],
            area * self.params.coverage_range[1]
        )

        return target & ~obstacles  # 返回目标形状，同时排除障碍物区域

    def __generate_random_shapes(self, min_shapes, max_shapes):
        """ 生成随机形状 """
        img, _ = random_shapes(self.shape, max_shapes, min_shapes=min_shapes, channel_axis=None,
                            allow_overlap=True, rng=np.random.randint(2**31 - 1))
        attempt = np.array(img != 255, dtype=bool)  # 转换为布尔类型的数组
        return attempt, np.sum(attempt)  # 返回形状及其面积

    def __generate_random_shapes_area(self, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)  # 生成随机形状
            if min_area is not None and min_area > area:  # 如果面积小于最小要求，跳过
                continue
            if max_area is not None and max_area < area:  # 如果面积大于最大要求，跳过
                continue
            return attempt  # 如果满足要求，则返回形状

        # 超过重试次数，输出警告信息
        logging.warning("无法在允许的尝试次数内生成符合给定面积约束条件的形状。"
                        "随机返回下一次尝试。")
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)  # 生成随机形状
        logging.warning("形状面积: ", area)  # 输出形状面积
        return attempt

    def __generate_exclusive_shapes(self, exclusion, min_shapes, max_shapes):
        attempt, area = self.__generate_random_shapes(min_shapes, max_shapes)  # 生成随机形状
        attempt = attempt & (~exclusion)  # 去除指定区域
        area = np.sum(attempt)  # 计算形状面积
        return attempt, area  # 返回形状及其面积

    # 创建目标图像并减去排除区
    def __generate_exclusive_shapes_area(self, exclusion, min_shapes, max_shapes, min_area, max_area, retry=100):
        for attemptno in range(retry):
            # 生成排除特定区域的随机形状
            attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)
            if min_area is not None and min_area > area:  # 如果面积小于最小要求，跳过
                continue
            if max_area is not None and max_area < area:  # 如果面积大于最大要求，跳过
                continue
            return attempt  # 如果满足要求，则返回形状

        # 超过重试次数，输出警告信息
        logging.warning("无法在允许的尝试次数内生成符合给定面积限制的形状。"
                        "随机返回下一次尝试。")
        attempt, area = self.__generate_exclusive_shapes(exclusion, min_shapes, max_shapes)  # 生成随机形状
        logging.warning("Size is: ", area)  # 输出形状面积
        return attempt