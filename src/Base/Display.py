"""
    这个代码定义了一个名为BaseDisplay的类，它是一个用于显示地图、轨迹和状态的基类。
    这个类有一个__init__方法，用于初始化一些属性，如箭头的大小和标记的大小。
    这个类有一个create_grid_image方法，用于在给定的轴上绘制一个带有不同颜色区域和障碍物的网格图像，并设置坐标轴的刻度和标签。
    这个类有一个draw_start_and_end方法，用于在轨迹的起点和终点处绘制白色或绿色或红色的菱形标记。
    这个类有一个draw_movement方法，用于在给定的位置之间绘制箭头或叉号，表示移动或停留。
    这个类有一个create_torch_image方法，用于将当前图像保存为PNG格式，并返回一个张量。
    这个类有一个create_video方法，用于根据给定的地图图像和轨迹创建并保存一个视频文件。
    这个类还有两个抽象方法display_episode和display_state，需要子类实现。
"""

import io
import torch
import torchvision
import torchvision.transforms.functional as TF
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.color import rgb2hsv, hsv2rgb
from tqdm import tqdm
from ..UAVMap.Map import UAVMap


class Display:
    def __init__(self, arrow_scale=14, marker_size=15):
        self.arrow_scale = arrow_scale  # 设置箭头的缩放比例
        self.marker_size = marker_size  # 设置标记的大小

    def create_grid_image(self,
                          ax: plt.Axes,
                          env_map: UAVMap,
                          value_map: np.ndarray,
                          green: np.ndarray = None):
        """ 用于创建一个网格图像，显示地图中的不同区域（如禁飞区、着陆区等），并根据值映射调整颜色。它还设置了一些绘图参数，如刻度标签、坐标轴范围和障碍物。 """
        area_y_max, area_x_max = env_map.size()
        if green is None:  # 判断二通道是否赋值
            green = np.zeros((area_y_max, area_x_max))

        nfz = np.expand_dims(env_map.nfz, -1)  # 禁飞区
        slz = np.expand_dims(env_map.start_land_zone, -1)  # 起飞区
        green = np.expand_dims(green, -1)  # 图像中的二通道（绿色通道）

        neither = np.logical_not(np.logical_or(np.logical_or(
            nfz, slz), green))  # 地图中既不是禁飞区、也不是起飞区、也不是green参数指定区域的部分

        base = np.zeros((area_y_max, area_x_max, 3))  # 创建一个全零数组作为基础颜色

        nfz_color = base.copy()
        nfz_color[..., 0] = 0.8  # 设置禁飞区的颜色等级(红)

        slz_color = base.copy()
        slz_color[..., 2] = 0.8  # 设置起飞区的颜色等级(蓝)

        green_color = base.copy()
        green_color[..., 1] = 0.8  # 设置绿色通道的颜色等级

        neither_color = np.ones(
            (area_y_max, area_x_max, 3),
            dtype=np.float)  # 设置既不是禁飞区、也不是起飞区、也不是green参数指定区域的部分为白色
        grid_image = green_color * green + nfz_color * nfz + slz_color * slz + neither_color * neither  # 合成网格图像

        # 根据value_map来转换网格图片的明亮度
        hsv_image = rgb2hsv(grid_image)
        hsv_image[..., 2] *= value_map.astype(
            'float32')  # 设置明亮度值(hsv三通道：色调H、饱和度S、明度V)
        grid_image = hsv2rgb(hsv_image)

        ### 自适应设置坐标轴刻度
        if (area_x_max, area_y_max) == (64, 64):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 14
            self.marker_size = 6
        elif (area_x_max, area_y_max) == (32, 32):
            tick_labels_x = np.arange(0, area_x_max, 2)
            tick_labels_y = np.arange(0, area_y_max, 2)
            self.arrow_scale = 8
            self.marker_size = 15
        elif (area_x_max, area_y_max) == (50, 50):
            tick_labels_x = np.arange(0, area_x_max, 4)
            tick_labels_y = np.arange(0, area_y_max, 4)
            self.arrow_scale = 12
            self.marker_size = 8
        else:
            tick_labels_x = np.arange(0, area_x_max, 1)
            tick_labels_y = np.arange(0, area_y_max, 1)
            self.arrow_scale = 5
            self.marker_size = 15

        plt.sca(ax)  # 将ax设为当前轴，这样后续的绘图操作都会在ax上进行
        plt.gca().set_aspect('equal', adjustable='box')  # 设置坐标轴缩放比例为相等
        plt.xticks(tick_labels_x)  # 设置x轴刻度
        plt.yticks(tick_labels_y)  # 设置y轴刻度
        plt.axis([0, area_x_max, area_y_max, 0])  # 设置坐标范围
        ax.imshow(grid_image.astype(float),
                  extent=[0, area_x_max, area_y_max, 0])  # 显示网格图像并设置显示区域
        # plt.axis('off')

        obst = env_map.obstacles  # 障碍物
        for i in range(area_x_max):
            for j in range(area_y_max):
                if obst[j, i]:
                    # (i, j)：矩形的左下角坐标，1：矩形的宽度，1：矩形的高度，fill=None：矩形是否填充颜色，None表示不填充，hatch=‘////’：矩形的填充样式，'////'表示用斜线填充，edgecolor=“Black”：矩形的边框颜色，"Black"表示黑色
                    rect = patches.Rectangle((i, j),
                                             1,
                                             1,
                                             fill=None,
                                             hatch='////',
                                             edgecolor="Black")
                    ax.add_patch(rect)

        # offset to shift tick labels
        locs, labels = plt.xticks()  # 获取x轴刻度的位置和标签
        locs_new = [x + 0.5 for x in locs]  # 给每个位置加上0.5，并赋值给新的变量locs_new
        plt.xticks(locs_new,
                   tick_labels_x)  # 设置x轴刻度的位置和标签为locs_new和tick_labels_x

        locs, labels = plt.yticks()  # 获取y轴刻度的位置和标签
        locs_new = [x + 0.5 for x in locs]  # 给每个位置加上0.5，并赋值给新的变量locs_new
        plt.yticks(locs_new,
                   tick_labels_y)  # 设置y轴刻度的位置和标签为locs_new和tick_labels_y

    def draw_start_end(self, trajectory):
        """ 用于在轨迹的起点和终点处绘制菱形标记 """
        first_state = trajectory[0][0]  # 初始状态
        final_state = trajectory[-1][3]  # 最终状态

        plt.scatter(first_state.position[0] + 0.5,
                    first_state.position[1] + 0.5,
                    s=self.marker_size,
                    marker="D",
                    color="w")

        if final_state.landed:  # 判断最终状态是否着陆
            plt.scatter(final_state.position[0] + 0.5,
                        final_state.position[1] + 0.5,
                        s=self.marker_size,
                        marker="D",
                        color="green")  # 着陆用绿色菱形标记
        else:
            plt.scatter(final_state.position[0] + 0.5,
                        final_state.position[1] + 0.5,
                        s=self.marker_size,
                        marker="D",
                        color="r")  # 未着陆用红色菱形标记

    def draw_movement(self, from_position, to_position, color):
        y, x = from_position[1], from_position[0]
        dy, dx = to_position[1] - y, to_position[0] - x
        if dx == 0 and dy == 0:
            plt.scatter(x + 0.5, y + 0.5, marker="X", color=color)
        else:
            if abs(dx) >= 1 or abs(dy) >= 1:
                plt.quiver(x + 0.5,
                           y + 0.5,
                           dx,
                           -dy,
                           color=color,
                           scale=self.arrow_scale,
                           scale_units='inches')  # 绘制箭头，单位为inches
            else:
                plt.quiver(x + 0.5,
                           y + 0.5,
                           dx,
                           -dy,
                           color=color,
                           scale=self.arrow_scale,
                           scale_units='inches')


    def create_torch_image(self):
        """ 用于将图像生成为PyTorch张量 """

        # 创建一个内存缓存区，用于存储图像数据
        buf = io.BytesIO()

        # 生成图像，并将图像保存到缓存区中
        plt.savefig(buf, format='png', dpi=180, bbox_inches='tight')

        # 将缓存区指针移动到起始位置
        buf.seek(0)

        # 关闭所有打开的图像窗口
        plt.close('all')

        # 从缓冲区中读取字节流
        img_byte_stream = buf.getvalue()

        # 使用decode_png将字节流转换为张量
        combined_image = torchvision.io.decode_png(torch.ByteTensor(bytearray(img_byte_stream)), mode=torchvision.io.ImageReadMode.RGB)
        
        return combined_image

