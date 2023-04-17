import numpy as np
import matplotlib.pyplot as plt
from ..UAVMap.Map import UAVMap
from ..Base.Display import Display


class CPPDisplay(Display):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法

    def display_episode(self, env_map: UAVMap, trajectory, plot=False, save_path=None):
        # 展示整个episode的轨迹，可以选择是否绘制图像和保存图像

        first_state = trajectory[0][0]  # 获得轨迹的初始状态
        final_state = trajectory[-1][3]  # 获得轨迹的最终状态

        fig_size = 5.5  # 设置图像大小
        fig, ax = plt.subplots(1, 1, figsize=[fig_size, fig_size])  # 创建图像和子图
        value_map = final_state.coverage * 1.0 + (~final_state.coverage) * 0.75  # 根据覆盖情况计算值映射

        self.create_grid_image(ax=ax, env_map=env_map, value_map=value_map, green=first_state.target)  # 创建网格图像

        self.draw_start_end(trajectory)  # 绘制起点和终点

        # 绘制轨迹箭头
        for exp in trajectory:
            self.draw_movement(exp[0].position, exp[3].position, color="black")

        # 保存图像并返回
        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight',
                        format='png', dpi=300)
        if plot:
            plt.show()

        return self.create_torch_image()  # 创建并返回torch图像

    def display_state(self, env_map, initial_state, state, plot=False):
        # 展示单个状态，可以选择是否绘制图像

        fig_size = 5.5  # 设置图像大小
        fig, ax = plt.subplots(1, 1, figsize=[fig_size, fig_size])  # 创建图像和子图
        value_map = state.coverage * 1.0 + (~state.coverage) * 0.75  # 根据覆盖情况计算值映射

        self.create_grid_image(ax=ax, env_map=env_map, value_map=value_map, green=initial_state.target)  # 创建网格图像

        color = "green" if state.landed else "r"  # 根据是否着陆选择颜色
        plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5,
                    s=self.marker_size, marker="D", color=color)  # 绘制当前状态的点

        if plot:
            plt.show()

        return self.create_torch_image()  # 创建并返回pytorch图像

    def draw_map(self, map_in):
        # 绘制地图

        rgb = map_in[0, :, :, :3]  # 从输入的地图中提取RGB值
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)  # 重新排列RGB通道的顺序
        plt.imshow(rgb.astype(float))  # 将RGB值转换为浮点类型并显示图像
        plt.show()  # 显示图像

    def draw_maps(self, total_map, global_map, local_map):
        fig, ax = plt.subplots(1, 3)  # 创建1行3列的子图

        # 绘制总地图
        rgb = total_map[0, :, :, :3]  # 提取总地图的RGB值
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)  # 重新排列RGB通道的顺序
        ax[0].imshow(rgb.astype(float))  # 将RGB值转换为浮点类型并显示图像

        # 绘制全局地图
        rgb = global_map[0, :, :, :3]  # 提取全局地图的RGB值
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)  # 重新排列RGB通道的顺序
        ax[1].imshow(rgb.astype(float))  # 将RGB值转换为浮点类型并显示图像

        # 绘制局部地图
        rgb = local_map[0, :, :, :3]  # 提取局部地图的RGB值
        rgb = np.stack([rgb[:, :, 0], rgb[:, :, 2], rgb[:, :, 1]], axis=2)  # 重新排列RGB通道的顺序
        ax[2].imshow(rgb.astype(float))  # 将RGB值转换为浮点类型并显示图像

        plt.show()  # 显示包含所有子图的图像