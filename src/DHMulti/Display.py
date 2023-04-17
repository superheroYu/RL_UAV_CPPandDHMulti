import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ..Base.Display import Display
from ..UAVMap.Map import UAVMap

class DHMultiDisplay(Display):
    
    def __init__(self):
        super().__init__()
        self.channel = None

    def set_channel(self, channel):
        self.channel = channel

    def display_episode(self, env_map: UAVMap, trajectory, plot=False, save_path=None):

        first_state = trajectory[0][0]
        final_state = trajectory[-1][3]

        fig_size = 5.5
        fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
        ax_traj = ax[0]
        ax_bar = ax[1]

        value_step = 0.4 / first_state.device_list.num_devices
        # Start with value of 200
        value_map = np.ones(env_map.size(), dtype=float)
        for device in first_state.device_list.get_devices():
            value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]

        self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)

        for device in first_state.device_list.get_devices():
            ax_traj.add_patch(
                patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
                               edgecolor="black"))

        self.draw_start_end(trajectory)

        for exp in trajectory:
            idx = exp[3].device_coms[exp[0].active_agent]
            if idx == -1:
                color = "black"
            else:
                color = exp[0].device_list.devices[idx].color

            self.draw_movement(exp[0].position, exp[3].position, color=color)

        # Add bar plots
        device_list = final_state.device_list
        devices = device_list.get_devices()
        colors = [device.color for device in devices]
        names = ["total"] + colors
        colors = ["black"] + colors
        datas = [device_list.get_total_data()] + [device.data for device in devices]
        collected_datas = [device_list.get_collected_data()] + [device.collected_data for device in devices]
        y_pos = np.arange(len(colors))

        plt.sca(ax_bar)
        ax_bar.barh(y_pos, datas)
        ax_bar.barh(y_pos, collected_datas)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(names)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("Data")
        ax_bar.set_aspect(- np.diff(ax_bar.get_xlim())[0] / np.diff(ax_bar.get_ylim())[0])

        # save image and return
        if save_path is not None:
            # save just the trajectory subplot 0
            extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            extent.x0 -= 0.3
            extent.y0 -= 0.1
            fig.savefig(save_path, bbox_inches=extent,
                        format='png', dpi=300, pad_inches=1)
        if plot:
            plt.show()

        return self.create_torch_image()
# class DHMultiDisplay(Display):
#     def __init__(self):
#         super().__init__()  # 调用父类 DHDisplay 的初始化方法
#         self.channel = None

#     def draw_start_end(self, trajectory):
#         for exp in trajectory:  # 遍历轨迹中的每个元组
#             state, action, reward, next_state = exp  # 将轨迹元组拆分为 state, action, reward, 和 next_state

#             # 如果 state 中的 movement_budget 等于初始 movement_budget，表示无人机刚开始移动
#             if state.movement_budget == state.initial_movement_budget:
#                 # 在无人机起始位置绘制一个白色的菱形标记
#                 plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5, s=self.marker_size, marker="D",
#                             color="w")

#             # 如果 next_state 是终止状态
#             if next_state.terminal:
#                 # 如果无人机成功降落
#                 if next_state.landed:
#                     # 在降落位置绘制一个绿色的菱形标记
#                     plt.scatter(next_state.position[0] + 0.5, next_state.position[1] + 0.5,
#                                 s=self.marker_size, marker="D", color="green")
#                 else:
#                     # 否则在降落位置绘制一个红色的菱形标记
#                     plt.scatter(next_state.position[0] + 0.5, next_state.position[1] + 0.5,
#                                 s=self.marker_size, marker="D", color="r")

#     # 设置通道的方法
#     def set_channel(self, channel):
#         self.channel = channel

#     # 绘制条形图的方法，显示设备的数据收集情况
#     def draw_bar_plots(self, final_state, ax_bar:plt.Axes):
#         # 获取设备列表和设备信息
#         device_list = final_state.device_list
#         devices = device_list.get_devices()
#         colors = [device.color for device in devices]
#         names = ["total"] + colors
#         colors = ["black"] + colors
#         datas = [device_list.get_total_data()] + [device.data for device in devices]
#         collected_datas = [device_list.get_collected_data()] + [device.collected_data for device in devices]
#         y_pos = np.arange(len(colors))

#         # 绘制水平条形图
#         plt.sca(ax_bar)
#         ax_bar.barh(y_pos, datas)
#         ax_bar.barh(y_pos, collected_datas)
#         ax_bar.set_yticks(y_pos)
#         ax_bar.set_yticklabels(names)
#         ax_bar.invert_yaxis()
#         ax_bar.set_xlabel("Data")
#         ax_bar.set_aspect(- np.diff(ax_bar.get_xlim())[0] / np.diff(ax_bar.get_ylim())[0])

#     # 显示单个episode的方法，包括设备位置、无人机轨迹和数据收集情况
#     def display_episode(self, env_map: UAVMap, trajectory, plot=False, save_path=None):
#         # 创建画布和子图
#         fig_size = 5.5
#         fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
#         ax_traj = ax[0]
#         ax_bar = ax[1]

#         # 计算设备影响值图
#         first_state = trajectory[0][0]
#         value_step = 0.4 / first_state.device_list.num_devices
#         value_map = np.ones(env_map.size(), dtype=float)
#         for device in first_state.device_list.get_devices():
#             value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]

#         # 绘制网格图像和设备位置
#         self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)
#         for device in first_state.device_list.get_devices():
#             ax_traj.add_patch(
#                 patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
#                             edgecolor="black"))

#         # 绘制无人机起点和终点
#         self.draw_start_end(trajectory)

#         # 绘制无人机移动轨迹
#         for exp in trajectory:
#             idx = exp[3].device_coms[exp[0].active_agent]
#             if idx == -1:
#                 color = "black"
#             else:
#                 color = exp[0].device_list.devices[idx].color
#             self.draw_movement(exp[0].position, exp[3].position, color=color)

#         # 绘制数据收集的条形图
#         final_state = trajectory[-1][3]
#         self.draw_bar_plots(final_state, ax_bar)

#         # 保存图片和显示
#         if save_path is not None:
#             # 仅保存轨迹子图
#             extent = ax[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#             extent.x0 -= 0.3
#             extent.y0 -= 0.1
#             fig.savefig(save_path, bbox_inches=extent,
#                         format='png', dpi=300, pad_inches=1)
#         if plot:
#             plt.show()

#         return self.create_torch_image()

#     # 显示特定状态的方法，包括设备位置、无人机位置和数据收集情况
#     def display_state(self, env_map, initial_state, state, plot=False):
#         # 创建画布和子图
#         fig_size = 5.5
#         fig, ax = plt.subplots(1, 2, figsize=[2 * fig_size, fig_size])
#         ax_traj = ax[0]
#         ax_bar = ax[1]

#         # 计算设备影响值图
#         value_step = 0.4 / initial_state.device_list.num_devices
#         value_map = np.ones(env_map.get_size(), dtype=float)
#         for device in initial_state.device_list.get_devices():
#             value_map -= value_step * self.channel.total_shadow_map[device.position[1], device.position[0]]

#         # 绘制网格图像和设备位置
#         self.create_grid_image(ax=ax_traj, env_map=env_map, value_map=value_map)
#         for device in initial_state.device_list.get_devices():
#             ax_traj.add_patch(
#                 patches.Circle(np.array(device.position) + np.array((0.5, 0.5)), 0.4, facecolor=device.color,
#                             edgecolor="black"))

#         # 绘制无人机位置
#         color = "green" if state.landed else "r"
#         plt.scatter(state.position[0] + 0.5, state.position[1] + 0.5,
#                     s=self.marker_size, marker="D", color=color, zorder=10)

#         # 绘制数据收集的条形图
#         self.draw_bar_plots(state, ax_bar)

#         if plot:
#             plt.show()

#         return self.create_torch_image()



            
        
