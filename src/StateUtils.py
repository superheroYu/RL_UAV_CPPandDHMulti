import math
import numpy as np

def pad_centered(state, map_in, pad_value):
    # 使用 math.ceil() 函数来计算填充所需的行和列数，以使 no_fly_zone 属性居中
    padding_rows = math.ceil(state.no_fly_zone.shape[0] / 2.0)
    padding_cols = math.ceil(state.no_fly_zone.shape[1] / 2.0)


    # 获取状态对象的当前位置坐标
    position_x, position_y = state.position

    # 计算状态对象的行和列偏移量，以使其位于填充后地图的中心位置
    position_row_offset = padding_rows - position_y
    position_col_offset = padding_cols - position_x

    # 使用 NumPy 库的 np.pad() 函数来对输入地图进行填充
    # pad_width 参数指定每个维度所需的填充量，mode 参数指定填充类型，constant_values 参数指定填充值
    return np.pad(map_in,
                pad_width=[[padding_rows + position_row_offset - 1, padding_rows - position_row_offset],
                            [padding_cols + position_col_offset - 1, padding_cols - position_col_offset],
                            [0, 0]],
                mode='constant',
                constant_values=pad_value)