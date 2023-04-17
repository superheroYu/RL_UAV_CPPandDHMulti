from enum import Enum

class GridActions(Enum):
    """ 有悬浮的网格动作 """
    NORTH = 0 # 向北
    EAST = 1 # 向东
    SOUTH = 2 # 向南
    WEST = 3 # 向西
    LAND = 4 # 着陆
    HOVER = 5 # 悬浮
        