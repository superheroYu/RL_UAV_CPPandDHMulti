from .GridActions import GridActions
from .State import State

class GridPhysics(object):
    def __init__(self):
        """ 网格世界物理引擎 """
        self.landing_attempts = 0 # 用于记录着陆尝试次数
        self.boundary_counter = 0 # 用于记录飞到禁飞区的次数
        self.state: State = None
        
    def movement_step(self, action: GridActions):
        """ 输入动作action进行移动 """
        old_position = self.state.position # 获取当前状态的位置
        x, y = old_position 
        
        if action == GridActions.NORTH:
            y += 1
        elif action == GridActions.SOUTH:
            y -= 1
        elif action == GridActions.WEST:
            x -= 1
        elif action == GridActions.EAST:
            x += 1
        elif action == GridActions.LAND:
            self.landing_attempts += 1 # 如果选择的动作是着陆，则着陆尝试次数+1
            if self.state.is_in_landing_zone(): # 如果当前位置在着陆区
                self.state.set_landed(True) # 设置着陆成功
        # print(self.state.is_in_landing_zone())
        self.state.set_position([x, y]) # 设置发生动作后的新位置
        if self.state.is_in_no_fly_zone(): # 如果新位置在禁飞区
            self.boundary_counter += 1 # 将进入禁飞区的次数+1
            x, y = old_position 
            self.state.set_position([x, y]) # 将UAV坐标设置成老坐标
        
        self.state.decrement_movement_budget() # 运动预算减少
        self.state.set_terminal(self.state.landed or (self.state.movement_budget == 0)) # 如果着陆或者电池耗尽则设置为终止状态
        
        return x, y
    
    def reset(self, state):
        """ 重置物理引擎 """
        self.landing_attempts = 0
        self.boundary_counter = 0
        self.state = state #.copy()
        
        
            
        
    
    
            
            