""" 此脚本用于对地图进行预处理 """

import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from .Agent import DDQNAgentParams

class Blind_Model(nn.Module):
    def __init__(self, in_features, hidden_layer_size, hidden_layer_num, out_features):
        super(Blind_Model, self).__init__()
        self.linear_list = nn.ModuleList()
        for k in range(hidden_layer_num):
            self.linear_list.append(nn.Linear(in_features=in_features if k ==0 else hidden_layer_size))
            self.linear_list.append(nn.ReLU())
        self.linear_out = nn.Linear(hidden_layer_size, out_features)
    
    def forward(self, x):
        for layer in self.linear_list:
            x = layer(x)
        return self.linear_out(x)
    
class Scalars_Model(nn.Module):
    def __init__(self, in_features, hidden_layer_size, hidden_layer_num, out_features):
        super(Scalars_Model, self).__init__()
        self.concat = nn.Identity()
        self.linear = Blind_Model(in_features, hidden_layer_size, hidden_layer_num, out_features)
    
    def forward(self, x):
        x = self.concat(torch.cat(x, dim=1))
        return self.linear(x)


class Total_Map_Model(nn.Module):
    def __init__(self):
        super(Total_Map_Model, self).__init__()

    def forward(self, boolean_map, float_map):
        # 将布尔类型的输入转换为 float 类型
        map_cast = boolean_map.to(torch.float32)

        # 沿着通道维度 (axis=3 对应于 PyTorch 的 axis=1) 拼接两个输入
        try:
            total_map = torch.cat([map_cast, float_map], dim=1)
            return total_map
        except:
            return map_cast


class Global_Map_Model(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_kernels,
                 conv_kernel_size,
                 conv_layers,
                 global_map_scaling,
                 total_map_model: Total_Map_Model = None):
        super(Global_Map_Model, self).__init__()
        self.global_map_scaling = global_map_scaling
        self.global_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels if k == 0 else conv_kernels,
                          conv_kernels,
                          conv_kernel_size,
                          stride=1,
                          padding='same'), nn.ReLU())
            for k in range(conv_layers)
        ])
        self.total_map_model = total_map_model if total_map_model != None else Total_Map_Model(
        )

    def forward(self, boolean_map=None, float_map=None, total_map=None):
        if total_map == None:
            conv_in = self.total_map_model(boolean_map, float_map)
        else:
            conv_in = total_map
        # 对全局地图进行平均池化
        global_map = nn.AvgPool2d((self.global_map_scaling,
                                   self.global_map_scaling))(conv_in).detach()

        # 对全局地图进行卷积操作
        global_map = self.global_convs(global_map)
        return global_map


class Local_Map_Model(nn.Module):
    def __init__(self,
                 in_channels,
                 conv_kernels,
                 conv_kernel_size,
                 conv_layers,
                 local_map_size,
                 total_map_model: Total_Map_Model = None):
        super(Local_Map_Model, self).__init__()
        self.local_map_size = local_map_size
        self.local_convs = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(in_channels if k == 0 else conv_kernels,
                          conv_kernels,
                          conv_kernel_size,
                          stride=1,
                          padding='same'), nn.ReLU())
            for k in range(conv_layers)
        ])
        self.total_map_model = total_map_model if total_map_model != None else Total_Map_Model(
        )

    def forward(self, boolean_map=None, float_map=None, total_map=None):
        if total_map == None:
            conv_in = self.total_map_model(boolean_map, float_map)
        else:
            conv_in = total_map
        # 对局部地图进行裁剪操作
        crop_frac = float(self.local_map_size) / float(
            conv_in.shape[2])  # (batch, C, H, W)
        local_map = transforms.CenterCrop(round(conv_in.shape[2] *
                                                crop_frac))(conv_in).detach()
        # 对局部地图进行卷积操作
        local_map = self.local_convs(local_map)
        return local_map


class CreateMapProc(nn.Module):
    def __init__(self,
                 params,
                 total_map_model=None,
                 global_map_model=None,
                 local_map_model=None):
        super(CreateMapProc, self).__init__()
        self.params = params
        self.total_map_model = Total_Map_Model(
        ) if total_map_model == None else total_map_model
        self.global_map_model = Global_Map_Model(
            self.params.in_channels, self.params.conv_kernels,
            self.params.conv_kernel_size, self.params.conv_layers,
            self.params.global_map_scaling, self.total_map_model
        ) if global_map_model == None else global_map_model
        self.local_map_model = Local_Map_Model(
            self.params.in_channels, self.params.conv_kernels,
            self.params.conv_kernel_size, self.params.conv_layers,
            self.params.local_map_size, self.total_map_model
        ) if local_map_model == None else local_map_model
        self.flatten = nn.Flatten()

    def forward(self, boolean_map, float_map, total_map=None):
        """ 对地图进行处理 """
        total_map = self.total_map_model(
            boolean_map,
            float_map) if total_map == None else total_map  # 获取总体地图
        
        global_map = self.global_map_model(total_map=total_map)  # 获取全局地图
        local_map = self.local_map_model(total_map=total_map)  # 获取局部地图
        flatten_global = self.flatten(global_map)  # 将全局地图为向量
        flatten_local = self.flatten(local_map)  # 将局部地图转换为向量
        return torch.cat([flatten_global, flatten_local],
                            dim=1)  # 拼接全局向量和局部向量


class Relative_Scalars_Model(nn.Module):
    def __init__(
        self,
        num_actions: int,
        params,
        device,
        agent=None
    ):
        super(Relative_Scalars_Model, self).__init__()
        self.num_actions = num_actions
        self.params = params
        self.device = device

        # 地图处理层，使用 CreateMapProc 类实现
        self.map_processing = CreateMapProc(self.params)

        # 用于拼接地图处理后的张量与状态张量的层
        self.concat = nn.Identity()

        # 定义隐藏层
        self.hidden_layers = None

        # 输出层，得到每个动作的值
        self.output_layer = nn.Linear(self.params.hidden_layer_size,
                                      self.num_actions)
        # 智能体
        self.agent = agent

    def forward(self,
                boolean_map=None,
                float_map=None,
                states_proc=None,
                total_map=None):
        # 使用 CreateMapProc 处理地图数据
        if total_map == None:
            flatten_map = self.map_processing(boolean_map=boolean_map,
                                              float_map=float_map)
        else:
            flatten_map = self.map_processing(total_map=total_map)

        # 拼接地图处理后的张量与状态张量
        layer = self.concat(torch.cat([flatten_map, states_proc], dim=1))

        if self.hidden_layers == None:
            in_features = layer.size()[1]
            self.hidden_layers = self.set_hidden_layers(in_features)
            if self.agent != None:
                self.agent.target_network.hidden_layers = self.set_hidden_layers(in_features)
                self.agent.resume_model() # 恢复模型
                self.agent.hard_update()
                self.agent.set_optim() # 设置优化器
                
        # 通过隐藏层进行处理
        layer = self.hidden_layers(layer)

        # 输出层计算每个动作的值
        output = self.output_layer(layer)

        return output
    
    def set_hidden_layers(self, in_features):
        hidden_layers = []
        for k in range(self.params.hidden_layer_num):
            hidden_layers.append(
                nn.Linear(
                    in_features
                    if k == 0 else self.params.hidden_layer_size,
                    self.params.hidden_layer_size))
            hidden_layers.append(nn.ReLU())
        hidden_layers = nn.Sequential(*hidden_layers).to(self.device)
        return hidden_layers



if __name__ == "__main__":

    class test_params:
        def __init__(self) -> None:
            self.hidden_layer_num = 2
            self.hidden_layer_size = 256
            self.use_global_local = True
            self.conv_kernels = 16
            self.conv_layers = 3
            self.global_map_scaling = 2
            self.in_channels = 3
            self.local_map_size = 17

    params = test_params()
    model = Relative_Scalars_Model(3, params=params)
    print(model)