## 目录

- [目录](#目录)
- [介绍](#介绍)
- [Requirements](#requirements)
- [如何使用](#如何使用)
- [资源](#资源)
- [Reference](#reference)
- [License](#license)
## 介绍

本仓库是基于论文 ["Multi-UAV Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning"](https://ieeexplore.ieee.org/document/9437338) 与论文 ["UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/9659413)以及其TensorFlow版本的代码["uav_data_harvesting"](https://github.com/hbayerlein/uav_data_harvesting)和["uavSim"](https://github.com/theilem/uavSim)进行改写而成的pytorch版本代码。


## Requirements

```
python==3.8 or newer
numpy==1.18.5 or newer
pytorch==1.31.1
matplotlib==3.3.0 or newer
scikit-image==0.18.3
tqdm==4.45.0 or newer
```
本仓库的代码运行在Windows系统采用单英伟达显卡进行训练，其他系统应该也可以运行此代码


## 如何使用

训练，如训练数据收集任务(DHMulti)多智能体

```
python train.py --target DHMulti --config config/manhattan32_DHMulti.json --id manhattan32_DHMulti 

--target                    训练的任务，可选DHMulti或CPP
--config                    训练相关配置文件的路径
--id                        训练采用的ID名，用于保存训练记录和模型
--generate_config           生成默认的配置文件
--resume                    模型的路径，用于恢复模型
--step                      已经训练的步数，配合--resume使用
--episode                   已经训练的回合数，配合--episode使用
```
一共提供两个场景：manhattan32和urban50，共4个配置文件"manhanttan32_cpp.json", "manhattan32_DHMulti.json", "urban50_cpp.json"和"urban50_DHMulti.json"

通过tensorboard观察训练中数据变化：

```
tensorboard --logdir logs
```

评估：

```
python eval.py --weights models/manhattan32_DHMulti_best --config config/manhattan32_DHMulti.json --id manhattan32_eval --samples 100

--weights                   训练好的模型路径
--config                    json格式的训练配置文件
--id                        输出文件的ID
--samples                   评估回合数
--seed                      设置随机种子
--show                      是否展示图像，无论是否展示会保存在eval文件夹中
--num_agents                智能体的数量如12 表示随机范围智能体数量范围在  [1,2]之间, 如果 11 则为单智能体
--target                    评估任务，选择DHMulti或CPP
```


## 资源
论文中的城市环境“manhattan32”和“urban50”已包含在“uavmap_figure”目录中。地图信息格式为PNG文件，一个像素代表一个网格世界单元。像素颜色根据以下规则确定单元格类型：

* 红色#ff0000不允许飞行区域(NFZ)
* 绿色#00ff00建筑物阻止无线连接(UAV可以飞越)
* 蓝色#0000ff起飞和降落区
* 黄色#ffff00建筑物阻止无线连接 + NFZ(UAV无法飞越)
* 如果您想创建一个新地图，您可以使用任何工具来设计一个与所需地图具有相同像素尺寸和上述颜色代码的PNG。

阴影地图为每个位置和每个物联网设备定义了是否存在直线视线(LoS)或非直线视线(NLoS)连接，当第一次使用新地图进行训练时会自动计算，并保存到“uavmap_figure”目录中作为NPY文件。


## Reference

本代码的文献引用与TensorFlow版本的一致：

[1] M. Theile, H. Bayerlein, R. Nai, D. Gesbert, M. Caccamo, “UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning" 20th International Conference on Advanced Robotics (ICAR), 2021. 

```
@inproceedings{theile2021uav,
  title={UAV path planning using global and local map information with deep reinforcement learning},
  author={Theile, Mirco and Bayerlein, Harald and Nai, Richard and Gesbert, David and Caccamo, Marco},
  booktitle={2021 20th International Conference on Advanced Robotics (ICAR)},
  pages={539--546},
  year={2021},
  organization={IEEE}
}
```

for the (multi-agent) Data Harvesting paper:

[2] H. Bayerlein, M. Theile, M. Caccamo, and D. Gesbert, “Multi-UAV path planning for wireless data harvesting with deep reinforcement learning," IEEE Open Journal of the Communications Society, vol. 2, pp. 1171-1187, 2021.

```
@article{bayerlein2021multi,
  title={Multi-uav path planning for wireless data harvesting with deep reinforcement learning},
  author={Bayerlein, Harald and Theile, Mirco and Caccamo, Marco and Gesbert, David},
  journal={IEEE Open Journal of the Communications Society},
  volume={2},
  pages={1171--1187},
  year={2021},
  publisher={IEEE}
}
```

## License 

This code is under a BSD license.