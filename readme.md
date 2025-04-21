## 目录

- [目录](#目录)
- [介绍](#介绍)
- [Requirements](#requirements)
- [如何使用](#如何使用)
- [资源](#资源)
- [Reference](#reference)
- [License](#license)
## 介绍

本项目是 PyTorch 实现版本，其核心思想源自论文 ["Multi-UAV Path Planning for Wireless Data Harvesting with Deep Reinforcement Learning"](https://ieeexplore.ieee.org/document/9437338) 和 ["UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/9659413)。代码实现参考并改编自原作者提供的 TensorFlow 版本：["uav_data_harvesting"](https://github.com/hbayerlein/uav_data_harvesting) 和 ["uavSim"](https://github.com/theilem/uavSim)。

项目中的多智能体部分采用了基于 Agent Environment Cycle (AEC) 的范式，这与 PettingZoo 的 AEC 环境类似，但本项目并未直接使用 PettingZoo 库。请注意将其与基于部分可观察随机博弈 (POSG) 的多智能体强化学习算法区分开来。当前实现未采用多进程或多线程进行数据收集，如有需要可自行修改。欢迎联系作者共同完善本项目。

## 更新

- 2025-04-21 更新了依赖包，可以使用较新版的各个依赖包，如torch，numpy，matplotlib，scikit-image，tqdm，tensorboard，opencv-python等，修复了一些文件夹未创建的问题，增加了.gitignore文件。

## Requirements

详细的依赖包及其版本请参见 `requirements.txt` 文件。

本仓库的代码运行在Windows系统采用单英伟达显卡进行训练，其他系统应该也可以运行此代码


## 如何使用

创建虚拟环境  ：

```bash
conda create -n uav python=3.9.20
conda activate uav
```

安装依赖包：
如果cuda版本为12.6，则使用以下命令安装依赖包：
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```
其他cuda版本请自行安装torch2.6.0，torchvision0.21.0，请参考https://pytorch.org/
再删除requirements.txt中torch和torchvision的版本，然后使用以下命令安装：
```bash
pip install -r requirements.txt
```

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
python eval.py --target DHMulti --weights models/manhattan32_DHMulti_best --config config/manhattan32_DHMulti.json --id manhattan32_eval --samples 100

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
论文中的城市环境"manhattan32"和"urban50"已包含在"uavmap_figure"目录中。地图信息格式为PNG文件，一个像素代表一个网格世界单元。像素颜色根据以下规则确定单元格类型：

* 红色#ff0000不允许飞行区域(NFZ)
* 绿色#00ff00建筑物阻止无线连接(UAV可以飞越)
* 蓝色#0000ff起飞和降落区
* 黄色#ffff00建筑物阻止无线连接 + NFZ(UAV无法飞越)
* 如果您想创建一个新地图，您可以使用任何工具来设计一个与所需地图具有相同像素尺寸和上述颜色代码的PNG。

阴影地图为每个位置和每个物联网设备定义了是否存在直线视线(LoS)或非直线视线(NLoS)连接，当第一次使用新地图进行训练时会自动计算，并保存到"uavmap_figure"目录中作为NPY文件。


## Reference

本代码的文献引用与TensorFlow版本的一致：

[1] M. Theile, H. Bayerlein, R. Nai, D. Gesbert, M. Caccamo, "UAV Path Planning using Global and Local Map Information with Deep Reinforcement Learning" 20th International Conference on Advanced Robotics (ICAR), 2021. 

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