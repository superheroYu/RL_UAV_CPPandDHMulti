import collections
import datetime
import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import distutils.util
from .Base.Display import Display


class ModelStatsParams(object):
    """ 用于定义模型参数的类 """
    def __init__(self,
                 save_model_path='models/save_model_path',
                 moving_average_length=50):
        self.save_model_path = save_model_path # 模型保存的位置
        self.moving_average_length = moving_average_length # 模型的平均移动长度
        self.log_file_name = datetime.datetime.now().strftime(
            "%Y%m%d-%H%M%S")  # 转化时间戳为年月日时分秒
        self.training_images = False


class ModelStats(object):
    """ 用于模型统计的类 """
    def __init__(self,
                 params: ModelStatsParams,
                 display: Display,
                 force_override: bool = False):
        self.params = params
        self.display = display
        self.model:torch.nn.Module = None # pytorch模型
        self.evaluation_value_callback = None
        self.env_map_callback = None
        self.log_value_callbacks = [] # 回调函数列表
        self.trajectory = [] # 轨迹
        self.model = None # 模型
        self.log_dir = "logs/" + params.log_file_name
        if os.path.isdir(self.log_dir): # 判断是否是路径
            if force_override: # 是否覆盖
                shutil.rmtree(self.log_dir) # 覆盖就强制删除
            else:
                print(self.log_dir, '已经存在.') # 
                resp = input('是否覆盖日志文件? [Y/n]\n')
                if resp == '' or distutils.util.strtobool(resp):
                    print('删除旧日志')
                    shutil.rmtree(self.log_dir)
                else:
                    raise AttributeError('Okay bye')
        
        self.writer = SummaryWriter(log_dir=self.log_dir) # 创建一个tensorboard写入器 self.log_dir
        print(self.log_dir)
        
        self.evaluation_deque = collections.deque(maxlen=params.moving_average_length)
        self.eval_best = -float('inf')
        self.bar = None
        
    def set_evaluation_value_callback(self, callback: callable):
        """ 设置用于评估值的回调函数 """
        self.evaluation_value_callback = callback

    def set_env_map_callback(self, callback: callable):
        """ 设置环境地图的回调函数 """
        self.env_map_callback = callback

    def add_log_data_callback(self, name: str, callback: callable):
        """ 添加回调函数 """
        self.log_value_callbacks.append((name, callback))
        
    def add_experience(self, experience):
        """ 添加经历 """
        self.trajectory.append(experience)

    def log_training_data(self, step):
        """ 训练数据日志 """
        self.log_data(step, "train/", self.params.training_images)

    def log_testing_data(self, step):
        """ 测试数据日志 """
        self.log_data(step, "test/")
        if self.evaluation_value_callback:
            self.evaluation_deque.append(self.evaluation_value_callback())

    def log_data(self, step, train_test:str, images=True):
        """ 日志数据 """
        for callback in self.log_value_callbacks:
            self.writer.add_scalar(train_test+callback[0], callback[1](), global_step=step)
        if images: # 判断是否画轨迹图
            trajectory_img = self.display.display_episode(self.env_map_callback(), trajectory=self.trajectory)
            self.writer.add_image(train_test + 'trajectory', trajectory_img, global_step=step)
            
    def set_model(self, model:torch.nn.Module):
        """ 设置pytorch模型 """
        self.model = model

    def save_if_best(self):
        """ 保存最好的模型 """
        if len(self.evaluation_deque) < self.params.moving_average_length:
            return

        eval_mean = np.mean(self.evaluation_deque)
        if eval_mean > self.eval_best:
            self.eval_best = eval_mean
            if self.params.save_model_path != '':
                print(f"保存最好的模型: {eval_mean=}")
                torch.save(self.model.state_dict(), self.params.save_model_path + '_best.pth')

    def training_ended(self):
        """ 保存最终模型 """
        if self.params.save_model_path != '':
            torch.save(self.model.state_dict(), self.params.save_model_path + '_unfinished.pth')
            print('最终模型保存在：', self.params.save_model_path + '_unfinished.pth')
        self.writer.close()

    def save_episode(self, save_path):
        """ 保存episode """
        f = open(save_path + ".txt", "w")
        for callback in self.log_value_callbacks:
            f.write(callback[0] + ' ' + str(callback[1]()) + '\n')
        f.close()

    def on_episode_begin(self):
        self.trajectory = []

    def on_episode_end(self, episode_count):
        self.writer.add_scalar("episode", episode_count, episode_count)
        
    def close(self):
        self.writer.close()
