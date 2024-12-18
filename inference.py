# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/7/3 15:00

"""
文件说明：  模型推理； 集成 1、阻塞事件识别 2、阻塞事件规则过滤、padding 3、模型推理，给出对应阻塞事件label
"""
import os
from collections import deque
import argparse
from glob import glob

import numpy as np
import pandas as pd
import pyabf
import torch
import matplotlib
# matplotlib.use('TkAgg')
import yaml
from matplotlib import pyplot as plt
from pymongo import MongoClient
from astropy.modeling import models, fitting
from model.S2SModel import Classifier
from model.models import ConvNetFour
from util.detection import get_signal_mean_sigmal, adept2State
from util.naporDataLoder import MetricMonitor, normal_naporDataset

class EventExtraction(object):
    """
    完成事件提取
    """
    def __init__(self, signal_data, window_size=50000, scale=12):
        self.signal_data = signal_data
        self.window_size = window_size
        self.scale = scale
        self.signal_queue = deque(maxlen=window_size)
        self.baseline = None
        self.std_dev = None
        self.threshold = None
        self.blocked_segments = []
        self.threshold_segment = []
        self.baseline_segment = []
        self.block_flag = []
        self.BlockDepth = []
        self.resDwellTime = []
        self.filter = []
        self.detection_signal(signal_data)
        self.idx = 0

    def __len__(self):
        return len(self.threshold_segment)

    def getitem(self):
        """
        :param idx:
        :return: 1、_baseline: 当前信号阻塞基线
                2、_threshold: 阈值
                3、_blocked_segments: 阻塞信号开始和结束下标[start, end]
                4、_block_flag: 等效电路法标志，1为正常
                5、_block_depth: 等效电路法计算的阻塞深度
                6、_resDwellTime: 阻塞持续时间
                7、_filter: 1表示满足limit要求
        """
        idx = self.idx
        _baseline = self.baseline_segment[idx]
        _threshold = self.threshold_segment[idx]
        _blocked_segments = self.blocked_segments[idx]
        _block_flag = self.block_flag[idx]
        _block_depth = self.BlockDepth[idx]
        _resDwellTime = self.resDwellTime[idx]
        _filter = self.filter[idx]
        _before_tag = self.blocked_segments[idx][1]+50
        if idx < self.__len__() - 2:
            _next_tag = self.blocked_segments[idx+2][0] - 50
        else:
            _next_tag = self.blocked_segments[idx][1] + 500000
        # if idx == 0:
        #     _before_tag, _next_tag = self.blocked_segments[0][1], self.blocked_segments[1][0]
        self.idx = idx + 1
        return _baseline, _threshold, _blocked_segments, _block_flag, _block_depth, _resDwellTime, _filter, _before_tag, _next_tag

    def update_baseline(self, new_signal):
        if len(self.signal_queue) == self.window_size:
            # 发生阻塞
            if self.threshold > new_signal or (self.baseline + self.scale * self.std_dev) < new_signal:
                return True
            else:
                # 添加基线判断
                if (new_signal > (self.baseline - 2 * self.std_dev)) and (
                        new_signal < (self.baseline + 2 * self.std_dev)):
                    removed_signal = self.signal_queue.popleft()
                    self.signal_queue.append(new_signal)
                    self.baseline = (self.baseline * self.window_size - removed_signal + new_signal) / self.window_size
                    self.threshold = self.baseline - self.scale * self.std_dev
                return False
        if len(self.signal_queue) < self.window_size:
            self.signal_queue.append(new_signal)
            self.baseline = np.mean(self.signal_queue)
            self.threshold = self.baseline - self.scale * self.std_dev
            return False

    def is_blocked(self, signal):
        if self.baseline is None:
            return False
        if (signal > (self.baseline - 2 * self.std_dev)) and (signal < (self.baseline + 2 * self.std_dev)):
            return True
        else:
            return False

    def process_signal_data(self, signal_data):
        """
        在信号数据中检测连续的阻塞事件，然后将这些事件的起始和结束索引记录在 blocked_segments 列表中
        :param signal_data: 一段信号，阻塞信号
        :return:
        """
        # ll = signal_data.shape[0]
        for idx, signal in enumerate(signal_data):
            # if idx == 1428545:
            #     xxx=1
            # 如果 update_baseline 方法返回 True，这意味着发生阻塞，队列已满且移动基线和标准差已更新，这样变动的阈值也随之更新。
            temp = self.update_baseline(signal)
            if temp:
                # 这一行调用 update_baseline 方法来更新移动基线和标准差。如果队列尚未达到 window_size，则该方法将仅将当前信号添加到队列中
                # 如果 blocked_segments 列表为空，或者上一个阻塞段的结束索引不是当前信号的前一个索引，那么会创建一个新的阻塞段。
                if not self.blocked_segments or self.blocked_segments[-1][1] != idx - 1:
                    # 如果需要创建新的阻塞段，那么就在 blocked_segments 列表中添加一个包含当前信号索引的新阻塞段
                    self.blocked_segments.append([idx, idx])
                    self.baseline_segment.append(self.baseline)
                    self.threshold_segment.append(self.threshold)
                    if self.blocked_segments.__len__() >= 2:
                        # 使用等效电路法
                        row = self.blocked_segments[-2]
                        uData = signal_data[row[0] - 50: row[1] + 50]
                        globtime = (row[0] - 50) / 1e2
                        fit = adept2State(globtime, uData, self.baseline, self.std_dev)
                        fit.FitEvent()
                        self.block_flag.append(fit.flag)
                        self.BlockDepth.append(fit.mdBlockDepth)
                        self.resDwellTime.append(fit.mdResTime)
                        self.filter.append(1)
                else:
                    # 在这种情况下，将更新上一个阻塞段的结束索引为当前信号的索引
                    self.blocked_segments[-1][1] = idx
        # 将信号最后的信号清除
        self.blocked_segments.pop()
        self.threshold_segment.pop()
        self.baseline_segment.pop()

    def detection_signal(self, signal_data):
        """
        阻塞事件识别
        :param data: 使用pyabf读取后格式,然后使用numpy转化格式
        :return:
        """
        # 根据前三秒获取方差, 参数0.5是区间
        mu, sig = get_signal_mean_sigmal(signal_data[0:50000], 0.5)
        self.std_dev = sig
        # 根据前面获得的高斯方差进行构建阻塞事件识别对象 # 处理数据
        self.process_signal_data(signal_data)


class FilterSignalBin(EventExtraction):
    """ 按规则过滤事件，并实现负值为0，（0-1）标准化，不足3W补齐 """

    def __init__(self, signal_data, label, window_size=50000, bin=512):
        super().__init__(signal_data, window_size)
        # _baseline, _threshold, _blocked_segments, _block_flag, _block_depth, _resDwellTime, _filter = EventExtraction.getitem(self)
        self.signal_data = signal_data
        self.current_block = None
        self.current_baseline = None
        self.current_threshold = None
        self.current_resDwellTime = None
        self.block_index = None
        self.label = label
        self.bin = bin
        self.before = None
        self.after = None

        self.current_depth = None
    def standardize_data(self,data):
        # 计算均值和标准差
        mean = data.mean()
        std = data.std()
        # 标准化数据

        # 使用 np.divide 处理除数为零的情况
        # where 参数可以指定分母为零时的结果,这个解决loss为nan的bug，避免了除零错误
        standardized_data = np.divide((data - mean), std, out=np.zeros_like(data), where=std != 0)
        # standardized_data = (data - mean) / std

        return standardized_data


    # 打印标准化后的数据
    def my_binning(self, sequence, baseline, bin):
        """
        对序列进行分箱统计。

        根据给定的基线值和分箱数量，将序列中的每个元素归入相应的箱中，并统计每个箱的元素数量。

        参数:
        - sequence: 待分箱的序列，应为数值类型的一维数据。
        - baseline: 基线值，用于确定分箱的大小。
        - bin: 分箱的数量，决定了分箱的精细程度。

        返回:
        - P: 分箱后的结果，每个箱中元素的数量。
        """
        # 初始化一个长度为bin的零数组，用于存放分箱后的元素数量
        P = np.zeros(bin)
        # 计算每个箱的大小，即基线值除以箱的数量
        p_step = baseline / bin
        # 去除掉NaN和Inf，用0替代
        sequence = np.nan_to_num(sequence, nan=0, posinf=0, neginf=0)
        for value in sequence:
            # 计算当前元素所属的箱的索引
            s = int(value / p_step)
            # 如果计算出的索引超出了箱的数量范围，将其设置为最后一个箱的索引
            if s >= bin:
                s = bin - 1
            # 将当前元素所属箱的数量加一
            P[s] += 1
        # P = P / len(sequence)
        # P = self.normalize_data(P)
        # P = np.nan_to_num(P, nan=0, posinf=0, neginf=0)
        # 返回分箱后的结果
        # P = P / max(P)
        return self.standardize_data(P)
    def fit_gaosi(self, data, bins):
        hx, xedge = np.histogram(data, bins)
        xedge = (xedge[1:] + xedge[:-1]) / 2
        g_init = models.Gaussian1D(amplitude=np.max(hx), mean=np.mean(data), stddev=np.std(data))
        fit_g = fitting.LevMarLSQFitter()
        g = fit_g(g_init, xedge, hx)
        return g.mean.value, g.stddev.value, g
    def next(self):
        _baseline, _threshold, _blocked_segments, _block_flag, _block_depth, _resDwellTime, _filter, before_tag, next_tag = EventExtraction.getitem(
            self)
        # 等效电路法认为正常
        if _block_flag == 1 and _filter == 1:
            if self.idx != 1:  # 排除第一个阻塞事件
                interval_length = _blocked_segments[0] - self.before
                interval_right_length = self.after - _blocked_segments[1]
                if interval_right_length < 0:
                    interval_right_length = 4  # 如果存在50点内就出现下一个信号
                if interval_length >= 9 and interval_right_length >= 9:
                    temp = self.signal_data[_blocked_segments[0] - 9: _blocked_segments[1] + 9]
                elif interval_length >= 9 and interval_right_length < 9:
                    temp = self.signal_data[
                           _blocked_segments[0] - 9: _blocked_segments[1] + interval_right_length - 2]
                elif interval_length < 9 and interval_right_length >= 9:
                    temp = self.signal_data[_blocked_segments[0] - interval_length + 2: _blocked_segments[1] + 9]
                else:
                    temp = self.signal_data[
                           _blocked_segments[0] - interval_length + 2: _blocked_segments[1] + interval_length - 2]
            else:
                temp = self.signal_data[_blocked_segments[0] - 9: _blocked_segments[1] + 9]  # 以前是这样减去9

            self.current_block = temp
            self.current_baseline = _baseline
            self.block_index = _blocked_segments
            self.current_resDwellTime = _resDwellTime
            self.current_threshold = _threshold
            self.current_depth = _block_depth

        self.before = before_tag
        self.after = next_tag

    def get_next_normal_sig(self):
        #  获取单个阻塞信号数据
        self.next()
        flag = False
        currentSignal = None
        mean, std = 0, 0
        if self.current_baseline:
            try:
                sig = self.current_block
                # 清除负数
                temp = np.where(sig > 0, sig, 0)
                temp = np.where(np.isnan(temp), 0, temp)
                sigma = (self.current_baseline - self.current_threshold) / 12
                mean, std, gao = self.fit_gaosi(temp, 100)
                # 标准化
                currentSignal = self.my_binning(temp, self.current_baseline, self.bin)
                currentSignal = torch.tensor([currentSignal], dtype=torch.float32)
                flag = True
            except:
                flag = False
        return currentSignal, flag, mean, std

class PatternClassifier():
    """ 推理，使用1D 卷积神经网络进行信号分类"""
    def __init__(self, args):
        self.args = args
        self.model = Classifier(class_num=args.classfication, bn=True, shortcut=False, activation=True)
        checkpoint = torch.load(self.args.model_path, map_location=torch.device(self.args.device))
        self.model.load_state_dict(checkpoint['model'])
        self.model = self.model.to(self.args.device)  # 模型部署到设备上
        self.model.eval()  # 模型设置为验证格式

    def validate(self, signal_data):
        autocast = torch.cuda.amp.autocast
        with torch.no_grad():  # 开始推理
            with autocast():
                signal_data = signal_data.unsqueeze(1)
                signal_data = signal_data.to(self.args.device, non_blocking=True)  # 读取图片
                output = self.model(signal_data)  # 前向传播
            y_pred = torch.softmax(output, dim=1)
            value, index = torch.max(y_pred, dim=1)
            softmax_value = value.cpu().tolist()
        return label_info[index[0].item()], softmax_value

def main_workflow(args, connection):
    resDwellTimes, blockDepths = [], []
    # 使用pyabf打开abf文件
    abf = pyabf.ABF(args.signal_path)
    # 获取数据
    data = abf.data
    signal_data = np.array(data[0])
    # _limits = None
    # for key, value in args.limit.items():
    #     if key in args.signal_path:
    #         _limits =value

    # 获取阻塞信号
    fs = FilterSignalBin(signal_data, label=0, window_size=1000, bin=512)

    # 深度学习模型加载
    PC = PatternClassifier(args)

    # 逐个处理阻塞事件
    for i in range(fs.__len__()):
        currentSignal, flg, mean, std = fs.get_next_normal_sig()
        if flg and fs.current_resDwellTime >= 0.8 and fs.current_resDwellTime <= 300:
            pred, softmax_value = PC.validate(currentSignal)
            baseline = fs.current_baseline
            resDwellTime = fs.current_resDwellTime
            threshold = fs.current_threshold
            block_index = fs.block_index
            doc = {'name': args.signal_path,
                   'baseline': baseline,
                   'resDwellTime': resDwellTime,
                   'threshold': threshold,
                   'depth': fs.current_depth,
                   'block_index': block_index,
                   'prediction': pred,
                   'Gaussian_mean': mean,
                   'Gaussian_std': std,
                   'softmax_value': softmax_value[0]
                   }
            connection.insert_one(doc)
        else:
            print("此事件无效")


def cat_ratio(Napor_conn, signal_path, level_name='prediction'):

    df = pd.DataFrame(list(Napor_conn.find({})))
    if df.empty:
        print("数据库为空")
        return None, None, None
    # 模型泛化性差
    df = df[df['name'] == signal_path]
    length = df.shape[0]
    grouped = df.groupby(level_name)[level_name].size()
    # 对元素个数进行排序，并将结果转换为DataFrame
    sorted_grouped = grouped.sort_values(ascending=False).reset_index(name='count')

    first = sorted_grouped[level_name][0]
    r = df['resDwellTime'].values
    d = df['depth'].values
    for index, row in sorted_grouped.iterrows():
        print('{}: {:.2f}%'.format(row[level_name], row['count'] / length * 100))
    sorted_grouped['ratio'] = sorted_grouped['count'] / length
    results[signal_path] = sorted_grouped
    return first, r, d

def cat_main_(Napor_conn, signal_path):
    df = pd.DataFrame(list(Napor_conn.find({})))
    if df.empty:
        return None
    # 模型泛化性差
    df = df[df['name'] == signal_path]
    return df

if __name__ == '__main__':
    results = {}
    # 样本标签对应名称
    with open(os.path.join('./config/filter.yaml'), 'r') as file:
        # 版本问题需要将Loader=yaml.CLoader加上
        config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)
    label_info = {}
    for da in config_yaml:
        if 'label' in config_yaml[da]:
            label_info[config_yaml[da]['label']] = da

    # 超参数设置
    with open(os.path.join('./config/inference.yaml'), 'r') as file:
        # 版本问题需要将Loader=yaml.CLoader加上
        config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)
    # 超参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_path', default=config_yaml['inference_settings']['test_data_path'], type=str)
    parser.add_argument('--device', default=config_yaml['inference_settings']['gpu'], type=str, help='GPU id to use.')
    parser.add_argument('--save_result_dir', default=config_yaml['inference_settings']['save_result_path'], type=str, help='推理结果保存')
    parser.add_argument('--description', default='测试脚本', help='描述')
    parser.add_argument('--save_result_detail_path', default=config_yaml['inference_settings']['save_result_detail_path'], type=str, help='推理详细结果保存')
    parser.add_argument('--model_path', default=config_yaml['inference_settings']['model_path'],
                        type=str, help='加载模型path')
    parser.add_argument('--classfication', default=config_yaml['inference_settings']['classfication'], type=int)  # 分类数
    args = parser.parse_args()
    mongoClient = MongoClient('mongodb://172.10.10.8:27017', username='root', password='123456')
    ourDB = mongoClient.get_database('Napor')
    Napor_conn = ourDB.get_collection('Ginsenoside_S2S_PQ2_2')
    Napor_conn.delete_many({})
    files = []
    for filename in os.listdir(args.signal_path):
        file_paths = os.path.join(args.signal_path, filename)
        for filename in glob(file_paths + '/*.abf'):
            files.append(filename)
    writer = pd.ExcelWriter(args.save_result_detail_path)
    for value in files:
        args.signal_path = value
        print(args.signal_path)
        # 写入数据库
        main_workflow(args, Napor_conn)
        # 统计数据分布
        print('---------------不加阈值分割线-----------------')
        # name, resDwellTime_, blockDepth_ = cat_ratio(Napor_conn, args.signal_path)
        # plt.figure(figsize=(10, 6))
        # # 绘制实验数据的散点图
        # plt.scatter(blockDepth_, resDwellTime_, color='red', marker='^', alpha=0.5, s=6)
        # plt.yscale('log')
        # plt.xlabel('Depth')
        # plt.ylabel('Dwell')
        # plt.xlim(0, 1)
        # plt.title(f'{value} VS ' + name)
        # plt.grid(True)
        # plt.show()
        # plt.close()
        # print("Scatter plots saved successfully!")
        df = cat_main_(Napor_conn, args.signal_path)
        if not df.empty:
            # 选择所需的四列
            selected_columns = ['prediction', 'Gaussian_mean', 'Gaussian_std', 'softmax_value', 'baseline']

            # 创建新的DataFrame只包含这四列
            new_df = df[selected_columns]

            new_df.to_excel(writer, sheet_name=args.signal_path.replace('/', '!').strip('.abf').split('!')[-1],
                        index=False)  # index=False表示不写入行索引
            writer.save()

    writer = pd.ExcelWriter(args.save_result_dir)
    for filename, df in results.items():
        df.to_excel(writer, sheet_name=filename.replace('/', '!').strip('.abf').split('!')[-1], index=False)  # index=False表示不写入行索引
        writer.save()



