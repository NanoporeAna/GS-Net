# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/7/3 11:27

"""
文件说明：  
"""
# -*- coding:UTF-8 -*-
# author:Lucifer_Chen
# contact: zhangchen@17888808985.com
# datetime:2023/9/28 10:01


import pywt
import yaml
from sklearn.model_selection import train_test_split

import math

import os
from glob import glob

import numpy as np
import pandas
import pandas as pd
import pyabf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import f1_score, accuracy_score, recall_score
from collections import defaultdict

class naporDataset(Dataset):
    def __init__(self, feature, label, baseline, padding, padding_type_, gaussian_, gaussian_std_, freq=False):
        self.feature = feature
        self.baseline = baseline
        self.padding = padding
        self.label = label
        self.padding_type = padding_type_
        self.gaussian = gaussian_
        self.gaussian_std = gaussian_std_
        self.freq = freq

    def __len__(self):
        return self.label.__len__()

    # 定义变换函数，这里我们应用小波变换
    def wavelet_transform(self, signal):
        """
        对信号进行离散小波变换。

        Args:
            signal (np.array): 输入信号。

        Returns:
            np.array: 变换后的频域特征。
        """
        coeffs = pywt.dwt(signal, 'db1', mode='symmetric')
        cA, cD = coeffs  # cA是近似系数，cD是细节系数
        # 这里我们可以选择只保留cA或cD，或者将它们连接起来
        # 这里简单地将它们连接起来作为特征
        transformed_signal = np.concatenate([cA, cD])
        return transformed_signal

    def __getitem__(self, idx):
        currentSingal = normal_naporDataset(self.feature[idx], self.baseline[idx], self.padding,
                                            padding_type=self.padding_type, gaussian_noise=self.gaussian,
                                            gaussian_sigma=self.gaussian_std[idx] / self.baseline[idx] / 2)
        if self.freq:
            wavelet = self.wavelet_transform(currentSingal)
            wavelet = torch.tensor(wavelet, dtype=torch.float32)
        currentSingal = torch.tensor(currentSingal, dtype=torch.float32)
        # currentSingal = torch.tensor(currentSingal, dtype=torch.bfloat16)
        # padding_mask = get_padding_mask(currentSingal)
        if self.freq:
            return wavelet, currentSingal, self.label[idx]
        else:
            return currentSingal, self.label[idx]
        # return currentSingal, padding_mask, self.label[idx]


def get_padding_mask(input):
    tensor = torch.where(input == 1, input, torch.zeros_like(input))
    return ~tensor.bool()


def normal_naporDataset(block_segment, baseline, padding, padding_type='right', gaussian_noise=False,
                        gaussian_sigma=0.0072, maskPadding=0):
    # 原先是0，现改成基线值
    block_segment = np.nan_to_num(block_segment, nan=baseline, posinf=baseline, neginf=baseline)
    # 信号每个值减掉基线值，使得数据最大值为0,因为输入到CNN中，0就相当于对此部分数据不学习
    block_segment = block_segment - baseline
    if baseline == 0:
        baseline = 1
    block_segment = block_segment / baseline
    length = len(block_segment)
    if length < padding:
        t = padding - length
        if padding_type == 'right':
            if gaussian_noise:
                block_segment = list(block_segment) + list(np.random.normal(1, gaussian_sigma, t))
            else:
                block_segment = list(block_segment) + [maskPadding] * t
        elif padding_type == 'left':
            if gaussian_noise:
                block_segment = list(np.random.normal(1, gaussian_sigma, t)) + list(block_segment)
            else:
                block_segment = [maskPadding] * t + list(block_segment)
        else:
            left = t // 2
            right = t - left
            if gaussian_noise:
                block_segment = list(np.random.normal(1, gaussian_sigma, left)) + list(block_segment) + list(
                    np.random.normal(maskPadding, gaussian_sigma, right))
            else:
                block_segment = [maskPadding] * left + list(block_segment) + [maskPadding] * right
    else:
        block_segment = list(block_segment[0:padding])
    return block_segment


def dataset_napor_data(data_path, args):
    files = []
    features = []
    labels = []
    baselines = []
    sigmas = []
    for filename in os.listdir(data_path):
        file_paths = os.path.join(data_path, filename)
        for filename in glob(file_paths + '/*.abf'):
            files.append(filename.strip('.abf'))
    for index, file in enumerate(files):
        label_name = file.split('/')[-2]
        abf_file_path = '..' + file + ".abf"
        # 使用pyabf打开abf文件
        abf = pyabf.ABF(abf_file_path)
        try:
            # 获取数据
            data = abf.data
            signal_data = np.array(data[0])
            result = pd.read_json('..' + file + 'filterNot.json')
            # 阻塞事件自适应基线
            baseline = result['baseline'].values
            dwellTime = result['dwellTime'].values
            blockDepth = result['blockDepth'].values
            # 阻塞事件开头和结束点
            block_segment = result['blockedSegment']
            # blockflag为1表示经过等效电路检测后正常
            blockFlag = result['blockFlag']
            threhold = result['threshold'].values
            sigma = (baseline[0] - threhold[0]) / 12
            before_tag = block_segment[0][1]  # 确保阻塞事件左边
            next_tag = block_segment[1][0]  # 确保阻塞事件右边
            sum_block = len(block_segment)
            for j, block in enumerate(block_segment):
                flag = False
                if dwellTime[j]>args[label_name]['DwellMin'] and dwellTime[j]<args[label_name]['DwellMax'] and blockDepth[j]>args[label_name]['BlockDepthMin'] and blockDepth[j]<args[label_name]['BlockDepthMax']:
                    flag = True
                if flag and blockFlag[j] == 1:
                    if j != 0:
                        interval_length = block[0] - before_tag
                        interval_right_length = next_tag - block[1]
                        if interval_right_length < 0:
                            interval_right_length = 20  # 如果存在50点内就出现下一个信号
                        if interval_length >= 500 and interval_right_length >= 500:
                            temp = signal_data[block[0] - 500: block[1] + 500]
                        elif interval_length >= 500 and interval_right_length < 500:
                            temp = signal_data[block[0] - 500: block[1] + interval_right_length - 2]
                        elif interval_length < 500 and interval_right_length >= 500:
                            temp = signal_data[block[0] - interval_length + 2: block[1] + 500]
                        else:
                            temp = signal_data[block[0] - interval_length + 2: block[1] + interval_length - 2]
                    else:
                        temp = signal_data[block[0] - 500: block[1] + 500]  # 以前是这样减去9
                    # 将小于0的都替换为0
                    t = np.where(temp > 0, temp, 0)
                    features.append(np.where(np.isnan(t), 0, t))
                    labels.append(args[label_name]['label'])
                    baselines.append(baseline[j])
                    sigmas.append(sigma)
                before_tag = block[1] + 50
                if j < sum_block - 2:
                    next_tag = block_segment[j + 2][0] - 50
                else:
                    next_tag = block[1] + 500000
        except:
            print(abf_file_path)
    df = pandas.DataFrame({'feature': features, 'labels': labels, 'baselines': baselines, 'sigma': sigmas})
    df.to_json(args['data']['save_path'])
    return features, labels, baselines, sigmas


# 测试准确率
def accuracy(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return accuracy_score(target, y_pred)


# 计算f1
def calculate_f1_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()

    return f1_score(target, y_pred, average='macro')


# 计算recall
def calculate_recall_macro(output, target):
    y_pred = torch.softmax(output, dim=1)
    y_pred = torch.argmax(y_pred, dim=1).cpu()
    target = target.cpu()
    # tp fn fp
    return recall_score(target, y_pred, average="macro", zero_division=0)


# 训练的时候输出信息使用
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

# 调整学习率
def adjust_learning_rate(optimizer, epoch, params, batch=0, nBatch=None):
    """ adjust learning of a given optimizer and return the new learning rate """
    new_lr = calc_learning_rate(epoch, params['lr'], params['epochs'], batch, nBatch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

""" learning rate schedule """
# 计算学习率
def calc_learning_rate(epoch, init_lr, n_epochs, batch=0, nBatch=None, lr_schedule_type='cosine'):
    if lr_schedule_type == 'cosine':
        t_total = n_epochs * nBatch
        t_cur = epoch * nBatch + batch
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * t_cur / t_total))
    elif lr_schedule_type is None:
        lr = init_lr
    else:
        raise ValueError('do not support: %s' % lr_schedule_type)
    return lr


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


"""  平滑损失函数 """
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss

def getNUM(df, output_file):
    grouped = df.groupby('labels').size().reset_index(name='count')
    # 将结果保存到指定的Excel文件中
    grouped.to_excel(output_file, index=False)

if __name__ == '__main__':
    with open(os.path.join('../config/filter.yaml'), 'r') as file:
        # 版本问题需要将Loader=yaml.CLoader加上
        config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)

    features, labels, baselines, sigmas = dataset_napor_data(config_yaml['data']['filter_path'], config_yaml)
    result = []
    for index, value in enumerate(features):
        result.append([value, baselines[index], sigmas[index]])
    train_features, test_features, train_labels, test_labels = train_test_split(result, labels, test_size=config_yaml['data']['split_rate'],
                                                                                stratify=labels)
    data_features, data_baselines, data_sigma = [], [], []
    for index, value in enumerate(train_features):
        data_features.append(value[0])
        data_baselines.append(value[1])
        data_sigma.append(value[2])
    df_train = pd.DataFrame(
        {'feature': data_features, 'labels': train_labels, 'baselines': data_baselines, 'sigma': data_sigma})
    getNUM(df_train, '../result/train_label.xlsx')
    df_train.to_json(config_yaml['data']["train_path"])

    data_Test_features, data_Test_baselines, data_Test_sigma = [], [], []
    for index, value in enumerate(test_features):
        data_Test_features.append(value[0])
        data_Test_baselines.append(value[1])
        data_Test_sigma.append(value[2])
    df_test = pd.DataFrame({'feature': data_Test_features, 'labels': test_labels, 'baselines': data_Test_baselines,
                            'sigma': data_Test_sigma})
    getNUM(df_test, '../result/val_label.xlsx')
    df_test.to_json(config_yaml['data']["val_path"])


