# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/8/2 14:33

"""
文件说明：  
"""
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

class myNaporDataset(Dataset):
    def __init__(self, signalData, label, baseline, bin):
        self.signalData = signalData
        self.label = label
        self.baseline = baseline
        self.bin = bin

    def __len__(self):
        return self.label.__len__()

    def normalize_data(self, data):
        mean = data.mean()
        std = data.std()
        return (data - mean) / std

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
    def __getitem__(self, idx):
        currentSingal = self.my_binning(self.signalData[idx], self.baseline[idx], self.bin)

        currentSingal = torch.tensor(currentSingal, dtype=torch.float32)
        return currentSingal, self.label[idx]


