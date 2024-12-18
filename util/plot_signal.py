# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/7/4 10:11

"""
文件说明：  
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_signal(signal_data, savename, title, threshold):
    plt.figure()
    plt.title(title)
    plt.plot(signal_data, label='Threshold signal')
    plt.axhline(threshold, color='red', linestyle='--')
    plt.savefig('../picture/' + savename + '.jpg')
    plt.show()

def getNUM(df, output_file):
    grouped = df.groupby('labels').size().reset_index(name='count')
    # 将结果保存到指定的Excel文件中
    grouped.to_excel(output_file, index=False)


# 自定义分箱函数，根据baseline 和 bin箱数来划分落到的bin里的概率
# sequence: 输入序列
# bin: 直方图的bin数
# 返回概率和bin
def my_binning(sequence, baseline, bin):
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
    for value in sequence:
        # 计算当前元素所属的箱的索引
        s = int(value / p_step)
        # 如果计算出的索引超出了箱的数量范围，将其设置为最后一个箱的索引
        if s >= bin:
            s = bin-1
        # 将当前元素所属箱的数量加一
        P[s] += 1
    # 返回分箱后的结果
    return P/float(len(sequence))

def histogram_mapping(sequence, bin=1024):
    # 计算直方图
    hist, bin_edges = np.histogram(sequence, bins=bin)
    # 计算概率
    probabilities = hist / float(len(sequence))
    return probabilities, bin_edges

def plot_signal_and_histograms(signal_data, baseline, savename):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # 绘制原始信号数据
    axs[0, 0].plot(signal_data)
    axs[0, 0].set_title('Original Signal Data')

    # 绘制不同bin数量的直方图
    for i, bin_count in enumerate([512, 1024, 2048]):
        t = i+1
        row = t // 2
        col = t % 2
        probabilities = my_binning(signal_data, baseline, bin_count)
        # 使用barh函数横向显示直方图
        axs[row, col].barh(np.arange(len(probabilities)), probabilities)

        axs[row, col].set_title(f'Histogram with {bin_count} bins')

    plt.tight_layout()
    plt.savefig('../picture/' + savename + '.jpg')

if __name__ == '__main__':
    filepath = '../data/train/Ginsenoside/hist_data_all.json'
    df = pd.read_json(filepath)
    random_rows = df.sample(n=100)
    for index, sig in random_rows.iterrows():
        # plot_signal(sig['feature'], str(index), 'signal', sig['baselines']-12*sig['sigma'])
        plot_signal_and_histograms(sig['signal'], sig['baselines'], str(index))
