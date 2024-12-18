# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/8/2 13:46

"""
文件说明：  本文件将不等长的纳米孔事件转化为等长的直方图
"""

import yaml
from sklearn.model_selection import train_test_split
import os
from glob import glob
import numpy as np
import pandas
import pandas as pd
import pyabf


def histogram_mapping(sequence, d):
    # 计算直方图
    hist, bin_edges = np.histogram(sequence, bins=d)

    # 计算概率
    probabilities = hist / float(len(sequence))

    return probabilities, bin_edges

def dataset_napor_data_new(data_path, args):
    files = []
    features = []
    labels = []
    baselines = []
    sigmas = []
    # hist_data_bin1024 = []
    # hist_data_bin2048 = []
    # hist_data_bin512 = []
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
                            interval_right_length = 4  # 如果存在50点内就出现下一个信号
                        if interval_length >= 9 and interval_right_length >= 9:
                            temp = signal_data[block[0] - 9: block[1] + 9]
                        elif interval_length >= 9 and interval_right_length < 9:
                            temp = signal_data[block[0] - 9: block[1] + interval_right_length - 2]
                        elif interval_length < 9 and interval_right_length >= 9:
                            temp = signal_data[block[0] - interval_length + 2: block[1] + 9]
                        else:
                            temp = signal_data[block[0] - interval_length + 2: block[1] + interval_length - 2]
                    else:
                        temp = signal_data[block[0] - 9: block[1] + 9]  # 以前是这样减去9
                    # 将小于0的都替换为0
                    t = np.where(temp > 0, temp, 0)
                    # 替换nan
                    t = np.where(np.isnan(t), 0, t)
                    # probabilities, _ = histogram_mapping(t, 1024)  # bin=1024
                    # pro2048, _ = histogram_mapping(t, 2048)
                    # pro512, _ = histogram_mapping(t, 512)
                    features.append(t)
                    # hist_data_bin1024.append(probabilities)
                    # hist_data_bin2048.append(pro2048)
                    # hist_data_bin512.append(pro512)
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
    df = pandas.DataFrame({'signal': features, 'labels': labels, 'baselines': baselines, 'sigma': sigmas})
    df.to_json(args['data']['save_path'])
    return df


def getNUM(df, output_file):
    grouped = df.groupby('labels').size().reset_index(name='count')
    # 将结果保存到指定的Excel文件中
    grouped.to_excel(output_file, index=False)

if __name__ == '__main__':
    with open(os.path.join('../config/filter.yaml'), 'r') as file:
        # 版本问题需要将Loader=yaml.CLoader加上
        config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)

    df = dataset_napor_data_new(config_yaml['data']['filter_path'], config_yaml)
    getNUM(df, '../result/count_hist.xlsx')



