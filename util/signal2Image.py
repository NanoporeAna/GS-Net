# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/8/2 10:18

"""
文件说明：  
"""
import matplotlib.pyplot as plt
import pandas
from pyts.approximation import PiecewiseAggregateApproximation
from pyts.datasets import load_gunpoint
from pyts.image import MarkovTransitionField, GramianAngularField

# df = pandas.read_json('../data/train/Ginsenoside/new_all_val.json.json')
# data = df['feature'].vaules
# data = data[:1000]
# n_sampels, n_timestamps = data.shape
# n_paa = 1024
# window_size = n_timestamps // n_paa
# paa = PiecewiseAggregateApproximation(window_size=window_size)
# X_paa = paa.transform(data)[:, :n_paa]

X, _, _, _ = load_gunpoint(return_X_y=True)
gasf = GramianAngularField(image_size=64)
X_gasf = gasf.transform(X)

plt.figure(figsize=(10, 10))
plt.imshow(X_gasf[0], cmap='rainbow')
plt.show()

