# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/8/22 12:29

"""
文件说明：  从tensorboard下载了训练集acc、loss和验证集acc、loss这四个csv文件，现在需要将其画在同一个图上
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data(train_acc_path, train_loss_path, val_acc_path, val_loss_path):
    # 读取CSV文件
    train_acc = pd.read_csv(train_acc_path)
    train_loss = pd.read_csv(train_loss_path)
    val_acc = pd.read_csv(val_acc_path)
    val_loss = pd.read_csv(val_loss_path)

    # 创建一个新的DataFrame，将所有数据合并到一起
    data = pd.concat([train_acc.assign(Type='Training Accuracy', Dataset='Training'),
                      val_acc.assign(Type='Validation Accuracy', Dataset='Validation'),
                      train_loss.assign(Type='Training Loss', Dataset='Training'),
                      val_loss.assign(Type='Validation Loss', Dataset='Validation')], ignore_index=True)

    # 使用seaborn绘制数据
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制左侧y轴（损失）
    sns.lineplot(data=data[data['Type'].str.contains('Loss')],
                 x='Step', y='Value', hue='Dataset',
                 style='Dataset', markers=False, dashes=True, ax=ax1)

    # 设置左侧y轴的标签
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建右侧y轴（准确率）
    ax2 = ax1.twinx()
    sns.lineplot(data=data[data['Type'].str.contains('Accuracy')],
                 x='Step', y='Value', hue='Dataset',
                 style='Dataset', markers=False, dashes=False, ax=ax2)

    # 设置右侧y轴的标签
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 设置图表标题和坐标轴标签
    plt.title('Training and Validation Accuracy and Loss')
    plt.xlabel('Epoch')

    # 添加图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='upper right')

    # 显示图表
    plt.show()
def plot_data2(train_acc_path, train_loss_path, val_acc_path, val_loss_path):
    # 读取CSV文件
    train_acc = pd.read_csv(train_acc_path)
    train_loss = pd.read_csv(train_loss_path)
    val_acc = pd.read_csv(val_acc_path)
    val_loss = pd.read_csv(val_loss_path)

    # 使用seaborn绘制数据
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制左侧y轴（损失）
    sns.lineplot(data=train_loss, x='Step', y='Value', color='tab:blue', linestyle='--', label='Training Loss', ax=ax1)
    sns.lineplot(data=val_loss, x='Step', y='Value', color='tab:orange', linestyle='--', label='Validation Loss', ax=ax1)

    # 设置左侧y轴的标签
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')

    # 创建右侧y轴（准确率）
    ax2 = ax1.twinx()
    sns.lineplot(data=train_acc, x='Step', y='Value', color='tab:blue', linestyle='-', label='Training Accuracy', ax=ax2)
    sns.lineplot(data=val_acc, x='Step', y='Value', color='tab:orange', linestyle='-', label='Validation Accuracy', ax=ax2)

    # 设置右侧y轴的标签
    ax2.set_ylabel('Accuracy')
    ax2.tick_params(axis='y')

    # 设置图表标题和坐标轴标签
    # plt.title('Training and Validation Accuracy and Loss')
    # plt.xlabel('Epoch')

    # 添加图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'], loc='center right')
    # 显示图表
    plt.show()

# root_path  ='D:\\BaiduSyncdisk\\简历\\纳米孔数据分析\\S2S结果\\'
root_path = ''
# CSV文件路径
train_acc_path = root_path+'S2SFocalLoss512BinwithBN_trainAcc.csv'
train_loss_path = root_path+'S2SFocalLoss512BinwithBN_trainLoss.csv'
val_acc_path = root_path+'S2SFocalLoss512BinwithBN_valAcc.csv'
val_loss_path = root_path+'S2SFocalLoss512BinwithBN_valLoss.csv'

# 调用函数
plot_data2(train_acc_path, train_loss_path, val_acc_path, val_loss_path)
