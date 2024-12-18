# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/8/21 10:24

"""
文件说明：  绘制t-SNE
"""
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib import pyplot as plt


def plot_tsne(features, labels):
    '''
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    '''
    import pandas as pd
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    import seaborn as sns

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print('tsne_features的shape:', tsne_features.shape)
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1])  # 将对降维的特征进行可视化
    plt.show()

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", class_num),
                    data=df).set(title="Bearing data T-SNE projection unsupervised")
# 读取数据
filepath = '/home/gpu3/HSQCAI/nap/S2SFocalLoss512BinWithBN_Result.json'
df = pd.read_json(filepath)

# 按照 'all_metadata' 列进行分组，并从每个分组中随机抽取100条记录
# 注意：如果某个分组的数据少于100条，则会抽取该分组所有数据
sampled_df = df.groupby('all_metadata').apply(lambda x: x.sample(n=400, random_state=42)).reset_index(drop=True)
fea = sampled_df['all_embeddings'].tolist()
# plot_tsne(np.array(fea), labels=sampled_df['all_metadata'].values)
# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2).fit_transform(np.array(fea))
# 定义一个函数来将数据缩放到0-1范围
def scale_to_01_range(x):
    # 计算分布范围
    value_range = (np.max(x) - np.min(x))

    # 移动分布使其从零开始
    # 通过从所有值中减去最小值
    starts_from_zero = x - np.min(x)

    # 使分布适应[0; 1]区间，通过除以其范围
    return starts_from_zero / value_range

# 提取x和y坐标表示图像在t-SNE图上的位置
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

# 初始化matplotlib绘图
plt.figure(figsize=(10, 8))

# 使用 Seaborn 绘制散点图
# 根据 'all_metadata' 列中的类别使用不同的颜色
sns.scatterplot(
    x=tx, y=ty,
    hue=sampled_df['all_metadata'],
    palette=sns.color_palette("hls", len(sampled_df['all_metadata'].unique())),
    legend="full",
    alpha=0.7
)
# # 自定义图例位置，使其位于图表外部
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
# 设置 x 轴的范围
plt.xlim(0, 1.2)

# 隐藏 x 轴的刻度
plt.xticks([])
plt.yticks([])
# 设置图表标题和轴标签
plt.title('t-SNE at S2S Model')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')

# 显示图表
plt.show()
temp = pd.DataFrame({'t-SNE Dimension 1': tx, 't-SNE Dimension 2': ty, 'label': sampled_df['all_metadata']})
temp.to_csv('/home/gpu3/HSQCAI/nap/result/S2S_t-SNE.csv', index=False)
