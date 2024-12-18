# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/7/5 13:03

"""
文件说明：  保存文件验证集测试结果
"""
import argparse

from torchsummary import summary
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from tqdm import tqdm

from model.S2SModel import Classifier
from model.models import ConvNetFour
from util.histDataLoader import myNaporDataset
from util.naporDataLoder import *
import os
import seaborn as sns

# 超参数设置
with open(os.path.join('./config/test.yaml'), 'r') as file:
    # 版本问题需要将Loader=yaml.CLoader加上
    config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)

# 超参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--test_data_path', default=config_yaml['test_settings']['test_data_path'], type=str)
parser.add_argument('--model_path', default=config_yaml['test_settings']['model_path'], type=str)
parser.add_argument('--split_rate', default=config_yaml['test_settings']['split_rate'], type=str)
parser.add_argument('--batchsize', default=config_yaml['test_settings']['batch_size'], type=int)
parser.add_argument('--num_work', default=config_yaml['test_settings']['num_works'], type=int)
parser.add_argument('--device', default=config_yaml['test_settings']['gpu'], type=str)
parser.add_argument('--classfication', default=config_yaml['test_settings']['classfication'], type=int)  # 分类数
parser.add_argument('--save_result_path', default=config_yaml['test_settings']['save_result_path'], type=str)  # 所有阻塞事件样本详细结果，包括softmax值
parser.add_argument('--save_accuracy_info', default=config_yaml['test_settings']['save_accuracy_info'], type=str)
parser.add_argument('--log_path', default=config_yaml['test_settings']['log_path'], type=str)  # 混淆矩阵图# 每个类别统计的结果
parser.add_argument('--confusion_name', default=config_yaml['test_settings']['confusion_name'], type=str)  # 混淆矩阵图

# 定义验证流程
def validate(val_loader, model, params, writer, label_info):
    metric_monitor = MetricMonitor()  # 验证流程
    model.eval()  # 模型设置为验证格式
    stream = tqdm(val_loader)  # 设置进度条
    predict_label = []
    all_embeddings = []
    all_metadata = []
    # 测试置信度值
    predict_softmax_value = []
    autocast = torch.cuda.amp.autocast
    with torch.no_grad():  # 开始推理
        for i, (signal, target) in enumerate(stream, start=1):
            with autocast():
                signal = signal.unsqueeze(1)
                signal = signal.to(params.device, non_blocking=True)  # 读取图片
                target = target.to(params.device, non_blocking=True)  # 读取标签
                output = model(signal)  # 前向传播
                # 将embedding和元数据添加到列表中
                embeddings_batch = output.detach().cpu().numpy()
                metadata_batch = [label_info.get(label) for label in target.tolist()]
                all_embeddings.extend(embeddings_batch)
                all_metadata.extend(metadata_batch)

                # 将每个样本的嵌入向量和元数据添加到列表中
                # for embedding, metadata in zip(embeddings_batch, metadata_batch):
                #     all_embeddings.append(list(embedding))
                #     all_metadata.append(metadata)
                writer.add_embedding(embeddings_batch, metadata=metadata_batch, global_step=i)

            y_pred = torch.softmax(output, dim=1)
            value, index = torch.max(y_pred, dim=1)
            predict_softmax_value += value.cpu().tolist()
            # y_pred = torch.argmax(y_pred, dim=1).cpu().tolist()
            predict_label += index.cpu().tolist()
            acc = accuracy(output, target)  # 计算acc
            metric_monitor.update('Accuracy', acc)
            stream.set_description(
                "Validation. {metric_monitor}".format(
                    metric_monitor=metric_monitor)
            )
        # 将所有embedding转换为numpy数组
        # all_embeddings = np.concatenate(all_embeddings)
        # writer.add_embedding(all_embeddings, metadata=all_metadata, global_step=i)
    df = pd.DataFrame({'all_embeddings': all_embeddings, 'all_metadata': all_metadata},)
    df.to_json('S2SFocalLoss512BinWithBN_Result.json')
    return metric_monitor.metrics['Accuracy']["avg"], predict_label, predict_softmax_value


def plot_mix_label(df, file_name, confusion_name, label_info):
    # 使用 groupby 方法按 'true_label' 列对数据进行分组
    grouped = df.groupby('true_label')


    xticks = [label_info[label] for label in grouped.groups.keys()]
    # 初始化一个空字典来存储每个类别的预测准确率
    accuracy_list = []
    for name, group in grouped:
        accuracy = accuracy_score(group['true_label'], group['predict'])
        num_samples = len(group)
        accuracy_list.append({'True Label': name, 'Num Samples': num_samples, 'Accuracy': accuracy})
    # 将结果存储在新的DataFrame中
    accuracy_df = pd.DataFrame(accuracy_list)

    # 按照label降序排序，
    accuracy_df = accuracy_df.sort_values(by='True Label', ascending=True)
    # 计算每个类别的数据数量
    accuracy_df['name'] = xticks
    # 将数据保存为CSV文件
    accuracy_df.to_csv(file_name, index=False)

    # 画出结果
    plt.figure(figsize=(10, 6))
    plt.bar(accuracy_df['True Label'], accuracy_df['Accuracy'])
    plt.xlabel('True Label')
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy per True Label')
    plt.show()
    plt.close()
    true_labels = df['true_label'].tolist()
    pred_labels = df['predict'].tolist()
    cm = confusion_matrix(true_labels, pred_labels)
    # 计算每个类别的总数
    class_totals = cm.sum(axis=1, keepdims=True)
    # 将混淆矩阵中的值转换为百分比
    cm_percentage = cm / class_totals.astype(float) * 100
    pd.DataFrame(cm_percentage, index=xticks, columns=xticks).to_csv(confusion_name+'confusion_matrix.csv', index=True)
    # 使用Seaborn绘制混淆矩阵
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm_percentage, annot=True, fmt=".1f", cmap="Blues", xticklabels=xticks,
                yticklabels=xticks)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(confusion_name+'.svg', format='svg', dpi=300)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    args.device = 'cuda:0'
    if torch.cuda.is_available():
        device = torch.device(args.device)
        # 固定随机种子，保证实验结果是可以复现的
        seed = 34
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    with open(os.path.join('./config/filter.yaml'), 'r') as file:
        # 版本问题需要将Loader=yaml.CLoader加上
        config_yaml = yaml.load(file.read(), Loader=yaml.CLoader)
    label_info = {}
    for da in config_yaml:
        if 'label' in config_yaml[da]:
            label_info[config_yaml[da]['label']] = da

    accs = []
    losss = []
    val_accs = []
    val_losss = []
    # writer = SummaryWriter(args.log_path)
    #
    # # df = pd.read_json(args.test_data_path)
    # # test_features, test_labels, test_baselines, sigmas = df['feature'].values, df['labels'].values, df[
    # #     'baselines'].values, df['sigma'].values
    # # test_dataset = naporDataset(test_features, test_labels, test_baselines, 30000, padding_type_='right',
    # #                             gaussian_=False, gaussian_std_=sigmas)
    # # 训练集和test在同一个数据集
    # df = pd.read_json(args.test_data_path)
    # signal_data, labels, baselines = df['signal'].values, df['labels'].values, df['baselines'].values
    # result = []
    # for index, value in enumerate(signal_data):
    #     result.append([value, baselines[index]])
    # train_features, test_features, train_labels, test_labels = train_test_split(result, labels,
    #                                                                             test_size=args.split_rate,
    #                                                                             stratify=labels, random_state=42)
    # data_features, data_baselines = [], []
    # for index, value in enumerate(test_features):
    #     data_features.append(value[0])
    #     data_baselines.append(value[1])
    # test_dataset = myNaporDataset(data_features, test_labels, data_baselines, 512)
    # del df
    #
    # val_loader = DataLoader(  # 按照批次加载验证集
    #     test_dataset, batch_size=args.batchsize, shuffle=False,
    #     num_workers=args.num_work, pin_memory=True,
    # )
    # model = ConvNetFour(label=args.classfication)  # 加载模型
    model = Classifier(class_num=args.classfication, bn=True, activation=True)

    checkpoint = torch.load(args.model_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model'])
    model = model.to(args.device)  # 模型部署到设备上
    summary(model,(1,512))
    # val_acc, pre_label, predict_softmax_value = validate(val_loader, model, args, writer, label_info)
    # re = {'predict': pre_label, 'true_label': test_labels, 'predict_softmax_value': predict_softmax_value}
    # t = pandas.DataFrame(re)
    # t.to_csv(args.save_result_path)
    # # t = pandas.read_csv(args.save_result_path)
    # plot_mix_label(t, args.save_accuracy_info, args.confusion_name, label_info)


