# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/7/3 14:08

"""
文件说明：  
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

from model.Fourier import FourierBlock, SignalEmbedding
torch.manual_seed(1)

# 创建卷积神经网络模型
class ConvNetFour(nn.Module):
    def __init__(self, label, drop_ratio=0):
        super(ConvNetFour, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(3, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(3, 2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(3, 2)
        )
        self.liner1 = nn.Linear(128 * 116, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.liner2 = nn.Linear(1024, label)

    def forward(self, x):
        x = self.layer1(x)  # (64,1,30000)--->(64,16,7499)
        x = self.layer2(x)  # (64,16,7499)--->(64,32,1874)
        x = self.layer3(x)  # (64,32,1874)----> (64, 64, 467)
        x = self.layer4(x)  # (64,64,467)----> (64, 128, 116)
        x = x.view(-1, 128 * 116)  # (64, 64, 467)--->(64, 29888)
        x = self.liner1(x)
        x = self.dropout(x)
        x = F.relu(x)  # (64, 29888)--->(64, 1024)
        x = self.liner2(x)
        return x


# 定义一个卷积神经网络类，基于ResNet架构，用于1维数据的处理
class ConvNetResNet(nn.Module):
    # 初始化函数，设定卷积层数量和dropout比例
    def __init__(self, label, drop_ratio=0):
        super(ConvNetResNet, self).__init__()

        # 第一层卷积模块
        # Layer 1
        self.conv1_1 = nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1)
        self.bn1_1 = nn.BatchNorm1d(16)
        self.conv1_2 = nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm1d(16)
        self.shortcut1 = nn.Conv1d(1, 16, kernel_size=1, stride=2)

        # 第二层卷积模块
        # Layer 2
        self.conv2_1 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2_1 = nn.BatchNorm1d(32)
        self.conv2_2 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm1d(32)
        self.shortcut2 = nn.Conv1d(16, 32, kernel_size=1, stride=2)

        # 第三层卷积模块
        # Layer 3
        self.conv3_1 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm1d(64)
        self.conv3_2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Conv1d(32, 64, kernel_size=1, stride=2)

        # 第四层卷积模块
        # Layer 4
        self.conv4_1 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4_1 = nn.BatchNorm1d(128)
        self.conv4_2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm1d(128)
        self.shortcut4 = nn.Conv1d(64, 128, kernel_size=1, stride=2)

        # 全连接层，将卷积后的特征映射到指定的标签数量
        self.fc = nn.Linear(240000, label)
        # Dropout层，用于防止过拟合
        self.dropout = nn.Dropout(p=drop_ratio)

    # 前向传播函数
    def forward(self, x):
        # 第一层卷积模块的计算，包含残差连接
        # Layer 1
        out = F.relu(self.bn1_1(self.conv1_1(x)))
        out = self.bn1_2(self.conv1_2(out))
        shortcut = self.shortcut1(x)
        out += shortcut
        layer1_out = F.relu(out)

        # 第二层卷积模块的计算，包含残差连接
        # Layer 2
        out = F.relu(self.bn2_1(self.conv2_1(layer1_out)))
        out = self.bn2_2(self.conv2_2(out))
        shortcut = self.shortcut2(layer1_out)
        out += shortcut
        layer2_out = F.relu(out)

        # 第三层卷积模块的计算，包含残差连接
        # Layer 3
        out = F.relu(self.bn3_1(self.conv3_1(layer2_out)))
        out = self.bn3_2(self.conv3_2(out))
        shortcut = self.shortcut3(layer2_out)
        out += shortcut
        layer3_out = F.relu(out)

        # 第四层卷积模块的计算，包含残差连接
        # Layer 4
        out = F.relu(self.bn4_1(self.conv4_1(layer3_out)))
        out = self.bn4_2(self.conv4_2(out))
        shortcut = self.shortcut4(layer3_out)
        out += shortcut
        layer4_out = F.relu(out)

        # 将卷积层的输出展平，准备进入全连接层
        layer4_out = layer4_out.view(layer4_out.size(0), -1)
        # 应用dropout层
        out = self.dropout(layer4_out)
        # 通过全连接层得到最终的输出
        out = self.fc(out)
        return out


# 定义信号的时域特征提取模块（CNN）
class TimeDomainFeatureExtractor(nn.Module):
    def __init__(self, cnn_in_channel=1, out_channels=64):
        super(TimeDomainFeatureExtractor, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(cnn_in_channel, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )

    def forward(self, x):
        x = self.layer(x)
        x = self.layer2(x)
        return x

# 定义信号的时域特征提取模块（CNN）
class TimeDomainFeatureExtractor2(nn.Module):
    def __init__(self, cnn_in_channel=1, out_channels=64):
        super(TimeDomainFeatureExtractor2, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(cnn_in_channel, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, out_channels, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.liner = nn.Linear(2047, 64)

    def forward(self, x):
        x = self.layer(x)
        x = self.layer2(x)
        x = self.liner(x)
        return x

# 定义信号的频域特征提取模块（FFT）
class FrequencyDomainFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, seq_len=30000, modes=256, mode_select_method='random'):
        super(FrequencyDomainFeatureExtractor, self).__init__()
        self.FourierModel = FourierBlock(in_channels, out_channels, seq_len, modes, mode_select_method)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.FourierModel(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DataEmbedding_onlypos(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_onlypos, self).__init__()

        self.value_embedding = SignalEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # try:
        x = self.value_embedding(x) + self.position_embedding(x)
        # except:
        #     a = 1
        return self.dropout(x)


# 定义整合模型
class FusionModel(nn.Module):
    def __init__(self, in_channels=1, out_channels=24, seq_len=30000, modes=256, d_model=192, mode_select_method='random'):
        super(FusionModel, self).__init__()
        self.time_model = TimeDomainFeatureExtractor() #
        self.freq_model = FrequencyDomainFeatureExtractor(in_channels, out_channels, seq_len, modes, mode_select_method)
        # 编码器
        self.token_emb = DataEmbedding_onlypos(128, d_model=d_model)
        n_head = d_model // 64
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, out_channels)

    def forward(self, x):
        time_feature = self.time_model(x)  # batchsize, 64, 2047
        freq_feature = self.freq_model(x)   # batchsize, 64, 2047
        # freq_feature = freq_feature.squeeze(dim=2)
        fused_feature = torch.cat((time_feature, freq_feature), dim=1)  # batchsize, 128, 2047
        fused_feature = fused_feature.permute(0, 2, 1)  # batchsize, 2047, 128, 2047
        token_em = self.token_emb(fused_feature)  # batchsize, 2047, 192
        fused_output = self.transformer(token_em)  # batchsize, 2047, 192
        # pooled_output = fused_output.mean(dim=1)
        pooled_output = fused_output[:, 0, :]
        logits = self.fc(pooled_output)
        return logits


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True)

    def forward(self, x):
        # LSTM期望的输入形状是(batch, seq_len, input_size)
        output, _ = self.lstm(x)
        return output


class CrossAttention(nn.Module):
    def __init__(self, key_dim):
        super(CrossAttention, self).__init__()
        self.scale = torch.sqrt(torch.tensor(key_dim, dtype=torch.float32))

    def forward(self, query, key, value):
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        # 使用注意力权重对值进行加权求和
        output = torch.matmul(attention_weights, value)
        return output


class FusionModel2(nn.Module):
    def __init__(self, time_series_dim, cnn_in_channel, cnn_out_channels, lstm_hidden_size, num_layers, num_classes):
        super(FusionModel2, self).__init__()
        self.cnn = TimeDomainFeatureExtractor2(cnn_in_channel, cnn_out_channels) # cnn_in_channel是频域输入维度
        self.lstm = BiLSTM(time_series_dim, lstm_hidden_size, num_layers)
        # 注意：这里我们假设使用LSTM的最后一个时间步的隐藏状态作为查询
        self.query_projection = nn.Linear(lstm_hidden_size * 2, cnn_out_channels)  # 双向LSTM，所以隐藏状态维度加倍
        self.cross_attention = CrossAttention(cnn_out_channels)
        self.fc = nn.Linear(cnn_out_channels, num_classes)  # 假设最终的分类层

    def forward(self, req, time_x):
        # 处理频域特征
        cnn_out = self.cnn(req)

        # 处理时域特征
        lstm_out = self.lstm(time_x)
        # 取双向LSTM的最后一个时间步的隐藏状态（合并两个方向）
        lstm_last_hidden = lstm_out[:, -1, :]
        # 将LSTM的隐藏状态投影到与CNN输出相同的特征维度
        query = self.query_projection(lstm_last_hidden)

        # 注意：这里我们仅对CNN输出的最后一个时间步进行交叉注意力（为了简化）
        # 在实际应用中，您可能需要设计更复杂的机制来处理整个序列
        key = cnn_out
        value = cnn_out

        # 调用交叉注意力模块
        fused_feature = self.cross_attention(query.unsqueeze(1), key, value)  # 增加一个维度以匹配 num_queries

        # 由于 fused_feature 的形状现在是 [batch_size, 1, cnn_out_channels]，
        # 我们可能需要进一步处理它（例如，取平均、最大池化或使用其他聚合方法）
        # 这里我们简单地 squeeze 掉 num_queries 维度
        fused_feature = fused_feature.squeeze(1)

        # 最终的分类
        output = self.fc(fused_feature)
        return output

