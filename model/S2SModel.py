# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/8/2 16:07

"""
文件说明：  
"""
import torch
from torch import nn


class Self_Attn_Spatial(nn.Module):
    """
    func: Self attention Spatial Layer 自注意力机制.通过类似Transformer中的Q K V来实现
    inputs:
        in_dim: 输入的通道数
        out_dim: 在进行self attention时生成Q,K矩阵的列数, 一般默认为in_dim//8
    """

    def __init__(self, in_dim, out_dim):
        super(Self_Attn_Spatial, self).__init__()
        self.chanel_in = in_dim
        self.out_dim = out_dim

        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, length = x.size()

        # proj_query中的第i行表示第i个像素位置上所有通道的值。size = B X N × C1
        proj_query = self.query_conv(x).view(m_batchsize, -1, length).permute(0, 2, 1)

        # proj_key中的第j行表示第j个像素位置上所有通道的值，size = B X C1 x N
        proj_key = self.key_conv(x).view(m_batchsize, -1, length)

        # Energy中的第(i,j)是将proj_query中的第i行与proj_key中的第j行点乘得到
        # energy中第(i,j)位置的元素是指输入特征图第j个元素对第i个元素的影响，
        # 从而实现全局上下文任意两个元素的依赖关系
        energy = torch.bmm(proj_query, proj_key)  # transpose check

        # 对行的归一化，对于(i,j)位置即可理解为第j位置对i位置的权重，所有的j对i位置的权重之和为1
        attention = self.softmax(energy)  # B X N X N

        proj_value = self.value_conv(x).view(m_batchsize, -1, length)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B X C X N
        out = out.view(m_batchsize, C,length)  # B X C X W X H
        # 跨连，Gamma是需要学习的参数
        out = self.gamma * out + x  # B X C X W X H
        # return out, attention
        return out

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, bn=True, padding=2, dilation=1, stride=1, shortcut=False, activation=False):
        super().__init__()
        self.shortcut = None
        self.activation = None
        if bn:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            )
        else:
            self.basic_block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 5, padding=padding, dilation=dilation, stride=stride),
                nn.ReLU()
            )
        if shortcut:
            # Add a shortcut connection
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                      stride=stride) if in_channels != out_channels else None
        # Add an activation function after the residual connection
        if activation:
            self.activation = nn.ReLU()

    def forward(self, x):
        out = self.basic_block(x)
        # Use the shortcut connection if needed
        if self.shortcut is not None:
            x = self.shortcut(x)
            out = out + x  # Residual connection
        if self.activation is not None:
            out = self.activation(out)

        return out



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 4, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Classifier(nn.Module):
    def __init__(self, class_num, bn=False, dropout=0.3, shortcut=False, activation=False):
        super().__init__()
        self.short = shortcut
        self.class_num = class_num
        self.convBlock = nn.Sequential(
            Block(1, 8, bn, shortcut=self.short, activation=activation),
            Self_Attn_Spatial(8, 16),
            nn.MaxPool1d(kernel_size=2),
            Block(8, 16, bn, shortcut=self.short, activation=activation),
            # Self_Attn_Spatial(16, 32),
            nn.MaxPool1d(kernel_size=2),
            Block(16, 32, bn, shortcut=self.short, activation=activation),
            # Self_Attn_Spatial(32, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(32, 64, bn, shortcut=self.short, activation=activation),
            # Self_Attn_Spatial(64, 64),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 64, bn, shortcut=self.short, activation=activation),
            nn.MaxPool1d(kernel_size=2),
            Block(64, 128, bn, shortcut=self.short, activation=activation),
            nn.MaxPool1d(kernel_size=2),
            Block(128, 256, bn, shortcut=self.short, activation=activation),
            nn.MaxPool1d(kernel_size=2),
            Block(256, 512, bn, shortcut=self.short, activation=activation),
            nn.MaxPool1d(kernel_size=2)
            # Block(512, 1024, False),
            # SpatialAttention()
        )
        self.classification = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(512, self.class_num)
        )

    def forward(self, x):
        x = self.convBlock(x)
        x, _ = torch.max(x, dim=2)
        return self.classification(x)