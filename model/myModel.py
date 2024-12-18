# -*- coding:UTF-8 -*-
# author:Lucifer_Chen(zhangchen)
# contact: 17888808985@163.com
# datetime:2024/7/25 15:05

"""
文件说明：
"""
import torch
import torch.nn as nn


# 定义BasicBlock类，作为ResNet1D的基本块
class BasicBlock(nn.Module):
    """
    ResNet基本块的实现，包含两个卷积层和两个批量归一化层。

    参数:
    - in_channels: 输入通道数。
    - out_channels: 输出通道数。
    - stride: 卷积层的步长，默认为1。
    - downsample: 下采样层，用于更新特征图大小，默认为None。

    属性:
    - expansion: 基本块的扩张率，用于计算输出通道数。
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        基本块的前向传播过程。

        参数:
        - x: 输入的特征图。

        返回:
        - 输出的特征图。
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# 定义ResNet1D类，用于实现一维ResNet网络
class ResNet1D(nn.Module):
    """
    一维ResNet网络的实现。

    参数:
    - block: 基本块的类型，可以是BasicBlock等。
    - layers: 每个阶段包含的基本块数量。
    - input_channels: 输入通道数。
    - num_classes: 分类的类别数。
    - zero_init_residual: 是否将残差块的最后一个批量归一化层的权重初始化为0。
    - groups: 卷积的组数。
    - width_per_group: 每个组的卷积通道数。
    - replace_stride_with_dilation: 使用扩张卷积替换步长为2的卷积。
    - norm_layer: 批量归一化层的类型。

    属性:
    - _norm_layer: 批量归一化层的函数。
    - inplanes: 输入平面数。
    - dilation: 扩张率。
    - groups: 卷积的组数。
    - base_width: 每个组的卷积通道数。
    - conv1: 第一个卷积层。
    - bn1: 第一个批量归一化层。
    - relu: ReLU激活函数。
    - maxpool: 最大池化层。
    - layer1-4: ResNet的四个阶段。
    - avgpool: 平均池化层。
    - fc: 全连接层。
    """

    def __init__(self, block, layers, input_channels=1, num_classes=10, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        构建ResNet的一个阶段。

        参数:
        - block: 基本块的类型。
        - planes: 该阶段的输出通道数。
        - blocks: 该阶段包含的基本块数量。
        - stride: 基本块的步长，默认为1。
        - dilate: 是否使用扩张卷积，默认为False。

        返回:
        - 该阶段的网络层。
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        ResNet的前向传播过程。

        参数:
        - x: 输入的数据。

        返回:
        - 输出的结果。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18_1d(input_channels, num_classes):
    return ResNet1D(BasicBlock, [2, 2, 2, 2], input_channels=input_channels, num_classes=num_classes)


def resnet34_1d(input_channels, num_classes):
    return ResNet1D(BasicBlock, [3, 4, 6, 3], input_channels=input_channels, num_classes=num_classes)


