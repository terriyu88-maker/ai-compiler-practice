"""Stage 4 使用的带 BatchNorm 的小型 CNN。

每个卷积块写成 Conv2d -> BatchNorm2d -> ReLU，正是 Stage 4 的折叠 pass 要识别的模式。
LeNet 本身没有 BN，所以这里单独引入一个模型，让 Conv+BN+ReLU 折叠有真实的落点。
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConvBNReLU(nn.Module):
    """一个卷积块：Conv2d -> BatchNorm2d -> ReLU。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvBNNet(nn.Module):
    """两个 ConvBNReLU 块 + 最大池化 + 全连接，输入 28x28 单通道。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBNReLU(1, 8, kernel_size=3, padding=1)
        self.block2 = ConvBNReLU(8, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        return self.fc(x)


@torch.no_grad()
def randomize_bn_stats(model: nn.Module) -> None:
    """把 BN 的 running 统计与仿射参数设为非平凡值。

    默认的 running_mean=0 / running_var=1 会让 BN 近似恒等，折叠后权重几乎不变，
    验证会失去意义。随机化后，折叠是否算对才真正经受考验。
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            channels = module.num_features
            module.weight.copy_(torch.randn(channels) * 0.2 + 1.0)
            module.bias.copy_(torch.randn(channels) * 0.2)
            module.running_mean.copy_(torch.randn(channels) * 0.5)
            module.running_var.copy_(torch.rand(channels) * 1.5 + 0.5)
