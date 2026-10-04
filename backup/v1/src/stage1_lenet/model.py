"""Stage 1 的模型定义。

v2 主模型是 LeNetBN：在 LeNet 的两个卷积后各插入一个 BatchNorm2d，形成 Conv-BN-ReLU 块，
贯穿整个项目（Stage 1-5）。

LeNet（无 BN）是 v1 遗留模型，仅保留给尚未迁移的 Stage 2 / Stage 4 使用，
待这两个阶段迁移到 LeNetBN 后删除。
"""

from torch import nn
from torch.nn import functional as F


class LeNetBN(nn.Module):
    """LeNet-BN：2 个 Conv-BN-ReLU 块 + 2 次最大池化 + 3 个全连接，适配 28x28 单通道 MNIST。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class LeNet(nn.Module):
    """v1 遗留模型（无 BN），仅供尚未迁移的 Stage 2 / Stage 4 使用。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
