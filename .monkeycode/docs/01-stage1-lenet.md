# Stage 1：LeNet-BN 模型与前向 / 反向传播

本阶段是整条主线的前置：先把唯一主模型 LeNet-BN 定义出来并训练到合格精度，再用不依赖 autograd 的手写前向 / 反向与 autograd 逐参数对比，确认我们真正掌握了这套计算的语义。后面 Stage 2 捕获它的图、Stage 3 改写它的图、Stage 4 编译它，全部基于同一个模型。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| 训练精度 | 5 epoch 测试准确率 >= 98% | 98.93% |
| 手写前向一致 | 最大绝对误差 < 1e-4 | 7.451e-08 |
| 手写梯度一致 | 最大绝对误差 < 1e-4 | 3.353e-08（覆盖全部 14 组参数） |
| 单元测试 | 通过 | 1 passed（全量 8 passed） |

## 2. 唯一主模型：LeNet-BN

LeNet-BN 在两个卷积后各插入一个 `BatchNorm2d`，其余拓扑与经典 LeNet 一致。逐层结构（含无参数层）：

| 序号 | 层 | 配置 | 输出 shape |
|------|----|------|-----------|
| 1 | Conv2d | in=1, out=6, k=5, padding=2 | 6x28x28 |
| 2 | BatchNorm2d | 6 | 6x28x28 |
| 3 | ReLU | - | 6x28x28 |
| 4 | MaxPool2d | k=2 | 6x14x14 |
| 5 | Conv2d | in=6, out=16, k=5 | 16x10x10 |
| 6 | BatchNorm2d | 16 | 16x10x10 |
| 7 | ReLU | - | 16x10x10 |
| 8 | MaxPool2d | k=2 | 16x5x5 |
| 9 | Flatten | start_dim=1 | 400 |
| 10 | Linear | 400 -> 120 | 120 |
| 11 | ReLU | - | 120 |
| 12 | Linear | 120 -> 84 | 84 |
| 13 | ReLU | - | 84 |
| 14 | Linear | 84 -> 10 | 10 |

可学习参数 61750 个（两个 BN 分别带来 bn1 的 12 个与 bn2 的 32 个参数）。打印各层 shape 时可见 `bn1`、`bn2` 与卷积输出同形，因为它们都是逐通道仿射。

## 3. BatchNorm 的两种语义与方案 B

BatchNorm 在训练态与推理态是两套行为：

- **训练态**：用当前 batch 在 `(N, H, W)` 上的均值与方差归一化，并用滑动平均更新 `running_mean` / `running_var`；
- **推理态**：用固定的 `running_mean` / `running_var` 归一化，batch 统计不再参与，也不更新。

推理态因此退化为逐通道仿射：

```
x_hat = (x - mean) / sqrt(var + eps)
y     = gamma * x_hat + beta = a * x + c
a     = gamma / sqrt(var + eps)
c     = beta - a * mean
```

方案 B 的口径：梯度检查在 eval 模式下进行，`running_mean` / `running_var` 当作固定常量（它们是 buffer，autograd 本就不为其产生梯度）。此时 BN 对 `x` 是仿射，`dx`、`dgamma`、`dbeta` 都有闭式解，因此梯度检查可以覆盖包括 BN 在内的全部可学习参数。训练脚本本身仍用训练态 BN，反向交给 autograd。

### 3.1 一个有意思的现象：卷积偏置的梯度约为 0

训练日志里 `conv1.bias` 与 `conv2.bias` 的首 batch 梯度 L2 范数是 `0.000001` 和 `0.000000`，而其他参数都在 `1e-2` 量级。原因：训练态 BN 会减掉 batch 均值，而卷积偏置给该通道所有元素加上同一个常数，这个常数被均值减法完全抵消，于是输出与偏置无关，梯度在数学上为零（残余的 1e-6 是浮点噪声）。

这条现象正好为上一条线埋线：既然 BN 后的卷积偏置是冗余的，Stage 3 把 BN 折进卷积时，折叠后的偏置 `c` 会由 BN 的 `beta`、`mean` 与卷积自己的偏置共同决定，而不再需要单独依赖卷积偏置。

### 3.2 训练态 BN 反向（选读推导，不落地代码）

设对通道 `c`，在 `(N, H, W)` 上的样本数为 `m`，`g = dL/dy`，`x_hat` 如上有：

```
dgamma = sum(g * x_hat)
dbeta  = sum(g)
dx     = gamma / sqrt(var + eps) * ( g - mean(g) - x_hat * mean(g * x_hat) )
```

其中 `mean` 在 `(N, H, W)` 上按通道进行。滑动统计的更新为：

```
running_mean = (1 - momentum) * running_mean + momentum * mean(x)
running_var  = (1 - momentum) * running_var  + momentum * var(x)
```

推导里最易错的是 `dx` 的三项：第一项来自 `x_hat`，第二、三项来自 `mean` 与 `var` 对 `x` 的依赖。方案 A 要把这段写进手写反向，方案 B 只保留推导，实测中选择方案 B。

## 4. src/stage1_lenet/model.py 逐行说明

代码位置：`src/stage1_lenet/model.py`（共 32 行）

### 4.1 文件头（L1-L5）

- L1-L5 文档字符串：说明本文件的主模型是 `LeNetBN`，贯穿 Stage 1-4。

### 4.2 LeNetBN（L11-L32）

- L11 `class LeNetBN(nn.Module):`：唯一主模型。
- L14 `def __init__(self, num_classes: int = 10):`：构造签名。
- L16 `self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)`：输入单通道，输出 6 通道，5x5 核，padding=2 保持 28x28。默认 `bias=True`。
- L17 `self.bn1 = nn.BatchNorm2d(6)`：6 通道的 BN，携带 `gamma`、`beta` 两个可学习参数与 `running_mean`、`running_var` 两个 buffer。
- L18 `self.conv2 = nn.Conv2d(6, 16, kernel_size=5)`：第二层卷积，无 padding；输入是池化后的 14x14，输出 10x10。
- L19 `self.bn2 = nn.BatchNorm2d(16)`：16 通道 BN。
- L20-L22 三个全连接：`16*5*5=400 -> 120 -> 84 -> 10`。展平维度的正确性由 L28 的第二次池化保证。
- L24-L32 `forward`：
  - L25 `F.relu(self.bn1(self.conv1(x)))`：`Conv -> BN -> ReLU` 块，对应主线的融合目标。
  - L26 `F.max_pool2d(x, 2)`：2x2 最大池化，28x28 -> 14x14。
  - L27-L28 第二个 `Conv -> BN -> ReLU` 块与池化，10x10 -> 5x5。
  - L29 `x = x.flatten(1)`：按第 1 维展平成 400。
  - L30-L31 两个带 ReLU 的全连接。
  - L32 `return self.fc3(x)`：输出 logits，不在分类层加 ReLU。

## 5. src/stage1_lenet/manual_backprop.py 逐行说明

代码位置：`src/stage1_lenet/manual_backprop.py`（共 245 行）

### 5.1 文件头与导入（L1-L18）

- L1-L10 文档字符串：声明不使用 autograd，并说明 BN 按方案 B 处理。
- L12-L13 `import torch` / `import torch.nn.functional as F`。
- L15-L16：导入种子工具与 `LeNetBN`。
- L18 `TOL = 1e-4`：前向与梯度的验收阈值。

### 5.2 卷积（L21-L47）

- L21 `conv2d_forward(x, weight, bias, padding=0, stride=1)`：im2col 卷积。
- L23 `cols = F.unfold(...)`：把输入按滑动窗口展开成 `(N, C*k*k, L)` 的矩阵。这里复用 PyTorch 的 unfold 只为得到索引展开，矩阵乘法与梯度全部自己算。
- L24 `w_mat = weight.reshape(weight.shape[0], -1)`：卷积核摊平成 `(C_out, C_in*k*k)`。
- L25-L26 由输入尺寸、padding、stride、核尺寸算输出高宽。
- L27-L29 `(w_mat @ cols + bias.view(1, -1, 1))`：一次矩阵乘完成所有窗口的卷积，再加偏置并 reshape 回 `(N, C_out, H_out, W_out)`。
- L30 返回输出与 cache（后续反向要用到 `cols` 与形状信息）。
- L33-L42 `conv2d_backward`：
  - L36-L37 `grad_w`：`g @ cols^T` 再对 batch 求和，得到权重梯度。
  - L38 `grad_b = g.sum(dim=(0, 2))`：偏置梯度是该通道所有输出位置的梯度之和。
  - L39 `grad_cols = w_mat.t() @ g`：把梯度映回展开空间。
  - L40-L46 `F.fold`：把展开空间的梯度散回输入张量，是 `unfold` 的逆操作。

### 5.3 BatchNorm（L50-L65）

- L50 `batchnorm2d_forward(x, weight, bias, running_mean, running_var, eps)`：eval 模式 BN。
- L52 `inv_std = torch.rsqrt(running_var + eps)`：`1/sqrt(var+eps)`，即推导里的 `1/sqrt(...)`。
- L53 `mean = running_mean.reshape(1, -1, 1, 1)`：改成按通道广播的形状。
- L54 `scale = (weight * inv_std).reshape(1, -1, 1, 1)`：即 `a = gamma/sqrt(var+eps)`。
- L55 `shift = bias - scale * mean`：即 `c = beta - a*mean`。
- L56 `x_hat = (x - mean) * inv_std`：归一化后的值，`dgamma` 需要它。
- L57 `return scale * x + shift, (x_hat, scale)`：输出与 cache。
- L60-L65 `batchnorm2d_backward`：
  - L62 `grad_gamma = (grad_out * x_hat).sum(dim=(0, 2, 3))`：`dgamma = sum(g * x_hat)`。
  - L63 `grad_beta = grad_out.sum(dim=(0, 2, 3))`：`dbeta = sum(g)`。
  - L64 `grad_x = grad_out * scale`：`dx = dy * a`，因为 `running_mean/var` 固定，仿射的斜率就是 `a`。
  - L65 返回三项，对应 `bn.weight`、`bn.bias` 与上游输入。

### 5.4 ReLU、池化、线性、交叉熵（L68-L122）

- L68-L73 ReLU：前向 `clamp(min=0)`，反向按 `x > 0` 掩码，`>0` 保证 0 处不传梯度。
- L76-L83 最大池化前向：把 `(N,C,H,W)` reshape 成 `(N,C,oh,kh,ow,kw)` 再 permute，让核内元素排在最后一维，`max(dim=-1)` 同时拿到最大值与索引。
- L86-L94 最大池化反向：用 `scatter_` 把上游梯度按前向记录的索引放回 `k*k` 的对应位置，其余置零。
- L97-L99 线性前向：`F.linear` 并缓存输入。
- L101-L106 线性反向：`grad_w = g^T @ x`、`grad_b = sum(g)`、`grad_x = g @ weight`。
- L109-L114 交叉熵前向：先减去每行最大值（数值稳定），再取 log-softmax，最后按 target 取值求平均。
- L117-L122 交叉熵反向：`softmax - one_hot` 再除以 batch 大小。

### 5.5 ManualLeNetBN（L125-L184）

- L126-L128 `__init__`：持有被检查的模型与 cache 字典。
- L130-L153 `forward`：按 `conv1 -> bn1 -> relu1 -> pool1 -> conv2 -> bn2 -> relu2 -> pool2 -> flatten -> fc1 -> relu3 -> fc2 -> relu4 -> fc3` 顺序执行，每一步都把中间量写入 cache。所有参数与 buffer 都用 `.detach()` 取值，确保这段计算完全脱离 autograd。
- L155-L184 `backward`：严格按前向的逆序回传。
  - L159-L168 全连接与 ReLU 的反向。
  - L170 把梯度 reshape 回池化输出形状。
  - L171-L176 `pool2 -> relu2 -> bn2 -> conv2` 的反向，注意 BN 反向夹在 ReLU 与卷积之间，顺序与图中节点一致。
  - L178-L183 `pool1 -> relu1 -> bn1 -> conv1` 的反向。
  - 返回的 `grads` 键名与 `named_parameters()` 完全对齐，才能逐参数比较。

### 5.6 随机化 BN、梯度检查、入口（L187-L245）

- L187-L195 `randomize_bn_buffers`：把两处 BN 的 `running_mean/running_var/weight/bias` 设为非平凡值。默认 `mean=0, var=1` 会让仿射近似恒等，梯度检查失去意义。
- L198-L199 `run_gradcheck`：设种子并构造 `LeNetBN().eval()`。`.eval()` 是方案 B 的前提。
- L200-L203 构造输入与标签。
- L205-L208：autograd 路径——前向、交叉熵、`backward()`，再用 `named_parameters()` 收集参考梯度。
- L210-L215：手写路径——`ManualLeNetBN` 前向得到 logits，手写交叉熵前向得到 loss 与 cache，再手写反向。
- L217-L221：逐参数计算与 autograd 的最大绝对误差，并取全体最大值。
- L223-L227：打印前向、loss、逐参数与全体的误差。
- L229-L234：返回结构化报告，便于测试断言。
- L237-L241 `main`：断言前向与梯度误差均小于 `TOL`。

### 5.7 实测梯度检查输出

```
forward  max abs err = 7.451e-08
loss     abs err     = 4.768e-07
  grad conv1.weight   max abs err = 1.583e-08
  grad conv1.bias     max abs err = 6.054e-09
  grad bn1.weight     max abs err = 2.794e-09
  grad bn1.bias       max abs err = 9.604e-10
  grad conv2.weight   max abs err = 1.863e-08
  grad conv2.bias     max abs err = 3.260e-09
  grad bn2.weight     max abs err = 6.170e-09
  grad bn2.bias       max abs err = 3.725e-09
  grad fc1.weight     max abs err = 1.490e-08
  grad fc1.bias       max abs err = 2.095e-09
  grad fc2.weight     max abs err = 1.397e-08
  grad fc2.bias       max abs err = 5.588e-09
  grad fc3.weight     max abs err = 3.353e-08
  grad fc3.bias       max abs err = 1.490e-08
grad     max abs err = 3.353e-08
PASS: forward / grad 与 autograd 的最大误差均 < 0.0001
```

全部 14 组参数误差都在 `1e-8` 量级，说明手写的卷积、BN、池化、线性、ReLU 与交叉熵反向与 autograd 语义一致。

## 6. src/stage1_lenet/train.py 逐行说明

代码位置：`src/stage1_lenet/train.py`（共 134 行）

### 6.1 导入与常量（L1-L15）

- L1-L8：`argparse`、`time`、`Path`、`torch`、`nn`、`DataLoader`、`datasets/transforms`。
- L10-L11：导入种子/设备工具与 `LeNetBN`。
- L13-L15 `ROOT` / `ARTIFACTS` / `DATA_DIR`：仓库根、产物目录、数据缓存目录。

### 6.2 数据与工具（L18-L67）

- L18-L27 `build_loaders`：用 MNIST 官方归一化常数 `(0.1307,)`、`(0.3081,)` 做变换；训练集 shuffle，测试集固定顺序；`num_workers=0` 在 2 核环境下避免额外进程开销。
- L30-L45 `capture_layer_shapes`：给每个子模块挂 forward hook，前向一遍记录各层输出 shape，用完移除 hook。因此日志里能列出 `conv1/bn1/conv2/bn2/fc1/fc2/fc3` 全部有输出的层。
- L48-L53 `grad_norms`：收集每个参数当前梯度的 L2 范数。
- L56-L67 `evaluate`：`no_grad` 下累计正确数与损失，返回准确率与平均损失。

### 6.3 主流程（L70-L130）

- L71-L75 参数：`--epochs`、`--batch-size`、`--lr`。
- L77-L79：设种子、取设备、建产物目录。
- L83-L85 `log`：同时打印与收集。
- L87-L89：建 loader，构造 `LeNetBN().to(device)`，用 Adam。
- L91-L92：打印设备、轮数、batch、lr 与参数量。
- L94-L97：取一个 batch，打印逐层输出 shape。
- L99-L122 训练循环：
  - L100 `model.train()`：切到训练态，BN 使用 batch 统计并更新滑动统计。
  - L104-L114 每个 batch：清梯度、前向、交叉熵、反向；在首个 batch 打印逐层梯度 L2 范数；更新参数。
  - L117-L122 每个 epoch 结束做一次评估并打印 train/test loss、准确率与耗时。
- L124-L129：保存 `artifacts/lenet_bn.pt`，复评并断言准确率 >= 98%，打印 PASS。
- L130：把整段日志写入 `artifacts/stage1_train_log.txt`。

### 6.4 实测训练日志

```
params=61750
逐层输出 shape:
  conv1    -> (64, 6, 28, 28)
  bn1      -> (64, 6, 28, 28)
  conv2    -> (64, 16, 10, 10)
  bn2      -> (64, 16, 10, 10)
  fc1      -> (64, 120)
  fc2      -> (64, 84)
  fc3      -> (64, 10)
首个 batch 的逐层梯度 L2 范数:
  conv1.weight   L2=0.117070
  conv1.bias     L2=0.000001
  bn1.weight     L2=0.010327
  bn1.bias       L2=0.007534
  conv2.weight   L2=0.283927
  conv2.bias     L2=0.000000
  bn2.weight     L2=0.017603
  bn2.bias       L2=0.011259
  fc1.weight     L2=0.487466
  fc1.bias       L2=0.022696
  fc2.weight     L2=0.252672
  fc2.bias       L2=0.050207
  fc3.weight     L2=0.190135
  fc3.bias       L2=0.086034
epoch 1/5 train_loss=0.1760 test_loss=0.0617 test_acc=97.94% time=10.9s
epoch 2/5 train_loss=0.0573 test_loss=0.0450 test_acc=98.56% time=10.0s
epoch 3/5 train_loss=0.0414 test_loss=0.0332 test_acc=98.96% time=10.0s
epoch 4/5 train_loss=0.0344 test_loss=0.0425 test_acc=98.68% time=10.0s
epoch 5/5 train_loss=0.0286 test_loss=0.0376 test_acc=98.93% time=10.3s
saved /workspace/artifacts/lenet_bn.pt
PASS: test_acc=98.93% (>= 98%)
```

两个卷积偏置的梯度接近 0，原因见 3.1；这是本阶段最有信息量的一条观测量。

## 7. 单元测试

`tests/test_stage1_gradcheck.py` 调用 `run_gradcheck(batch_size=8)`，断言前向、loss、梯度误差均 < 1e-4。它会一并检查 BN 的 `weight/bias` 梯度，因此方案 B 的公式一旦写错就会被抓住。

## 8. 与后续阶段的联系

- Stage 2 将 `symbolic_trace(LeNetBN)`，图中的 `conv -> bn -> relu` 三段正是本阶段的 L28 与 L30 两行代码展开的结果。
- Stage 3 的 BN 折叠公式 `W' = s*W`、`b' = s*(b - mean) + beta`，与 L50-L57 的 `scale`、`shift` 是完全同一个 `a`、`c`。本阶段在训练态观察到「卷积偏置冗余」，Stage 3 则在推理态把 BN 折进卷积，两者互为印证。
- Stage 4 编译 LeNet-BN 时，Inductor 会自行处理 BN；本阶段的手写反向则提供了「这些算子的语义到底是什么」的基准。

## 9. 产物清单

- `artifacts/lenet_bn.pt`：训练好的 LeNet-BN 权重。
- `artifacts/stage1_train_log.txt`：完整训练日志（含逐层 shape 与梯度范数）。
