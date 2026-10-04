# Stage 4：手写 fx pass —— Conv+BN+ReLU 融合

本阶段是整个项目的重心：亲手写一个 graph-level 优化 pass，并证明它语义等价。

具体做法是：在推理态下把 `BatchNorm2d` 的逐通道仿射变换折进前置 `Conv2d` 的权重与偏置，使图中 `Conv2d -> BatchNorm2d -> ReLU` 三个节点收敛为一个融合节点。折叠后 BN 从计算图中彻底消失，卷积直接输出等价结果，ReLU 作为尾随逐元素算子保留在融合模块内。

LeNet 没有 BN，所以本阶段单独引入 `ConvBNNet`（见 `src/stage4_pass/model.py`）作为被优化对象。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| 数值对齐 | 最大绝对误差 < 1e-5 | 1.192e-07 |
| 节点数下降 | 融合后节点数减少 | 11 -> 7 |
| BatchNorm 消除 | 图中不再有 BN | 2 -> 0 |
| 延迟有可测量变化 | 给出中位数与加速比 | 5.661ms -> 4.591ms，1.23x |
| 单元测试 | 通过 | 3 passed（全量 9 passed） |

## 2. 折叠公式推导（推理态）

记卷积（含偏置）与 BatchNorm 推理态的逐通道计算为：

```
卷积:  z_c = (W_c * x) + b_c
BN:    y_c = gamma_c * (z_c - mu_c) / sqrt(sigma_c^2 + eps) + beta_c
```

其中 `mu_c`、`sigma_c^2` 是 running_mean、running_var，`gamma_c`、`beta_c` 是 BN 的可学习缩放与偏移。

把 `z_c` 代入 BN，令 `s_c = gamma_c / sqrt(sigma_c^2 + eps)`：

```
y_c = s_c * ((W_c * x) + b_c) - s_c * mu_c + beta_c
    = (s_c * W_c) * x + (s_c * (b_c - mu_c) + beta_c)
```

于是得到等价卷积：

```
W'_c = s_c * W_c
b'_c = s_c * (b_c - mu_c) + beta_c
```

这正是 `fold_conv_bn` 的计算。两点前提：

- 只在**推理态**成立。训练态 BN 用的是当前 batch 的均值方差，并且会更新 running 统计，无法预先折叠。
- BN 的缩放是逐输出通道的，所以 `s_c` 需要 reshape 成 `(C_out, 1, 1, 1)` 才能广播到卷积权重 `(C_out, C_in/groups, kH, kW)` 上。
- 若卷积无偏置，等价于 `b_c = 0`，代码中已处理。

ReLU 的关键性质：`relu(W*x+b)` 不能合并进 `W`、`b`（它非线性），因此 ReLU 只能在同一 kernel 内作为尾随 epilogue 执行，而无法被代数折叠。所以本 pass 的实质是「Conv+BN 代数折叠」+「Conv 与 ReLU 结构合并为一个节点」。

## 3. src/stage4_pass/model.py 逐行说明

代码位置：`src/stage4_pass/model.py`（共 54 行）

### 3.1 文件头与导入（L1-L9）

- L1-L5 文档字符串：说明为什么需要单独的带 BN 模型——LeNet 没有 BN，折叠便无从演示。
- L7 `import torch`：用于 `torch.randn`、`torch.no_grad`。
- L8 `from torch import nn`：模型基类与层。
- L9 `from torch.nn import functional as F`：函数式 ReLU、max_pool2d。

### 3.2 ConvBNReLU 块（L12-L21）

- L12 `class ConvBNReLU(nn.Module):`：把 `Conv2d -> BatchNorm2d -> ReLU` 打包成一个模块，便于在 `ConvBNNet` 里复用。
- L15 `__init__(self, in_channels, out_channels, kernel_size=3, padding=1)`：构造签名，默认 3x3 卷积、padding=1 保持空间尺寸。
- L17 `self.conv = nn.Conv2d(..., bias=True)`：显式 `bias=True`，因为折叠公式要把 BN 的偏移合并进卷积偏置；若这里没偏置，公式中的 `b` 取 0。
- L18 `self.bn = nn.BatchNorm2d(out_channels)`：每个输出通道一组 `gamma/beta/running_mean/running_var`。
- L21 `return F.relu(self.bn(self.conv(x)))`：链式前向。用函数式 `F.relu` 而非 `nn.ReLU` 模块，使 fx 追踪后它成为 `call_function` 节点，与 `conv`/`bn` 的 `call_module` 节点形成可识别的三段模式。

### 3.3 ConvBNNet 主体（L24-L38）

- L27 `__init__(self, num_classes=10)`：默认 10 类。
- L29-L30 `block1 = ConvBNReLU(1, 8, ...)`、`block2 = ConvBNReLU(8, 16, ...)`：两个卷积块，通道 1->8->16。
- L31 `self.fc = nn.Linear(16 * 14 * 14, num_classes)`：卷积块输出 16x14x14（28x28 经两次 padding=1 卷积不改变尺寸，池化后 14x14），展平后接全连接。
- L34-L38 `forward`：依次过两个块、最大池化、展平、全连接。注意池化放在第二个 ReLU 之后，所以两个块内部都是完整的 Conv-BN-ReLU，pass 恰好能匹配两次。

### 3.4 randomize_bn_stats（L41-L54）

- L41 `@torch.no_grad()`：这是一个纯赋值辅助函数，禁用梯度记录。
- L42 `def randomize_bn_stats(model)`：随机化所有 BN 的统计与仿射参数。
- L43-L46 文档字符串：解释动机——默认 `running_mean=0`、`running_var=1` 时 BN 近似恒等，折叠几乎不改变权重，验证会变成「永远通过」。随机化后折叠公式的对错才真正暴露。
- L48-L49 `for module in model.modules(): if isinstance(module, nn.BatchNorm2d)`：遍历所有 BN。
- L50 `channels = module.num_features`：该 BN 的通道数。
- L51-L52：`weight` 取 `1.0 ± 0.2`，`bias` 取 `N(0, 0.2)`。
- L53-L54：`running_mean` 取 `N(0, 0.5)`，`running_var` 取 `0.5 ~ 2.0`（恒正）。

## 4. src/stage4_pass/fusion_pass.py 逐行说明

代码位置：`src/stage4_pass/fusion_pass.py`（共 107 行）

### 4.1 文件头与导入（L1-L17）

- L1-L11 文档字符串：给出折叠公式与「BN 消失、ReLU 保留」的说明。
- L13 `import torch`：`torch.zeros`、`torch.sqrt`、`torch.no_grad`。
- L14 `import torch.fx as fx`：用于 `isinstance(..., fx.Node)` 类型判断。
- L15 `from torch import nn`：`nn.Conv2d`、`nn.BatchNorm2d`。
- L16 `from torch.fx import GraphModule`：类型标注。
- L17 `from torch.nn import functional as F`：识别 `F.relu` 节点。

### 4.2 fold_conv_bn（L20-L42）

- L20 签名：输入已绑定的卷积与 BN 模块，返回新的、等价的卷积。
- L22 `weight = conv.weight.detach()`：取权重并断开梯度，本函数只做推理态数值变换。
- L23 `bias = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)`：统一成张量 `b`，无偏置时用 0。
- L25 `scale = bn.weight / sqrt(bn.running_var + bn.eps)`：即推导中的 `s_c`。注意用 `bn.eps` 而不是硬编码 1e-5，尊重模块自身配置。
- L26 `folded_weight = weight * scale.reshape(-1, 1, 1, 1)`：`scale` 形状 `(C_out,)`，reshape 成 `(C_out,1,1,1)` 后按输出通道广播乘到卷积权重上，即 `W'_c = s_c * W_c`。
- L27 `folded_bias = (bias - bn.running_mean) * scale + bn.bias`：即 `b'_c = s_c*(b_c - mu_c) + beta_c`。
- L29-L38 构造新的 `nn.Conv2d`：除 `bias=True` 外，`in_channels/out_channels/kernel_size/stride/padding/dilation/groups` 全部沿用原卷积，保证形状与感受野不变。分组卷积的 `s_c` 依然逐输出通道，公式不变。
- L39-L41 `with torch.no_grad(): fused.weight.copy_(...); fused.bias.copy_(...)`：把折叠结果写入新卷积。用 `copy_` 而不是直接赋值属性，保持参数张量的身份与设备一致。
- L42 `return fused`。

### 4.3 FusedConvBNReLU（L45-L53）

- L45 `class FusedConvBNReLU(nn.Module):`：融合后的单个模块。
- L48-L50 `__init__`：`self.conv = fold_conv_bn(conv, bn)`，把折叠后的卷积存为子模块。
- L52-L53 `forward`：`return F.relu(self.conv(x))`，把 ReLU 作为 epilogue 接在折叠卷积之后。这样图中原本的 conv/bn/relu 三个节点可以合并为一个 `call_module` 节点。

### 4.4 _match_triples 模式匹配（L56-L77）

这段是 fx 模式的识别逻辑，从 ReLU 节点反向找它的两个祖先：

- L60 `triples = []`：收集结果。
- L61 `for node in gm.graph.nodes:`：按拓扑序遍历所有节点。
- L62 `if node.op != "call_function" or node.target is not F.relu: continue`：锚点必须是函数式 ReLU。用 `is not` 比较函数对象，精确且不依赖名字。
- L64 `bn_node = node.args[0] if node.args else None`：ReLU 的唯一输入，应指向 BN 节点。
- L65-L66 校验 `bn_node` 是 `call_module` 节点（即产自一个模块调用）。
- L67-L68 `isinstance(modules.get(bn_node.target), nn.BatchNorm2d)`：该模块确实是 `BatchNorm2d`。用 `modules.get` 而非下标，避免目标名不存在时抛异常。
- L69-L73 对 conv 重复同样的三步校验：BN 的输入是 `call_module` 且目标是 `nn.Conv2d`。
- L74 `if len(bn_node.users) != 1 or len(conv_node.users) != 1: continue`：安全改写的前提——BN 的输出只被这个 ReLU 消费、conv 的输出只被这个 BN 消费。若存在分叉（同一张量被多处使用），删除节点会破坏其他消费者，必须跳过。
- L76 `triples.append((conv_node, bn_node, node))`：记录三元组。
- L77 `return triples`。

### 4.5 fuse_conv_bn_relu 主改写（L80-L107）

- L82 `modules = dict(gm.named_modules())`：建立限定名到模块对象的映射，用于查 conv/bn。
- L83 `graph = gm.graph`：图对象。
- L84 `triples = _match_triples(gm, modules)`：先一次性匹配，避免边遍历边改写导致迭代器失效。
- L86-L87：遍历三元组，`index` 用于生成唯一模块名。
- L88-L89：取出 conv、bn 模块对象。
- L90 `name = f"fused_conv_bn_relu_{index}"`：融合模块名。
- L91 `gm.add_submodule(name, FusedConvBNReLU(conv, bn))`：把融合模块注册到 GraphModule 上。注意此时只算出折叠权重，尚未改图。
- L93-L94 `with graph.inserting_before(conv_node): fused_node = graph.call_module(name, args=(conv_node.args[0],))`：在 conv 节点之前插入新的 `call_module` 节点。`conv_node.args[0]` 是卷积的输入张量，直接成为融合节点的输入。
- L95 `relu_node.replace_all_uses_with(fused_node)`：把所有消费 ReLU 输出的节点改为消费融合节点，图的下游因此无缝迁移。
- L96-L98 `graph.erase_node(relu_node / bn_node / conv_node)`：按「下游到上游」的顺序删除原节点。此时 relu 已无用户，bn 的唯一用户 relu 已删，conv 的唯一用户 bn 已删，删除安全。
- L100-L102：记录 `conv + bn + relu -> fused_xxx` 的改写说明，供报告输出。
- L104 `gm.delete_all_unused_submodules()`：清掉图中已不再引用的 `block*.conv`、`block*.bn` 子模块。否则 BN 虽不在计算图里，仍会挂载在 GraphModule 上，`BatchNorm2d 模块数 -> 0` 就不成立。
- L105 `graph.lint()`：校验图结构一致（节点参数指向存在的节点、无悬空引用）。这一步能在改写出错时尽早抛出。
- L106 `gm.recompile()`：根据新图重新生成 `forward` 代码，使 `gm.code` 与 `gm(x)` 生效。
- L107 `return rewritten`。

## 5. src/stage4_pass/verify.py 逐行说明

代码位置：`src/stage4_pass/verify.py`（共 91 行）

- L1 文档字符串：声明三项验证目标。
- L3-L8 导入：`argparse`、`Path`、`torch`、`fx`、`nn`。
- L10-L13 导入公共种子工具、计时器与 Stage 4 的 pass、模型。
- L15-L16 `ROOT` / `ARTIFACTS`：仓库根与产物目录。
- L19-L20 `count_modules(model, cls)`：统计某类模块的数量，用于对比 BN 是否消除。
- L24-L28 参数：`--batch-size`、`--iters`、`--report` 输出路径。
- L30 `set_seed()`：固定随机种子。
- L31 `ARTIFACTS.mkdir(parents=True, exist_ok=True)`。
- L34-L36 `log`：同时打印与收集。
- L38 `model = ConvBNNet().eval()`：`.eval()` 是折叠成立的前提，BN 必须走 running 统计。
- L39 `randomize_bn_stats(model)`：让 BN 非平凡。
- L40 `x = torch.randn(args.batch_size, 1, 28, 28)`：构造输入。
- L42-L43 记录折叠前的参考输出 `reference`。
- L45 `gm = fx.symbolic_trace(model)`：把模型捕获为 GraphModule。
- L46 `gm.eval()`：显式把 GraphModule 设为推理态（GraphModule 默认 training=True，必须切换）。
- L47 `before_code = gm.code`：保存改写前的 forward 代码。
- L48 `before_nodes = len(list(gm.graph.nodes))`：改写前节点数。
- L49 `before_bn = count_modules(gm, nn.BatchNorm2d)`：改写前 BN 数。
- L51 `rewritten = fuse_conv_bn_relu(gm)`：执行折叠 pass，`gm` 被就地改写。
- L52-L54：记录改写后的代码、节点数、BN 数。
- L56-L58 计算 `fused_out` 与 `reference` 的最大绝对误差。
- L60-L62 在 `no_grad` 下分别对原始模型与折叠模型计时。
- L64-L82 输出报告：改写说明、BN 数变化、节点数变化、误差、延迟与加速比、折叠后 forward 代码。
- L78 `speedup = eager.median_ms / fused.median_ms`：用中位数计算，`fused.median_ms` 为 0 时回退为 NaN，避免除零。
- L84 `args.report.write_text(...)`：写报告。
- L85-L86：把改写前后的 forward 代码分别落盘，便于对照。
- L90-L91 入口。

## 6. 实测结果

从仓库根目录运行：

```bash
python3 -m src.stage4_pass.verify
```

```
折叠改写:
  block1_conv + block1_bn + relu -> fused_conv_bn_relu_0
  block2_conv + block2_bn + relu_1 -> fused_conv_bn_relu_1

BatchNorm2d 模块数: 2 -> 0
图节点数         : 11 -> 7
最大绝对误差     : 1.192e-07

原始模型 : median=5.661ms min=5.035ms max=17.143ms
折叠模型 : median=4.591ms min=4.333ms max=5.588ms
加速比(median): 1.23x
```

折叠后的 forward 代码：

```python
def forward(self, x):
    fused_conv_bn_relu_0 = self.fused_conv_bn_relu_0(x);  x = None
    fused_conv_bn_relu_1 = self.fused_conv_bn_relu_1(fused_conv_bn_relu_0);  fused_conv_bn_relu_0 = None
    max_pool2d = torch.nn.functional.max_pool2d(fused_conv_bn_relu_1, 2, stride = None, padding = 0, dilation = 1, ceil_mode = False, return_indices = False);  fused_conv_bn_relu_1 = None
    flatten = max_pool2d.flatten(1);  max_pool2d = None
    fc = self.fc(flatten);  flatten = None
    return fc
```

节点数从 11 降到 7，正是两个三元组各减少 2 个节点：

- 改写前：`x` + `block1_conv` + `block1_bn` + `relu` + `block2_conv` + `block2_bn` + `relu_1` + `max_pool2d` + `flatten` + `fc` + `output` = 11。
- 改写后：`x` + `fused_conv_bn_relu_0` + `fused_conv_bn_relu_1` + `max_pool2d` + `flatten` + `fc` + `output` = 7。

误差 1.192e-07 是浮点重结合的正常量级，远小于 1e-5，说明折叠在数学上精确。延迟从中位数 5.661ms 降到 4.591ms（1.23x）；本机 2 核抖动大（原始模型单次最高 17ms），所以与 Stage 5 一样，结论表述为「有稳定但幅度有限的加速」，精确倍数不作为硬指标。

单元测试覆盖三点（`tests/test_stage4_fusion.py`）：

- `test_fold_conv_bn_matches_eval_formula`：随机 conv/bn 上直接核对折叠结果与 `bn(conv(x))` 一致。
- `test_pass_is_numerically_equivalent`：整个 pass 前后输出一致。
- `test_pass_removes_bn_and_shrinks_graph`：BN 归零、节点数恰好减少 4。

## 7. 与 Stage 5 的联系

Stage 5 观察到 Inductor 把 `convolution` 当作 `extern_kernels` 直接调用，自定义 kernel 只覆盖逐元素与归约算子。这解释了为什么本阶段的手写 pass 有独立价值：

- BN 是逐通道仿射，属于「可代数消除」的算子。若不在图层面折叠，后端要么把 BN 当成独立逐元素 kernel（多一趟读写的访存开销），要么依赖后端自己的折叠逻辑。
- 折叠后，卷积已经是「带最终偏置的单次计算」，BN 参数被吸收进权重，图更小、算子更少、访存往返更短。这与 Stage 5 里「融合减少访存」是同一个原理在不同层面的体现：Stage 4 在**图层面**减少算子与数据流，Stage 5 在**kernel 层面**合并逐元素计算。

## 8. 产物清单

- `artifacts/stage4_fusion_report.txt`：完整的验证报告。
- `artifacts/stage4_model_before.py`：折叠前的 GraphModule forward 代码。
- `artifacts/stage4_model_after.py`：折叠后的 GraphModule forward 代码。
