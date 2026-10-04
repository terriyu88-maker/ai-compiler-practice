# AI 编译器入门实践计划书（单模型主线）

从 PyTorch 计算图出发，走到 TVM / MLIR，建立「图捕获 -> 图变换 -> 降级 -> 代码生成」的完整直觉。

全程只用一个模型 LeNet-BN，避免主线对象切换带来的可比性问题。

## 0. 定位与目标

- 目标：亲手完成一次从模型定义到 IR 观察、再到 graph-level 优化的闭环，理解编译器在其中每一步的职责。
- 终态能力：
  1. 能读、能改 torch.fx 的 GraphModule，手写一个等价性可验证的 pass。
  2. 能读懂 torch.compile / Inductor 生成的算子代码，说清融合发生了什么。
  3. 能区分 PyTorch graph、ONNX graph、Relax、TIR、MLIR dialect 的抽象层次与分工。
- 环境约束：CPU-only，2 核，约 6 GB 可用内存，无 GPU。所有实验按 CPU 规模设计，计时以相对加速比为准。

## 1. 唯一主模型：LeNet-BN

整个项目的模型只有一个：在经典 LeNet 的两个卷积后各插入一个 `BatchNorm2d`，形成 `Conv -> BN -> ReLU` 的标准块。这样同一张模型图可以同时支撑：

- 图捕获（fx）：`Conv`、`BN`、`ReLU` 是三个可识别的节点。
- 图变换（手写 pass）：`Conv+BN+ReLU` 折叠有真实落点，BN 是逐通道仿射，可代数折进卷积权重。
- 降级与代码生成（Inductor）：可观察后端如何处理 BN（是否自行折叠）与逐元素融合。
- IR 桥接（ONNX）：可对比 fx 与 ONNX 在 BN 表达方式上的差异。

模型结构（输入 1x28x28，10 类，逐层列全，含无参数层）：

| 序号 | 层 | 配置 | 输出 shape |
|------|----|------|-----------|
| 1 | Conv2d | in=1, out=6, k=5, padding=2, bias=True | 6x28x28 |
| 2 | BatchNorm2d | 6 | 6x28x28 |
| 3 | ReLU | - | 6x28x28 |
| 4 | MaxPool2d | k=2, stride=2 | 6x14x14 |
| 5 | Conv2d | in=6, out=16, k=5, bias=True | 16x10x10 |
| 6 | BatchNorm2d | 16 | 16x10x10 |
| 7 | ReLU | - | 16x10x10 |
| 8 | MaxPool2d | k=2, stride=2 | 16x5x5 |
| 9 | Flatten | start_dim=1 | 400 |
| 10 | Linear | in=400, out=120, bias=True | 120 |
| 11 | ReLU | - | 120 |
| 12 | Linear | in=120, out=84, bias=True | 84 |
| 13 | ReLU | - | 84 |
| 14 | Linear | in=84, out=10, bias=True | 10 |

可学习参数合计约 6.2 万（conv1 156、bn1 12、conv2 2416、bn2 32、fc1 48120、fc2 10164、fc3 850）。其中两个 `Conv-BN-ReLU` 三元组是 Stage 3 折叠 pass 的目标。

LeNet-BN 的拓扑与经典 LeNet 一致，只是两个卷积后各多一个 BN；Stage 1 训练与手写反向都围绕这个结构展开。

## 2. 主线四步与阶段对应

| 主线步骤 | 对应阶段 | 在这个模型上做什么 |
|----------|----------|--------------------|
| 图捕获 | Stage 2 | `symbolic_trace(LeNet-BN)`，print 节点、边、直方图 |
| 图变换 | Stage 3 | 手写 pass，把 2 个 `Conv-BN-ReLU` 折成 2 个节点并验证等价 |
| 降级 | Stage 4 | `torch.compile`，导出 Inductor 生成的算子代码并分析融合 |
| 代码生成 / IR 桥接 | Stage 5 | 导出 ONNX，与 fx GraphModule 逐层对照 |
| 复盘 | Stage 6 | 一张跨抽象层次的对比表 |

Stage 0 与 Stage 1 是进入这条主线的前置：环境准备与模型本身。

## 3. 阶段划分与验收标准

### Stage 0 环境与骨架
- 内容：安装 torch(cpu)、torchvision、numpy、matplotlib、pytest；建立目录与公共工具；跑一个冒烟测试。
- 交付：`requirements.txt`、`src/common/`、冒烟脚本。
- 验收：`import torch` 成功，一个两层 MLP 前向输出 shape 正确。
- 状态：已完成，无需改动。

### Stage 1 LeNet-BN 定义与前向 / 反向传播
- 内容：
  1. `nn.Module` 版 LeNet-BN（结构见第 1 节）。
  2. 训练脚本：打印每层输出 shape、每层梯度范数、每 epoch 的 loss 与测试准确率。
  3. 手写版：用原始 tensor 手推梯度，与 autograd 逐参数对比（BN 的处理见第 5 节决策）。
- 交付：`src/stage1_lenet/model.py`、`train.py`、`manual_backprop.py`、`artifacts/lenet_bn.pt`。
- 验收：5 epoch 测试准确率 >= 98%；手写梯度与 autograd 最大误差 < 1e-4。
- 学习点：编译器必须保住的前向 / 反向语义；BN 在训练态与推理态的语义差异，为 Stage 3 的折叠前提埋线。
- 状态：**已完成**。test_acc=98.93%，手写前向误差 7.451e-08、梯度误差 3.353e-08（覆盖 14 组参数），参数量 61750，产物 `artifacts/lenet_bn.pt`、`artifacts/stage1_train_log.txt`。

### Stage 2 torch.fx 图捕获
- 内容：`symbolic_trace(LeNet-BN)`；打印节点列表（op / target / args / users）；重建拓扑序依赖边；统计算子直方图；导出 GraphModule 的 Python 代码；标出 `Conv-BN-ReLU` 候选链。
- 交付：`src/stage2_fx/inspect_graph.py`、`artifacts/lenet_bn_graph.txt`。
- 验收：完整打印计算图与依赖边，并标出 2 条 `Conv -> BN -> ReLU` 融合候选。
- 状态：**已完成**。module 级 16 节点/15 边，operator 级 34 节点/33 边，找到 2 条 `conv2d -> batch_norm -> relu -> max_pool2d` 候选链；产物 `artifacts/lenet_bn_graph.txt` 及两份 GraphModule forward 代码。实现要点：为穿透 BN 需临时屏蔽 `_check_input_dim` 的符号追踪控制流（`disable_bn_input_check`）。

### Stage 3 手写 fx pass：Conv+BN+ReLU 融合
- 内容：
  1. 推导把 BN 折叠进 conv 权重与偏置的公式（推理态）：`s = gamma / sqrt(running_var + eps)`，`W' = s*W`，`b' = s*(b - running_mean) + beta`。
  2. 用 fx 遍历定位 `Conv-BN-ReLU` 三元组，替换为单个融合节点。
  3. 数值对齐（最大绝对误差）、节点数前后对比、CPU 延迟前后对比。
- 交付：`src/stage3_pass/fusion_pass.py`、`verify.py`、`artifacts/stage3_fusion_report.txt`。
- 验收：输出误差 < 1e-5，节点数下降，延迟有可测量变化。
- 学习点：编译器工程师的核心动作——写 pass 并证明其语义等价。这是本项目的重心。
- 状态：**已完成**。在 LeNet-BN 上折叠 2 个 `Conv+BN+ReLU` 三元组：误差 9.686e-08，节点 16→12，BN 模块 2→0，延迟 2.381ms→2.113ms（1.13x）；产物 `artifacts/stage3_fusion_report.txt`、`stage3_model_before/after.py`。

### Stage 4 torch.compile / Inductor 观察
- 内容：对 LeNet-BN `torch.compile`；用 `TORCH_COMPILE_DEBUG=1` 导出生成的算子代码；对比 eager 与 compiled 的延迟；记录后端是否自行折叠 BN。
- 交付：`src/stage4_inductor/compile_demo.py`、`artifacts/inductor_generated_code.py`、`artifacts/stage4_inductor_log.txt`。
- 验收：能指出生成的 kernel 把哪些算子融在了一起，并解释融合为什么减少了访存。
- 学习点：把 Stage 3 的图级优化与后端 kernel 级融合对照——同一原理的两个层次。
- 说明：本机 CPU-only，Inductor 生成 C++ / OpenMP kernel 而非 Triton kernel，观察目标一致。
- 状态：**已完成**。对 LeNet-BN 编译：eager 1.883ms → compiled 0.935ms（2.01x），误差 0.0，生成 426 行、4 个 `cpp_fused_*` kernel（BN+ReLU+MaxPool 融合，卷积仍走 extern）；产物 `artifacts/inductor_generated_code.py`、`stage4_inductor_log.txt`。关键观察：默认 CPU 路径未把 BN 折进卷积，而是把 BN 作为逐元素 epilogue 融合，与 Stage 3 的图层代数折叠形成层次对照。

### Stage 5 桥接到编译器 IR（ONNX 图层）
- 内容：导出 ONNX（默认常量折叠与关闭折叠各一份），用 onnx / onnx.reference 打印图结构，与 fx GraphModule 逐层对照，观察两者在算子粒度、常量表达、图边界上的差异。
- 交付：`src/stage5_bridge/export_onnx.py`、`artifacts/lenet_bn.onnx`、`artifacts/lenet_bn_nofold.onnx`、IR dump 与报告。
- 验收：能给出 fx GraphModule 与 ONNX graph 的抽象层次对比结论。
- 学习点：ONNX 导出器在常量折叠开启时自动把推理态 BN 折进卷积，与 Stage 3 手写折叠是同一套代数；关闭折叠后 BN 作为独立节点保留，14 个计算节点与 fx 完全对应。
- 说明：本阶段只做到 ONNX 图层，TVM / MLIR 作为后续自学延伸，不纳入本计划安装范围。
- 状态：**已完成**。opset 17；folded 12 节点（BN 折进 Conv，initializer 61706 = 61750-44）、nofold 14 节点（BN 保留）；数值最大误差 folded 4.292e-06 / nofold 1.431e-06；产物 `artifacts/lenet_bn.onnx`、`lenet_bn_nofold.onnx`、`stage5_onnx_report.txt`、`stage5_onnx_graph.txt`、`stage5_onnx_nofold_graph.txt`。

### Stage 6 复盘
- 内容：一张对比表，覆盖 PyTorch eager / fx GraphModule / Inductor CPU kernel / ONNX graph / TVM Relax / MLIR 的抽象层次与职责；整理延伸阅读清单。
- 交付：`artifacts/notes.md`、`.monkeycode/docs/06-stage6-review.md`。
- 学习点：把五层抽象压进同一张表，看清「信息逐层减少」与「优化发生在不同层次」；用同一个 BN 贯穿案例收束全项目。
- 状态：**已完成**。产出 `artifacts/notes.md`（四层已实测对比表 + BN 贯穿案例 + 融合三层次 + 数字汇总 + TVM/MLIR/PTX 延伸对照 + 阅读清单）与文档 `06-stage6-review.md`。

## 4. 目标目录结构

```
.
├── README.md                 # 本计划书
├── requirements.txt
├── src/
│   ├── common/
│   │   ├── seed.py           # 固定随机种子、设备选择
│   │   └── timer.py          # 基准计时：warmup + 多次取中位数
│   ├── stage1_lenet/
│   │   ├── model.py          # nn.Module 版 LeNet-BN（唯一模型）
│   │   ├── manual_backprop.py# 手写 forward/backward，与 autograd 对比
│   │   └── train.py          # 训练 + 逐层 shape / grad norm 记录
│   ├── stage2_fx/
│   │   └── inspect_graph.py  # symbolic_trace + 图与依赖打印
│   ├── stage3_pass/
│   │   ├── fusion_pass.py    # Conv+BN+ReLU 融合 pass
│   │   └── verify.py         # 数值对齐 + 节点数 + 延迟
│   ├── stage4_inductor/
│   │   └── compile_demo.py   # torch.compile + 导出生成的 kernel
│   └── stage5_bridge/
│       └── export_onnx.py
├── artifacts/                # 权重、图 dump、生成的代码、计时结果
└── tests/                    # 数值正确性测试
```

## 5. 技术选型与风险

- CPU-only：LeNet-BN 用完整 MNIST（约 10 分钟量级）。
- 计时方法：必须 warmup 后多次取中位数，避免 CPU 抖动造成误判；2 核机器上加速比按区间表述。
- Stage 5 只依赖 onnx，安装风险低。
- Inductor 的 C++ 后端依赖 Python 头文件（已安装 `python3.11-dev`）；调试目录显式指向 `/tmp`，避免污染工作区。

### Stage 1 手写反向如何处理 BN（已决策：方案 B）

BN 在训练态的反向涉及 batch 统计的梯度，比纯卷积更复杂。两种选择：

- 方案 A：手写反向覆盖 BN 训练态完整反向（含均值/方差反向的闭式公式），梯度检查在 train mode 进行。
- **方案 B（已采用）**：手写反向在 eval mode 下把 BN 当作固定仿射处理，反向只有 `dx = dy*a`（`a = gamma/sqrt(running_var+eps)`），实现与检查都直观，并复用 Stage 3 的同一套数学。

采用方案 B 的具体安排：

- 训练脚本正常使用 train-mode BN，反向由 autograd 负责。
- 手写反向的梯度检查单独在 eval mode 下运行：BN 冻结为仿射，梯度检查聚焦 conv / pool / fc / relu 的链式法则。
- BN 训练态反向的完整公式写入 Stage 1 文档，作为选读推导，不落地代码。

## 6. 延伸路线：从 ONNX 到 TVM / MLIR

本项目在 Stage 5 停在 ONNX 图层。之后想继续自学 TVM / MLIR，**同一个 LeNet-BN 可以直接作为贯穿各栈的共享示例**。原因：它是固定单输入、静态 shape 的小模型，算子种类覆盖 conv / BN / relu / maxpool / flatten / linear，正好把布局变换、算子融合、常量折叠、BN legalization 这些主题都覆盖到，而且规模小到每层 IR 都能逐行看懂。

四步主线在其他栈上的对应：

| 主线步骤 | 本项目 | TVM (Relax) | MLIR (torch-mlir) |
|----------|--------|-------------|-------------------|
| 图捕获 | torch.fx / ONNX | ONNX 或 torch.export 导入为 Relax | torch.export / TorchScript 导入 torch dialect |
| 图变换 | 手写 fx pass | Relax pass（折叠 BN、算子融合） | torch-mlir-opt / mlir-opt 重写 |
| 降级 | Inductor 调度 | `LegalizeOps` -> TIR | torch -> linalg / tosa dialect |
| 代码生成 | Inductor C++ kernel | `tvm.build` 生成 C / LLVM | llvm dialect -> 机器码 |

可直接复用的既有资产：

- Stage 5 导出的 ONNX 文件是 TVM 与 MLIR 两个前端的天然入口。
- Stage 3 的 BN 折叠公式与「数值对齐 + IR 对比 + 计时」验证方法原样可用。
- Stage 4 导出的 Inductor C++ 代码作为「好的生成代码」参照物。
- `src/common/timer.py` 可直接用于对比任意后端产物的延迟。

建议新增的目录（属于延伸阶段，不纳入本计划交付）：

- `src/extension_tvm/`：ONNX -> Relax -> `tvm.build` -> 数值与延迟对比。
- `src/extension_mlir/`：torch-mlir 导入 -> 打印各 dialect IR -> 降到 LLVM。

诚实的边界：

- CPU-only 环境看不到 GPU 调度与 TensorCore 相关内容，但布局、融合、降级这些核心概念仍然完整可见。
- TVM / MLIR 安装体积大，建议使用独立虚拟环境，且不纳入本计划的安装范围。
- 训练与反向不在 TVM / MLIR 的典型推理链路内，本延伸只做推理。

## 7. 延伸路线细化：从 PyTorch 到 PTX（MLIR GPU 路径）

这是第 6 节中 MLIR 方向的详细展开。目标是全程在无 GPU 的机器上走到 PTX；只有执行 kernel 才需要真实 GPU。

### 7.1 IR 阶梯

| 层 | IR | 谁产生 | 关键动作 |
|----|----|--------|----------|
| 前端 | torch.fx / ExportedProgram | torch.export | 捕获前向图，权重变常量 |
| 高层 | torch dialect | torch-mlir | 保留 PyTorch 语义 |
| 中层 | linalg-on-tensors | torch-mlir | 结构化算子 + 张量抽象 |
| 并行 | scf.parallel / scf.forall | tiling | 展开循环、显式分块 |
| 异构 | gpu dialect | gpu mapping | block / thread 映射、kernel outline |
| 目标 | nvvm dialect | convert-gpu-to-nvvm | 贴 GPU ISA |
| 编译 | LLVM IR | gpu-to-llvm / mlir-translate | 通用后端 |
| 汇编 | PTX | llc NVPTX | 设备汇编 |
| 二进制 | cubin (SASS) | ptxas | 仅执行需要 |

### 7.2 阶段 0：环境准备

```bash
# 独立 venv，避免污染主 torch 环境
python3 -m venv /workspace/envs/mlir/.venv
/workspace/envs/mlir/.venv/bin/python -m pip install -U pip setuptools wheel

# 验证 NVPTX 后端存在（不需要 GPU）
llc --version
```

- `torch` + `torch-mlir`：前端导入，版本强绑定，按 snapshot 要求装 torch。
- `torch-mlir-opt` / `mlir-opt` / `mlir-translate`：随 torch-mlir release tarball 提供，用于拼 pass 流水线。
- `llc`（LLVM 的 NVPTX 后端）：LLVM IR -> PTX。
- `ptxas`（CUDA toolkit）：PTX -> cubin，可选。

必做选择：确定 target chip（`sm_75` / `sm_80` / `sm_89` …）。无设备可探测，必须手动指定，它决定 PTX 版本与可用指令。下文以 `sm_80` 为例。

### 7.3 阶段 1：图捕获（PyTorch -> ExportedProgram）

```python
# 先 eval，再把权重当常量导出
model.eval()
ep = torch.export.export(model, (example_input,))
torch.export.save(ep, "artifacts/lenet_bn.pt2")
```

静态 shape、单输入最省事；train-mode BN 带状态，必须先 `.eval()`，让 BN 退化为常量仿射。对应主线的「图捕获」。

### 7.4 阶段 2：导入 MLIR（ExportedProgram -> torch dialect -> linalg）

```bash
# 用 torch-mlir 的 Python API 导入为 torch dialect
# 具体函数名以所装 snapshot 为准，常见为 torch_mlir.fx.export_and_import

# torch dialect -> linalg-on-tensors
torch-mlir-opt model.torch.mlir \
  --torch-decompose-complex-ops \
  --convert-torch-to-linalg \
  -o model.linalg.mlir
```

产物：`model.torch.mlir`（保留 PyTorch 语义）、`model.linalg.mlir`（结构化算子）。

### 7.5 阶段 3：高层优化（linalg 内部）

```bash
# 逐元素融合 + 规范化 + 常量折叠
mlir-opt model.linalg.mlir \
  --canonicalize \
  --cse \
  --linalg-fuse-elementwise-ops \
  --linalg-fold-unit-extent-dims \
  -o model.linalg.opt.mlir
```

这一层与 Stage 3 手写 pass 是同类工作：BN 折叠、`conv+relu` 合并、死代码消除。可与 fx 版本的节点数做对照。

### 7.6 阶段 4：降级到 gpu dialect（核心调度）

整条链里唯一需要人工设计的一步：选分块尺寸、决定哪些维度映射到 block / thread。

```bash
# linalg -> 并行循环 -> 映射到 GPU -> outline 成 kernel
mlir-opt model.linalg.opt.mlir \
  --convert-linalg-to-parallel-loops \
  --gpu-map-parallel-loops \
  --convert-parallel-loops-to-gpu \
  --gpu-kernel-outlining \
  -o model.gpu.mlir
```

推进顺序：先用逐元素算子（relu / add）跑通，再加池化，最后做卷积。卷积 tiling 是难点，需要 `scf.forall` + tile 或 transform dialect。产物中可见 `gpu.launch`、outlined `gpu.module` 与 `gpu.func`。

### 7.7 阶段 5：降级到 NVVM

```bash
# 贴 sm_80 目标并把 gpu 降级到 nvvm/llvm
mlir-opt model.gpu.mlir \
  --nvvm-attach-target="chip=sm_80" \
  --convert-gpu-to-nvvm \
  --gpu-to-llvm \
  -o model.nvvm.mlir
```

### 7.8 阶段 6：代码生成（-> LLVM IR -> PTX）

```bash
# 转成 LLVM IR
mlir-translate --mlir-to-llvmir model.nvvm.mlir -o model.ll

# LLVM NVPTX 后端直接产 PTX，无需 CUDA toolkit
llc -march=nvptx64 -mcpu=sm_80 model.ll -o model.ptx
```

可选最后一步：

```bash
# PTX -> cubin，需要 CUDA toolkit，仍然不需要 GPU
ptxas -arch=sm_80 model.ptx -o model.cubin

# MLIR 也可一步到位产 cubin（内部调 ptxas）
mlir-opt model.nvvm.mlir --gpu-module-to-binary -o model.binary.mlir
```

### 7.9 阶段 7：验证

无 GPU 时可做：

- 读 PTX：每个融合 kernel 是一个 `.visible .entry <name>(...)`，对照 Stage 3 的融合节点与 Stage 4 的 C++ kernel，三者应能一一对上。
- 查线程配置：`.reqntid` 或 `mov.u32 %ntid`，对应阶段 4 的 block / thread 绑定。
- 查访存：`ld.global` / `st.global` / `.shared`，看清哪些中间张量被融合掉、哪些走共享内存。
- `ptxas -v` 报告寄存器数与 shared memory 用量，判断是否超限。

有 GPU 时可做：

- 用 `mlir-cuda-runner` 或 `cuda-python` 加载 cubin 执行，做数值对齐与延迟对比，复用 `src/common/timer.py` 的「warmup + 中位数」方法。

### 7.10 目录与里程碑

```
src/extension_mlir/
├── export_program.py   # torch.export -> .pt2
├── to_mlir.py          # 导入 torch dialect 并降到 linalg
├── lower_gpu.sh        # linalg -> gpu -> nvvm -> ptx 的 pass 流水线
└── inspect_ptx.py      # 解析 .visible .entry / 线程配置 / 访存
```

- M1：单个逐元素算子走通 PyTorch -> PTX，产出可读 PTX。
- M2：接入 LeNet-BN，在 linalg 层验证 BN 折叠与逐元素融合。
- M3：完成卷积 tiling，产出多 kernel PTX，与 Stage 3/4 的融合结论对照。
- M4：有 GPU 时补执行验证与延迟对比。

### 7.11 已知坑

- **版本三角**：torch-mlir wheel、torch、LLVM commit 必须匹配，snapshot 窗口很窄。
- **必须指定 sm_XX**：漏了会在 NVVM 降级时失败。
- **静态 shape**：动态 shape 需要额外处理，LeNet-BN 天然静态。
- **只能推理**：训练与反向不在这条链路内，BN 必须 eval。
- **先易后难**：卷积 tiling 是唯一真正费时的部分，务必从逐元素算子起步。
- **资源约束**：本沙箱可用内存只剩约 190Mi 且无任何 LLVM 工具，这条延伸建议在资源更充裕的机器上执行。

### 7.12 与 Triton 路径的对照

同样到 PTX，另一条路是 `torch.compile` 在 GPU 上生成 Triton kernel，再由 Triton 降到 PTX。那条路自动化程度高、几乎零手工调度；MLIR 这条路手工拼 pass、手选 tiling，适合理解「降级与调度」。两条路的 PTX 结构可以放在一起对比，是本项目很好的收尾素材。

## 8. 里程碑

- M1（Stage 0-1）：LeNet-BN 基线 + 手写反向传播。
- M2（Stage 2-3）：图捕获 + 自写融合 pass —— 核心。
- M3（Stage 4-5）：降级与 IR 桥接。
- M4（Stage 6）：复盘。

## 9. 协作方式

每个 stage 按「写代码 -> 运行 -> 记录 artifact -> 对照验收标准 -> 汇报 -> 确认后进入下一阶段」推进，阶段结束做一次 git commit。

## 10. 文档

逐行代码说明存放在 `.monkeycode/docs/`，索引见 `.monkeycode/docs/README.md`。文档与代码同步更新：做到哪个阶段，就更新到哪个阶段。

`.monkeycode/docs/` 下 `00`-`06` 覆盖全部 Stage，文档与代码同步。
