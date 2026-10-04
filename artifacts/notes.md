# LeNet-BN 编译器链路复盘笔记（Stage 6）

本笔记汇总 Stage 0-5 的实测结果，给出跨抽象层次的对比表，并把「编译器一步在做什么」压成可复用的结论。全程只有一个模型 LeNet-BN，五个阶段的产物都能互相对照。

## 0. 一句话

一个模型从 PyTorch 代码出发，会依次变成：**可执行的 eager 程序 → 带模块语义的图 → 代数化简后的图 → 后端调度出的循环代码 → 中立的 ONNX 图**；每一层保留的信息、丢失的信息、以及能做的优化都不同。

## 1. 四层已实测抽象的总表

| 维度 | PyTorch eager | fx GraphModule | Inductor CPU kernel | ONNX graph |
|------|---------------|----------------|---------------------|------------|
| 产物 | Python 函数直接执行 | `GraphModule` + `graph.nodes` | 生成的 C++/OpenMP 源码 | `.onnx` protobuf |
| 图的基本单位 | 逐条算子调用 | `call_module` / `call_function` 节点 | 融合后的循环 kernel | 标准算子节点 |
| 权重表达 | 张量（`Parameter`） | 子模块参数，图中不可见 | 传给 kernel 的内存指针/常量 | `initializer`（显式张量） |
| 类型/shape | 运行时确定 | 元数据稀少 | 编译期固化到循环边界 | 输入输出显式 `FLOAT[...]`，中间靠形状推断 |
| 谁生成 | 用户写 `forward` | `torch.fx.symbolic_trace` | `torch.compile` → Inductor | `torch.onnx.export` |
| 主要职责 | 正确、可微 | 提供可改写的图 IR | 生成高效、访存友好的循环 | 跨框架/后端交换模型 |
| 本项目的 BN | `BatchNorm2d` 模块 | 一个 `call_module` 节点 | 逐元素 epilogue，不折进卷积 | 常量折叠时折进 Conv；否则独立 `BatchNormalization` 节点 |
| 证据 | Stage 1 训练/梯度检查 | Stage 2 图 dump | Stage 4 生成代码 | Stage 5 报告 |

补充一行：Stage 3 的「融合后图」介于 fx 与 Inductor 之间——它仍是 fx GraphModule，但两个 `Conv-BN-ReLU` 被折成一个 `call_module` 融合节点。

## 2. 案例贯穿：BN 在不同层次被如何对待

这是全项目最有信息量的一条线，因为同一个 `BatchNorm2d` 在五个层次里有三种命运：

1. **Stage 1（语义层）**：训练态 BN 用 batch 统计并更新滑动均值/方差；推理态退化成逐通道仿射 `y = a·x + c`，其中 `a = gamma/sqrt(var+eps)`、`c = beta - a·mean`。手写反向证明了对这套语义的掌握。
2. **Stage 2（图捕获层）**：`bn1`、`bn2` 是两个独立的 `call_module` 节点，与卷积、ReLU 平级。
3. **Stage 3（图变换层）**：利用推理态 BN 是仿射的事实，把整个三元组代数折叠进卷积权重：`W' = s*W`、`b' = s*(b - mean) + beta`。BN 从图中消失（2 → 0）。
4. **Stage 4（后端调度层）**：Inductor **没有**做这个代数消除。它把 BN 当作逐元素算子，与相邻的 ReLU / MaxPool 融进同一个循环（`cpp_fused__native_batch_norm_legit_no_training_...`），`gamma/beta/mean/var` 作为 kernel 入参。省的是访存，卷积权重没有变。
5. **Stage 5（IR 桥接层）**：ONNX 导出器在 `do_constant_folding=True` 时自动做 Stage 3 那套折叠，BN 节点消失、initializer 减少 44 个元素；关闭折叠则保留独立 `BatchNormalization` 节点。

结论：**代数折叠（减少算子/权重）与调度融合（减少访存）是两个不同层次的优化**。Stage 3 属于前者，Stage 4 属于后者，Stage 5 展示了导出器会选择前者。

## 3. 融合的三个层次

| 层次 | 谁在做 | 手段 | 本项目证据 | 效果 |
|------|--------|------|-----------|------|
| 图级代数折叠 | Stage 3 手写 pass | 把 BN 的缩放/平移算进卷积权重 | BN 2→0，节点 16→12，误差 9.686e-08 | 算子变少，延迟 2.381→2.113ms（1.13x） |
| 后端 kernel 融合 | Stage 4 Inductor | 多个逐元素算子合成一个循环 | 4 个 `cpp_fused_*` kernel | eager 1.883→0.935ms（2.01x） |
| 导出期常量折叠 | Stage 5 ONNX | 推理态 BN 折进 Conv 常量 | 12 节点，initializer 61706（-44） | 图更小，数值等价（4.292e-06） |

## 4. 实测数字汇总

| 阶段 | 关键数字 |
|------|----------|
| Stage 1 训练 | test_acc 98.93%，params 61750 |
| Stage 1 梯度检查 | forward 7.451e-08，loss 4.768e-07，grad 3.353e-08 |
| Stage 2 图捕获 | module 16 节点/15 边；operator 34 节点/33 边；2 条 Conv→BN→ReLU 候选 |
| Stage 3 折叠 | 误差 9.686e-08，节点 16→12，BN 2→0，2.381→2.113ms（1.13x） |
| Stage 4 编译 | 误差 0.0，1.883→0.935ms（2.01x），426 行、4 kernel，首编译 46.5s |
| Stage 5 ONNX | folded 12 节点/err 4.292e-06；nofold 14 节点/err 1.431e-06 |

## 5. 抽象层次对照结论

1. **越往下，信息越少、越规整**：eager 保留完整 Python 语义；fx 保留模块边界；kernel 只剩循环与访存；ONNX 只保留标准算子与张量类型。
2. **权重的位置逐层变化**：子模块参数 → 图中不可见 → 内存指针/常量 → 显式 initializer。理解「常量在哪儿」是理解常量折叠的前提。
3. **优化发生在不同层次、回答不同问题**：图级优化回答「能不能少算」，后端优化回答「算得更快（少访存）」，图交换格式回答「怎么跨栈表示」。
4. **同一语义可以有多种等价表示**：`Conv+BN` 与「折叠后的 Conv」数值等价、`F.linear` 与 `Gemm(transB=1)` 等价、`x.flatten(1)` 与 `Flatten(axis=1)` 等价。编译器工作的大量时间花在「表示转换 + 证明等价」上。

## 6. 延伸路线对照（未实测，属自学）

同一模型可直接复用于 TVM / MLIR，四步主线在其他栈上的对应：

| 主线步骤 | 本项目 | TVM (Relax) | MLIR (torch-mlir) |
|----------|--------|-------------|-------------------|
| 图捕获 | torch.fx / ONNX | 由 ONNX 或 torch.export 导入为 Relax | torch.export / TorchScript 导入 torch dialect |
| 图变换 | 手写 fx pass | Relax pass（BN 折叠、算子融合） | torch-mlir-opt / mlir-opt 重写 |
| 降级 | Inductor 调度 | `LegalizeOps` → TIR | torch → linalg / tosa dialect |
| 代码生成 | Inductor C++ kernel | `tvm.build` 生成 C / LLVM | llvm dialect → 机器码 |

从 MLIR 一路到 GPU 汇编（PTX）的 IR 阶梯：

| 层 | IR | 谁产生 | 关键动作 |
|----|----|--------|----------|
| 前端 | torch.fx / ExportedProgram | torch.export | 捕获前向，权重变常量 |
| 高层 | torch dialect | torch-mlir | 保留 PyTorch 语义 |
| 中层 | linalg-on-tensors | torch-mlir | 结构化算子 + 张量抽象 |
| 并行 | scf.parallel / scf.forall | tiling | 展开循环、显式分块 |
| 异构 | gpu dialect | gpu mapping | block / thread 映射 |
| 目标 | nvvm dialect | convert-gpu-to-nvvm | 贴 GPU ISA |
| 编译 | LLVM IR | gpu-to-llvm | 通用后端 |
| 汇编 | PTX | llc NVPTX | 设备汇编 |
| 二进制 | cubin (SASS) | ptxas | 仅执行需要 |

可直接复用的既有资产：Stage 5 的 `artifacts/lenet_bn.onnx` 是 TVM/torch-mlir 的入口；Stage 3 的折叠公式与「数值对齐 + IR 对比 + 计时」方法原样可用；Stage 4 的 C++ 代码是「好生成代码」的参照；`src/common/timer.py` 可对比任意后端延迟。

## 7. 延伸阅读清单

官方文档：

- torch.fx：https://pytorch.org/docs/stable/fx.html
- torch.compile 与 Inductor：https://pytorch.org/docs/stable/torch.compiler.html
- ONNX IR 规范：https://github.com/onnx/onnx/blob/main/docs/IR.md
- MLIR：https://mlir.llvm.org/
- torch-mlir：https://github.com/llvm/torch-mlir
- Apache TVM：https://tvm.apache.org/docs/

论文与设计文（按主题）：

- TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)
- Relax: Composable Abstractions for End-to-End Dynamic Machine Learning
- MLIR: A Compiler Infrastructure for the End of Moore's Law (CGO 2020)
- Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation (PLDI 2013)
- TorchInductor: a PyTorch-native compiler with define-by-run IR and symbolic shapes（PyTorch dev-discuss 设计文）

阅读顺序建议：先读 torch.fx 与 ONNX IR，把「图」的概念夯实；再读 Inductor 设计文，理解「生成循环」；最后进 MLIR Toy tutorial 与 TVM Relax，进入真正的编译器基础设施。

## 8. 边界与已知限制

- 环境为 CPU-only、2 核：看不到 GPU 调度与 TensorCore，但布局、融合、降级这些核心概念完整可见。
- Inductor 在 CPU 生成 C++/OpenMP 而非 Triton，kernel 形态不同但「融合」观察目标一致。
- TVM / MLIR / PTX 未在本项目实测，第 6 节为路线对照与延伸建议。
- 训练与反向不在推理链路的典型范围内；本项目的训练只为提供可用模型与反向语义基准。
