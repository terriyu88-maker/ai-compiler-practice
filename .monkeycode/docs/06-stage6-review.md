# Stage 6：跨抽象层次复盘

Stage 6 不写新代码，目标是把前五个阶段的实测结果收束成一张跨抽象层次的对比表，并整理延伸阅读清单。主交付物是 `artifacts/notes.md`，本文件说明它的结构、每一行结论的证据来源，以及整条链路的核心结论。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| 对比表 | 覆盖 PyTorch eager / fx / Inductor kernel / ONNX 四个已实测层次 | 见 `artifacts/notes.md` 第 1 节 |
| 延伸对照 | 覆盖 TVM Relax / MLIR / PTX 的抽象层次与职责 | 见 `artifacts/notes.md` 第 6 节 |
| 阅读清单 | 给出官方文档与关键论文 | 见 `artifacts/notes.md` 第 7 节 |
| 交付 | `artifacts/notes.md` | 已生成 |

## 2. 交付物结构：如何阅读 `artifacts/notes.md`

- **第 0 节 一句话**：把整条链路压成一行——模型依次变成 eager 程序、带模块语义的图、代数化简后的图、后端循环代码、中立的 ONNX 图。
- **第 1 节 四层已实测抽象总表**：主表，按「产物 / 图的基本单位 / 权重表达 / 类型 shape / 谁生成 / 职责 / BN 处理 / 证据」八个维度对比四个层次。
- **第 2 节 BN 贯穿案例**：用同一个 `BatchNorm2d` 串起五个阶段，说明它在语义层、图捕获层、图变换层、后端调度层、IR 桥接层分别是什么形态。
- **第 3 节 融合的三个层次**：区分「图级代数折叠」「后端 kernel 融合」「导出期常量折叠」。
- **第 4 节 实测数字汇总**：Stage 1-5 的关键指标一览。
- **第 5 节 抽象层次对照结论**：四条可复用的结论。
- **第 6 节 延伸路线对照**：TVM Relax / MLIR / 到 PTX 的 IR 阶梯（未实测，属自学方向）。
- **第 7 节 延伸阅读清单**：官方文档 + 论文，附建议阅读顺序。
- **第 8 节 边界与已知限制**：CPU-only、无 GPU 等诚实说明。

## 3. 主对比表（精简版）

完整表见 `artifacts/notes.md` 第 1 节，这里给精简版：

| 维度 | PyTorch eager | fx GraphModule | Inductor CPU kernel | ONNX graph |
|------|---------------|----------------|---------------------|------------|
| 产物 | Python 函数 | `GraphModule` | C++/OpenMP 源码 | `.onnx` |
| 基本单位 | 算子调用 | fx 节点 | 融合循环 | 标准算子 |
| 权重表达 | `Parameter` | 子模块参数（图中不可见） | 指针/常量 | `initializer` |
| 谁生成 | 用户 `forward` | `symbolic_trace` | `torch.compile` | `torch.onnx.export` |
| 职责 | 正确、可微 | 可改写图 IR | 高效循环 | 跨栈交换 |
| BN | 模块 | 一个节点 | 逐元素 epilogue | 折叠或不折叠 |

## 4. 每一行结论的证据来源

复盘表里的每个数字都不是估计值，都能回溯到具体产物：

| 结论 | 证据产物 |
|------|----------|
| eager 语义与反向正确 | `artifacts/stage1_train_log.txt`、`tests/test_stage1_gradcheck.py` |
| fx module 16 节点、operator 34 节点 | `artifacts/lenet_bn_graph.txt` |
| 折叠误差 9.686e-08、节点 16→12 | `artifacts/stage3_fusion_report.txt` |
| 编译 2.01x、4 个 `cpp_fused_*` kernel | `artifacts/stage4_inductor_log.txt`、`inductor_generated_code.py` |
| ONNX folded 12 节点、nofold 14 节点 | `artifacts/stage5_onnx_report.txt`、`stage5_onnx_graph.txt`、`stage5_onnx_nofold_graph.txt` |

对应逐行文档：`00-stage0-environment.md`、`01-stage1-lenet.md`、`02-stage2-fx.md`、`03-stage3-fusion-pass.md`、`04-stage4-inductor.md`、`05-stage5-onnx.md`。

## 5. 核心结论

1. **越往下，信息越少、越规整**：eager 保留完整 Python 语义，fx 保留模块边界，kernel 只剩循环与访存，ONNX 只保留标准算子与张量类型。
2. **权重位置逐层变化**：子模块参数 → 图中不可见 → 内存指针/常量 → 显式 initializer。「常量在哪儿」决定了常量折叠能不能发生。
3. **优化分层次**：图级优化回答「能不能少算」（Stage 3 折叠 BN），后端优化回答「算得更快、少访存」（Stage 4 融合循环），IR 交换格式回答「怎么跨栈表示」（Stage 5 ONNX）。
4. **同一语义有多种等价表示**：`Conv+BN` 与折叠后的 `Conv`、`F.linear` 与 `Gemm(transB=1)`、`x.flatten(1)` 与 `Flatten(axis=1)`。编译器大量工作是在做「表示转换」并证明「等价」。

## 6. BN 案例：同一算子的三种命运

这是全项目最值得记住的一条线：

| 阶段 | BN 的形态 | 手段 | 结果 |
|------|-----------|------|------|
| Stage 1 | 训练态更新统计 / 推理态仿射 | 手写前向反向 | 语义基准 |
| Stage 2 | 独立 `call_module` 节点 | fx 捕获 | 图上可见 |
| Stage 3 | 被代数折进卷积 | 手写 fx pass | 节点 2→0 |
| Stage 4 | 与 ReLU/MaxPool 融合的 epilogue | Inductor 调度 | 省访存，不折权重 |
| Stage 5 | 导出器自动折叠 / 或保留节点 | ONNX 常量折叠 | 12 或 14 节点 |

## 7. 与后续学习的衔接

延伸路线在 `artifacts/notes.md` 第 6、7 节。可直接复用的资产：

- `artifacts/lenet_bn.onnx` 是 TVM 与 torch-mlir 的入口。
- Stage 3 的折叠公式与「数值对齐 + IR 对比 + 计时」方法对任意后端都适用。
- Stage 4 的 C++ 生成代码是判断别的后端「生成得好不好」的参照物。
- `src/common/timer.py` 可直接用于对比任意后端产物的延迟。

## 8. 产物清单

- `artifacts/notes.md`：本阶段的复盘主交付物。
