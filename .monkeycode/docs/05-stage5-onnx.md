# Stage 5：ONNX 导出与 fx 对照（IR 桥接）

本阶段把 LeNet-BN 从 PyTorch 的计算图导出成工业界通用的 ONNX 图，再把 fx GraphModule 与 ONNX graph 放在一起逐层对照。目的是看清「同一个模型、不同的中间表示（IR）」在算子粒度、常量表达、图边界上的差异，并观察 ONNX 导出器在导出时自动做的优化。本阶段只做到 ONNX 图层，TVM / MLIR 是后续自学延伸（见根 README 第 7、8 节）。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| 导出成功 | 生成合法 ONNX，`onnx.checker` 通过 | 通过 |
| 数值对齐 | ONNX 输出与 PyTorch 最大绝对误差 < 1e-4 | 折叠版 4.292e-06、不折叠版 1.431e-06 |
| 图对比 | 给出 fx 与 ONNX 的算子级对照与结论 | 见第 5 节 |
| 单元测试 | 通过 | 2 passed（全量 8 passed） |

## 2. 背景：为什么需要 ONNX

PyTorch 训练出来的模型，其「图」是框架内部的表示。部署到不同硬件或不同推理引擎时，需要一个中立、跨框架的格式来承载模型，ONNX（Open Neural Network Exchange）就是这个角色。ONNX 定义了一套标准算子和一个图结构：节点是算子，权重作为 initializer 单独存放，输入输出有显式类型与 shape。

它和 fx 最大的差别在「抽象层次」：

- fx 是 PyTorch 自己的图，`call_module` 节点直接指向 `nn.Conv2d`/`nn.BatchNorm2d` 这样的子模块，权重藏在子模块参数里，图里看不到；
- ONNX 是后端友好的图，算子被替换成 `Conv`/`BatchNormalization`/`Gemm` 这类标准算子，权重被显式提成 initializer，类型、shape、opset 版本都写清楚。

导出过程同时是一次「图变换」：导出器会顺手做一些优化，本阶段最重要的观察就发生在这里。

## 3. src/stage5_bridge/export_onnx.py 逐行说明

代码位置：`src/stage5_bridge/export_onnx.py`（共 258 行）

### 3.1 文件头与导入（L1-L38）

- L1-L9 文档字符串：说明本阶段做四件事——导出、打印 ONNX 图、与 fx 对照、数值对齐。
- L11-L14 `argparse`、`warnings`、`Counter`、`Path`；导入 `warnings` 是为了在生成紧凑 IR 文本时屏蔽 `printable_graph` 的弃用警告。
- L16-L21：`numpy`、`onnx`、`torch`、`torch.fx`；从 `onnx` 取 `TensorProto`，从 `onnx.reference` 取 `ReferenceEvaluator`（ONNX 官方的参考解释器，用 numpy 逐算子执行，可在无 onnxruntime 的情况下跑模型）。
- L23-L24：导入种子工具与主模型 `LeNetBN`。
- L26-L29 `ROOT` / `ARTIFACTS` / `WEIGHTS` / `NOFOLD_PATH`：仓库根、产物目录、训练权重、第二个（不折叠）导出的路径。
- L31-L37 `FX_TO_ONNX`：fx 里的算子/模块到 ONNX 算子的映射表。`Linear` 对应 `Gemm`（ONNX 用 Gemm 表达全连接），`max_pool2d` 对应 `MaxPool`，其余一一对应。
- L38 `METHOD_TO_ONNX`：`x.flatten(1)` 在 fx 里是 `call_method`，对应 ONNX 的 `Flatten`。

### 3.2 加载模型（L41-L49）

- L41-L49 `load_model`：先检查权重文件是否存在，不存在则提示先跑 Stage 1 训练；然后 `LeNetBN()` + `load_state_dict` 载入训练好的参数，`.eval()` 切到推理态。
- `.eval()` 是导出 ONNX 的前提：只有推理态 BN 才退化为常量仿射，导出器才能把它折进卷积。

### 3.3 导出（L52-L63）

- L52-L63 `export_to_onnx(model, out_path, opset, folding)`：
  - L53 `dummy = torch.randn(1, 1, 28, 28)`：导出需要一个示例输入来确定 shape，trace 时不会真正算它的值。
  - L54-L63 `torch.onnx.export(...)`：
    - `input_names=["input"]` / `output_names=["logits"]`：给图输入输出起名，便于后续用名字喂数据。
    - `opset_version=17`：指定 ONNX 算子集版本。
    - `dynamo=False`：使用传统 TorchScript 导出器。这样只依赖 `onnx` 包，不需要额外安装 `onnxscript`。
    - `do_constant_folding=folding`：是否开启常量折叠。这个开关是本阶段第二个变体的关键。

### 3.4 紧凑 IR 文本与类型/属性（L66-L93）

- L66-L70 `compact_graph_text`：调用 `onnx.helper.printable_graph` 生成只含结构的 ONNX 文本（列出 initializer 名与 shape，但不内联权重数值）。`warnings.catch_warnings` 用来屏蔽该函数在新版 onnx 中的弃用提示。这比 `onnx.printer.to_text` 更小、更适合作为 IR dump。
- L73-L84 `tensor_type_str`：把 ONNX 的 `ValueInfoProto` 打印成 `FLOAT[1, 1, 28, 28]` 这种可读格式，依次读出每个维度的 `dim_value`（静态）或 `dim_param`（符号），并用 `TensorProto.DataType.Name` 把整型类型码转回 `FLOAT`。
- L87-L93 `attr_to_str`：把节点属性转成字符串。`onnx.helper.get_attribute_value` 会按属性类型解出 Python 值；对 `bytes`（字符串属性）和 `bytes 列表`（字符串数组属性）做解码，其余直接 `str`。

### 3.5 打印 ONNX 图（L96-L134）

- L96-L100：打印 IR 版本、producer 与 opset。
- L103-L109：打印图输入输出及其类型/shape。
- L111-L117：遍历 `graph.initializer`，打印每个权重的名字、shape、dtype，并累加元素总数。这里能直接看到「BN 的参数是否还存在于图中」。
- L119-L123：`Counter(node.op_type for node in g.node)` 得算子直方图。
- L126-L133：按拓扑序打印每个节点：`op_type`、`name`、输入、输出，以及所有属性（卷积的 `kernel_shape`/`pads`、Gemm 的 `transB` 等）。

### 3.6 fx 节点序列（L137-L156）

- L137-L156 `fx_sequence`：对模型做 `fx.symbolic_trace` 得到 module 级图，把每个节点映射成「预期 ONNX 算子」：
  - `call_module`：按子模块的类名（`module + name`）查 `FX_TO_ONNX`；
  - `call_function`：按函数全名（如 `torch.nn.functional.relu`）查表；
  - `call_method`：按方法名查 `METHOD_TO_ONNX`（`flatten -> Flatten`）；
  - `placeholder`/`output`：保持原样，作为图边界标记。

### 3.7 抽象层次对照（L159-L180）

- L159-L169：先打印 fx module 级的整条节点序列与其对应 ONNX 算子。可看到 `conv1 -> Conv`、`bn1 -> BatchNormalization`、`fc1 -> Gemm` 等。
- L171-L172：统计 fx 的算子数时排除 `placeholder`/`output`，只留计算节点，才能和 ONNX 节点数直接比。
- L173-L180：打印 `fx / onnx folded / onnx nofold` 三列对照表，以及三者节点总数。

### 3.8 数值对齐（L183-L195）

- L183-L195 `verify_numerics`：
  - L184-L187：固定种子造 `x:(4,1,28,28)`，用 PyTorch 前向得到参考输出。
  - L189 `onnx.checker.check_model`：先校验 ONNX 模型合法。
  - L190-L191 `ReferenceEvaluator(onnx_model).run(None, {"input": x.numpy()})`：用 ONNX 参考解释器执行，输入用输入名 `input`（对应导出的 `input_names`）。
  - L193-L195：算两者最大绝对误差并返回。

### 3.9 主流程（L198-L258）

- L198-L203 参数：`--out`（折叠版路径）、`--opset`、`--report`。
- L205-L212：设种子、建目录、加载模型，导出折叠版（`folding=True`）与不折叠版（`folding=False`），再 `onnx.load` 回来。
- L220-L223：打印折叠版的图结构。
- L225-L228：做 fx 与两种 ONNX 变体的对照。
- L230-L236：对两个变体分别做数值对齐。
- L239-L244：输出结论。
- L246-L252：把报告、两份紧凑 ONNX 文本图分别写盘：
  - `artifacts/stage5_onnx_report.txt`：本脚本的分析报告。
  - `artifacts/stage5_onnx_graph.txt`：折叠版由 `compact_graph_text` 生成的标准 IR 文本。
  - `artifacts/stage5_onnx_nofold_graph.txt`：不折叠版。
- L253-L254：断言两个变体误差都 < 1e-4。

## 4. 实测：ONNX 图结构（折叠版）

```
IR version=8  producer=pytorch 2.14.0
opset: domain=ai.onnx version=17

图输入:  input    FLOAT[1, 1, 28, 28]
图输出:  logits   FLOAT[1, 10]

initializer（权重/常量，共 10 个）:
  fc1.weight  shape=[120, 400]
  fc1.bias    shape=[120]
  fc2.weight  shape=[84, 120]
  fc2.bias    shape=[84]
  fc3.weight  shape=[10, 84]
  fc3.bias    shape=[10]
  onnx::Conv_36  shape=[6, 1, 5, 5]
  onnx::Conv_37  shape=[6]
  onnx::Conv_39  shape=[16, 6, 5, 5]
  onnx::Conv_40  shape=[16]
initializer 元素总数 = 61706

节点直方图:
  Conv 2   Flatten 1   Gemm 3   MaxPool 2   Relu 4
```

节点列表（折叠版，共 12 个）：

```
#0  Conv      input, onnx::Conv_36, onnx::Conv_37 -> ...
#1  Relu      ...
#2  MaxPool   ...
#3  Conv      /MaxPool_output_0, onnx::Conv_39, onnx::Conv_40 -> ...
#4  Relu
#5  MaxPool
#6  Flatten   axis=1
#7  Gemm      transB=1
#8  Relu
#9  Gemm      transB=1
#10 Relu
#11 Gemm      -> logits
```

关键点：

- 图中**没有 `BatchNormalization` 节点**。两个卷积的输入是 `onnx::Conv_36/37`、`onnx::Conv_39/40`，这些是按 BN 折叠公式算好的新权重与新偏置，作为 initializer 直接塞进了 `Conv`。
- `initializer 元素总数 = 61706`，正好是原模型 61750 减去 BN 的 44 个参数（bn1 的 12 个 + bn2 的 32 个）。BN 的参数被「吸收」进了卷积权重。
- `Flatten` 的 `axis=1` 对应 `x.flatten(1)`；`Gemm` 的 `transB=1` 表示权重矩阵在计算时转置，这是 ONNX 表达全连接的标准写法。

## 5. 实测：fx 与两种 ONNX 变体的对照

fx module 级节点序列：

```
x            -> placeholder
conv1        -> Conv
bn1          -> BatchNormalization
relu         -> Relu
max_pool2d   -> MaxPool
conv2        -> Conv
bn2          -> BatchNormalization
relu_1       -> Relu
max_pool2d_1 -> MaxPool
flatten      -> Flatten
fc1          -> Gemm
relu_2       -> Relu
fc2          -> Gemm
relu_3       -> Relu
fc3          -> Gemm
output       -> output
```

算子数量对照：

| 算子 | fx | ONNX 折叠 | ONNX 不折叠 |
|------|----|-----------|-------------|
| BatchNormalization | 2 | **0** | 2 |
| Conv | 2 | 2 | 2 |
| Flatten | 1 | 1 | 1 |
| Gemm | 3 | 3 | 3 |
| MaxPool | 2 | 2 | 2 |
| Relu | 4 | 4 | 4 |
| **计算节点总数** | **14** | **12** | **14** |

结论：

1. fx 保留 `Conv`/`BN`/`Linear` 为 `call_module` 叶子节点，权重藏在子模块里；ONNX 把权重显式提成 initializer，运算节点换成后端标准算子。
2. `do_constant_folding=True` 时，导出器自动把推理态 BN 折进卷积权重与偏置：`BatchNormalization` 节点消失，节点数 14 -> 12。这与 Stage 3 手写的折叠是**同一套代数**（同一个 `s = gamma / sqrt(var+eps)`），区别只在于这里是导出器自动完成。
3. 关闭常量折叠后，BN 作为独立的 `BatchNormalization` 节点保留，14 个计算节点与 fx 完全一一对应。

## 6. 实测：数值对齐

```
折叠版:    shape torch=(4, 10) onnx=(4, 10)  max abs err = 4.292e-06
不折叠版:  shape torch=(4, 10) onnx=(4, 10)  max abs err = 1.431e-06
```

两个变体都与 PyTorch 前向在 1e-6 量级一致，说明折叠在数值上确实等价（误差来自 BN 折叠后浮点乘加的舍入顺序变化）。

## 7. 单元测试

`tests/test_stage5_onnx.py` 用随机化 BN 的全新 `LeNetBN().eval()` 独立验证（不依赖训练权重）：

- `test_constant_folding_folds_bn_into_conv`：导出折叠版与不折叠版，断言折叠版 `BatchNormalization` 数量为 0、节点数 12，不折叠版为 2、14，且两者 `Conv` 都是 2。
- `test_onnx_matches_pytorch`：两个变体都通过 `onnx.checker`，且 `ReferenceEvaluator` 的输出与 PyTorch 最大误差 < 1e-4。

## 8. 与前后阶段的联系

- 与 Stage 2 对照：Stage 2 的 fx 图是「捕获」，本阶段把同一张图导出成 ONNX，可以看到导出器如何把 `call_module` 拆成标准算子、把权重提成 initializer。
- 与 Stage 3 对照：Stage 3 手写 pass 把 `Conv+BN+ReLU` 折成单节点；本阶段发现 ONNX 导出器在 `do_constant_folding=True` 时自动做同样的 BN 折叠。手写 pass 让你理解「折叠的公式和等价性验证」，ONNX 让你看到「工业格式默认已经这么干」。
- 与 Stage 4 对照：Inductor 在 CPU 路径**没有**做 BN 代数列消除，只把 BN 当逐元素 epilogue 融合；ONNX 导出器则**做了**代数折叠。同一模型的三个层次——手写图变换（Stage 3）、后端运行时融合（Stage 4）、导出期常量折叠（Stage 5）——对 BN 的处理各不相同，这正是「分层编译器」值得观察的地方。
- 后续延伸：本阶段产出的 `artifacts/lenet_bn.onnx` 是 TVM（导入为 Relax）与 torch-mlir（导入 torch dialect）的天然入口。

## 9. 产物清单

- `artifacts/lenet_bn.onnx`：默认常量折叠导出的 ONNX 模型（12 节点，BN 已折进卷积）。
- `artifacts/lenet_bn_nofold.onnx`：关闭常量折叠导出的 ONNX 模型（14 节点，保留 BN）。
- `artifacts/stage5_onnx_report.txt`：本脚本完整分析报告。
- `artifacts/stage5_onnx_graph.txt`：折叠版的 ONNX 标准 IR 文本。
- `artifacts/stage5_onnx_nofold_graph.txt`：不折叠版的 ONNX 标准 IR 文本。
