# Stage 2 分享文档：把模型变成一张能看懂、能改动的图

> 面向第一次接触 AI 编译器的同学。本文件讲清 Stage 2 为什么做、怎么做、做出了什么，以及这篇结果对后面阶段的铺垫。逐行代码细节见 `02-stage2-fx.md`，本文件只讲主线。

## 0. 一句话

Stage 2 把 PyTorch 模型「翻译成一张结构化的计算图」。这一步是 AI 编译器的**最前端**：后面所有优化（折叠、融合、导出）都从这张图开始。

一个类比：写代码时，编译器先把你写的文本解析成语法树（AST），后续的分析和优化都在 AST 上做。AI 编译器同样如此，它的「AST」就是计算图。Stage 1 的模型还是一段 Python `forward` 代码，Stage 2 之后它变成了一张图。

---

## 1. 为什么做

PyTorch 模型默认是一段 Python 代码：`forward` 里顺序写着 `self.conv1(x)`、`self.bn1(...)`、`F.relu(...)`。这段代码能跑，但对编译器有个根本问题——**它是「过程」，不是「数据」**。编译器没法程序化地遍历它、搜索其中的模式、把某几个操作替换成另一个操作。

要优化，先得把「过程」变成「数据」。图捕获做的正是这件事：把「按顺序执行的每一步」抽出来，变成节点（做了什么）和边（数据从哪流到哪）。有了这张图，编译器才能问出这些问题：

- 图里一共有多少个算子？都是什么类型？
- 哪几个算子首尾相连，构成一个可以合并的「形状」？
- 权重/常量在图的什么位置？

Stage 2 具体干两件事：

1. **看清模型**：把 LeNet-BN 捕获成图，打印节点、边、算子类型，让模型的内部结构第一次「可视化」。
2. **找到优化落点**：在图上标出 `Conv → BN → ReLU` 这种可以融合的形状，为 Stage 3 的手写折叠 pass 提供靶子。

---

## 2. 怎么做的

### 2.1 图捕获的基本原理

核心工具是 `torch.fx.symbolic_trace`（简写 `fx`）。它的做法很巧：**让模型对着「假张量」空跑一遍**。

- 假张量叫 Proxy，它不是真正的数值，而是一个「占位符」。
- 模型照常执行 `forward`，每做一次运算（卷积、BN、ReLU……），fx 就记录一个**节点**；数据从哪个节点流到哪个节点，记录成一条**边**。
- 跑完之后，模型变成一个 `GraphModule`——功能和原来一样，但内部是结构化的图数据。

几个要点概念（更细的表见 `02-stage2-fx.md` 第 2 节）：

| 概念 | 含义 |
|------|------|
| `Node` | 图上一个节点，带 `op`（类型）、`name`、`target`、`args`（输入）、`users`（下游）|
| `Graph` | 节点的有序集合，遍历 `graph.nodes` 得到的是拓扑序 |
| `GraphModule` | 由图生成的 `nn.Module`，`gm.code` 能看到等价的 Python `forward` |
| `op` 类型 | `placeholder`（输入）、`get_attr`（取权重）、`call_module`（调子模块）、`call_function`（调函数）、`call_method`（调方法）、`output`（输出）|

### 2.2 一粗一细两张图是怎么来的

关键在一个开关：`Tracer.is_leaf_module`，它决定「遇到一个子模块时，是记一个节点就跳过，还是走进去继续追」。

**粗粒度（模块级）**——用默认 Tracer（`inspect_graph.py:169`）：

```python
module_gm = fx.symbolic_trace(model)
```

默认规则对 `torch.nn` 下的模块返回「是叶子」，于是 `self.conv1(x)` 只记成一个 `call_module` 节点（target=`conv1`），不进入 `Conv2d.forward` 内部。`bn1`、`fc1` 同理。得到 **16 个节点、15 条边**。

**细粒度（算子级）**——用自定义 Tracer（`inspect_graph.py:49-53`, `:170`）：

```python
class FunctionalTracer(fx.Tracer):
    def is_leaf_module(self, module, qualname):
        return False   # 对所有模块都说「不是叶子」

expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))
```

改成 False 后，追到 `self.conv1(x)` 时会**真的执行** `Conv2d.forward`：权重读取 `self.conv1.weight` 被记成 `get_attr` 节点，内部运算 `F.conv2d(...)` 被记成 `call_function` 节点。得到 **34 个节点、33 条边**。

同样的展开对两类模块的效果：

| 模块级节点 | 算子级展开 | 新增 |
|------------|-----------|------|
| `call_module conv1` | `get_attr conv1.weight` + `get_attr conv1.bias` + `call_function torch.conv2d` | +2 |
| `call_module bn1` | `get_attr bn1.{weight,bias,running_mean,running_var}` + `call_function F.batch_norm` | +4 |
| `call_module fc1` | `get_attr fc1.{weight,bias}` + `call_function torch._C._nn.linear` | +2 |

七个带参数模块合计展开出 18 个 `get_attr`，正好是 34 − 16 = 18。

**顺手踩的一个坑**：`BatchNorm.forward` 第一行要做维度检查（`if input.dim() != 4 ...`），这是 Python 控制流，Proxy 假张量参与不了，一拆开黑盒就报 `TraceError`。处理办法是在追踪期间把这个检查临时换成空操作（`disable_bn_input_check`，`inspect_graph.py:30-46`），追完还原。

### 2.3 怎么标出「值得优化的形状」

这一步是**项目手写的代码**，fx 只给图，不给判断。函数是 `find_fusion_candidates()`（`inspect_graph.py:99-127`），核心是一个顺着连线走的模式匹配：

```python
for node in gm.graph.nodes:                       # 扫描所有节点
    if node.target is not F.relu:                 # ① 先找到 ReLU
        continue
    norm = node.args[0]                           # ② 看它的输入是不是 batch_norm
    if short_name(norm.target) not in BN_NAMES:
        continue
    conv = norm.args[0]                           # ③ 再往前是不是 conv2d
    if short_name(conv.target) != "conv2d":
        continue
    pool = <在 node.users 里找池化>                # ④ 后面接不接 MaxPool
    candidates.append(" -> ".join([...]))
```

判断依据只有两条：**算子的名字**（`"relu"`、`"batch_norm"`、`"conv2d"`）和**连线的位置**（一律取 `args[0]`）。

结果：在算子级图上找到 2 条候选链（`lenet_bn_graph.txt:294-296`）：

```
conv2d -> batch_norm -> relu -> max_pool2d
conv2d_1 -> batch_norm_1 -> relu_1 -> max_pool2d_1
```

值得一提：同一个函数在模块级图上跑，输出是「(无：该粒度下看不到内部算子)」（`lenet_bn_graph.txt:101-102`）。因为模块级图里 `conv1`/`bn1` 是 `call_module`，看不到 `conv2d`/`batch_norm` 这些算子名。这正说明**同一个模型、不同粒度，能看到的东西不同**。

---

## 3. 做的结果

### 3.1 产物清单

| 产物 | 内容 |
|------|------|
| `src/stage2_fx/inspect_graph.py` | 完成整件事的脚本 |
| `src/common/graph_draw.py` | 可选的 Graphviz 出图工具（见下方「看图」补充） |
| `artifacts/lenet_bn_graph.txt`（301 行）| 完整图 dump：节点列表、依赖边、算子直方图、候选链 |
| `artifacts/lenet_bn_graphmodule_module_level.py` | 模块级图的等价 `forward` 代码 |
| `artifacts/lenet_bn_graphmodule_operator_level.py` | 算子级图的等价 `forward` 代码 |

补充：脚本支持 `--draw-svg`（默认关闭），会额外导出 `artifacts/lenet_bn_module_level.svg` 与 `artifacts/lenet_bn_operator_level.svg`，把一粗一细两张图直接画出来，最直观地展示「模块是黑盒方框、算子级全部展开」的差别。该功能依赖 `pydot` 与系统 `graphviz`，缺失时自动跳过，不影响主流程。

关键数字：模块级 **16 节点 / 15 边**，算子级 **34 节点 / 33 边**，找到 **2 条** `Conv→BN→ReLU→MaxPool` 候选链。

### 3.2 模块级图怎么看

最直观的入口是 `gm.code`，也就是 `lenet_bn_graphmodule_module_level.py`：

```python
def forward(self, x):
    conv1 = self.conv1(x);  x = None
    bn1 = self.bn1(conv1);  conv1 = None
    relu = torch.nn.functional.relu(bn1, inplace=False);  bn1 = None
    max_pool2d = torch.nn.functional.max_pool2d(relu, 2, ...);  relu = None
    conv2 = self.conv2(max_pool2d);  ...
    ...
    return fc3
```

**新人读法**：

- **每行大致对应一个节点**。`self.conv1(x)` 是一个 `call_module` 节点，`F.relu(...)` 是一个 `call_function` 节点。
- **从上到下就是数据流**。`conv1` 的输出喂给 `bn1`，`bn1` 的输出喂给 `relu`，顺着变量名就能看出连线。
- **看到的是「层」的视角**。`self.conv1`、`self.bn1` 直接就是 `nn.Conv2d`、`nn.BatchNorm2d` 子模块对象，模型的结构一目了然，读起来最像你写的原代码。
- `; x = None` 这种是 fx 插入的「变量用完置空」，方便内存复用，新手可以先忽略。

### 3.3 算子级图怎么看

入口是 `lenet_bn_graphmodule_operator_level.py`，开头几行长这样：

```python
def forward(self, x):
    conv1_weight = self.conv1.weight          # get_attr 节点
    conv1_bias = self.conv1.bias              # get_attr 节点
    conv2d = torch.conv2d(x, conv1_weight, conv1_bias, (1,1), (2,2), (1,1), 1)  # call_function
    bn1_weight = self.bn1.weight
    bn1_bias = self.bn1.bias
    bn1_running_mean = self.bn1.running_mean
    bn1_running_var = self.bn1.running_var
    batch_norm = torch.nn.functional.batch_norm(conv2d, bn1_running_mean, bn1_running_var,
                                                weight=bn1_weight, bias=bn1_bias, training=False, ...)
    ...
```

**新人读法**：

- **黑盒被拆开了**。原来模块级的一行 `conv1 = self.conv1(x)`，在这里变成三行：先取权重、取偏置，再做 `torch.conv2d`。
- **`self.conv1.weight` 这种就是 `get_attr` 节点**，代表「从模块上取一个属性」，它把权重变成了图上看得见的一个点。
- **`torch.conv2d(...)` / `F.batch_norm(...)` 是 `call_function` 节点**，代表真正的运算。
- 这里看到的是「算子」的视角：没有「一层卷积」的概念，只有「取权重 + 做卷积」这些原子操作。

### 3.4 给新人的看图三步法

1. **先读 `gm.code`**（两份 `GraphModule_*.py`）。这是最快的入口，代码形态直接对应图的结构。
2. **再查 `lenet_bn_graph.txt`**。里面有算子直方图（各类型节点有多少）、完整依赖边列表、候选链。想确认「谁连着谁」就查它。
3. **盯住两样东西**：节点的 `op` 类型（是 `call_module` 还是 `call_function`/`get_attr`），以及节点的 `users`（它被谁消费）。这两样拼起来就是图的全貌。

记住两张图各自的擅长：

- **模块级**保留「层」的高级语义，读起来像原模型，适合谈结构。
- **算子级**暴露内部算子和权重，适合做模式匹配和改写。

---

## 4. 总结与预告

### 总结

- Stage 2 做的事，本质是**把模型从「代码」变成「图」**，这是编译器能开工的前提。
- 用同一个模型、只换一个 Tracer 开关，就得到**一粗一细两张图**：模块级（16 节点，看层）和算子级（34 节点，看算子）。
- 「值得优化的形状」是**手写的模式匹配**找出来的：沿 `relu ← batch_norm ← conv2d` 的连线走，找到 2 条融合候选链。
- 顺带暴露了一个真实工程坑：拆开 BN 黑盒时会撞上维度检查的 Python 控制流，需要临时屏蔽。

### 预告后面

- **Stage 3（折叠 pass）**：拿这里的候选链当靶子，在模块级图上写一个 pass，把 `Conv+BN+ReLU` 折成单个融合节点——用 `s = gamma/sqrt(var+eps)` 把 BN 的缩放平移算进卷积权重，然后证明数值等价。这是整个项目的重心（写 pass + 证明等价）。
- **Stage 4（Inductor）**：交给 `torch.compile` 自动编译，看后端把哪些算子融进同一个 kernel、为什么省访存。对照「手写图层折叠」和「后端自动融合」两个层次。
- **Stage 5（ONNX）**：把同一个模型导出成 ONNX 图，对照 fx graph 和 ONNX graph，观察导出器如何把权重提成显式 initializer，以及常量折叠开关对 BN 的影响。

一句话把这条线串起来：**Stage 2 把模型变成图 → Stage 3 在图上手写一次优化并证明等价 → Stage 4/5 看工业系统在同一件事上怎么做**。Stage 2 是这条链的起点。
