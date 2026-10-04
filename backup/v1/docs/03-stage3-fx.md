# Stage 3：torch.fx 图捕获与依赖分析

本阶段把 LeNet 转换成 GraphModule，打印节点、算子、依赖边与算子直方图，并对两种图粒度做对比。这是整个项目里最接近「AI 编译器输入端」的一步：编译器拿到的就是这种计算图。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| 打印完整计算图与依赖边 | 节点、args、users、边齐全 | 已生成 `artifacts/lenet_graph.txt` |
| 标出候选融合模式 | 至少找出 Conv -> ReLU 链 | 找到 2 条 Conv -> ReLU -> MaxPool |
| 单元测试 | 通过 | 2 passed |

## 2. torch.fx 的核心概念

| 概念 | 含义 |
|------|------|
| `Node` | 图中的一个节点，有 `op`、`name`、`target`、`args`、`users` 等属性 |
| `Graph` | 节点的有序集合，迭代 `graph.nodes` 得到的就是拓扑序 |
| `GraphModule` | 由 `Graph` 生成的 `nn.Module` 子类，`gm.code` 可看到等价 Python 前向代码 |
| `op` | 节点类型，常见有 `placeholder`、`get_attr`、`call_function`、`call_method`、`call_module`、`output` |
| `get_attr` | 从模块上取属性（权重、偏置等），是常量来源 |
| `Tracer.is_leaf_module` | 决定哪些 `nn.Module` 不再展开、直接作为一个 `call_module` 节点。默认对 `torch.nn` 模块返回 True。 |

## 3. src/stage3_fx/inspect_graph.py 逐行说明

代码位置：`src/stage3_fx/inspect_graph.py`

### 3.1 文件头与导入（L1-L21）

- L1-L6 文档字符串：说明脚本职责，以及要对比的两种粒度。
- L8 `import argparse`：解析命令行参数。
- L9 `from collections import Counter`：统计各类节点数量。
- L10 `from pathlib import Path`：路径操作。
- L11 `from typing import Any`：类型标注。
- L13 `import torch.fx as fx`：导入 fx 模块。
- L15-L16：导入公共种子工具与 LeNet。
- L18-L19：仓库根目录与产物目录。
- L21 `POOL_NAMES = {...}`：池化算子的名字集合，用于识别池化节点。

### 3.2 自定义 Tracer（L24-L28）

- L24 `class FunctionalTracer(fx.Tracer):`：继承默认 Tracer。
- L25 文档字符串：说明用途。
- L27 `def is_leaf_module(self, module, qualname: str) -> bool:`：重写该判定方法。`module` 是当前模块，`qualname` 是限定名。
- L28 `return False`：对所有模块都返回「不是叶子」，于是 `nn.Conv2d`、`nn.Linear` 会被穿透，其内部调用 `F.conv2d`、`F.linear` 直接进入图中，权重变成 `get_attr` 节点。这就是 operator 级图的来源。

### 3.3 辅助格式化（L31-L42）

- L31-L32 `short_name(target)`：取可调用对象的名字。`torch.nn.functional.relu` 得到 `"relu"`；对非函数目标退化为 `str(target)`。
- L35 `def fmt(value):`：把节点参数格式化成可读字符串。
- L36-L37 `if isinstance(value, fx.Node): return value.name`：参数若是节点，输出节点名，例如 `conv2d`。
- L38-L39 元组/列表：递归格式化，例如 `(max_pool2d, conv2_weight, conv2_bias, (1, 1), ...)`。
- L40-L41 字典：递归格式化。
- L42 `return repr(value)`：其余标量（整数、元组中的步长等）用 `repr`。

### 3.4 汇总 summarize（L45-L67）

- L45 `def summarize(gm, log) -> tuple[list, Counter]:`：输出节点、边、直方图，并返回边与直方图。
- L46 `nodes = list(gm.graph.nodes)`：取出拓扑序节点列表。`graph.nodes` 是双向链表，转成 list 便于索引。
- L47 `edges = [(producer.name, node.name) for node in nodes for producer in node.all_input_nodes]`：依赖边。`node.all_input_nodes` 给出该节点直接消费的所有生产者节点，包括 `args` 与 `kwargs` 中的节点。
- L48 `histogram = Counter(node.op for node in nodes)`：统计各 `op` 出现次数。
- L50 `log(f"节点数={len(nodes)} 依赖边数={len(edges)}")`：概览。
- L51-L53：按 `op` 名字排序打印直方图。`{op:<16}` 左对齐补齐到 16 字符。
- L55-L61 节点列表：
  - L56 `for index, node in enumerate(nodes):`：带上序号，序号即拓扑序位置。
  - L57 `target = short_name(node.target) if node.op != "placeholder" else str(node.target)`：`placeholder` 的 target 是参数名字符串，直接用；其余取短名。
  - L58 `users = [user.name for user in node.users]`：`node.users` 是所有消费该节点的节点。
  - L59-L61：分三行打印节点行、`args`、`users`。把 args 与 users 单独成行，避免行过长。
- L63-L65 `依赖边`：逐条打印 `producer -> consumer`。
- L67 `return edges, histogram`。

### 3.5 融合候选查找 find_fusion_candidates（L70-L93）

- L70 `def find_fusion_candidates(gm) -> list[str]:`：返回形如 `conv2d -> relu -> max_pool2d` 的候选链。
- L71 文档字符串。
- L72 `candidates = []`。
- L73 `for node in gm.graph.nodes:`：遍历所有节点。
- L74 `if node.op != "call_function" or short_name(node.target) != "relu": continue`：只从 ReLU 节点出发。融合模式通常以激活函数结尾。
- L76 `producer = node.args[0] if node.args else None`：ReLU 的第一个参数就是它的输入。
- L77-L78 `if not isinstance(producer, fx.Node) or producer.op != "call_function": continue`：输入必须是算子节点。`call_module` 节点不是函数，所以 module 级图在这里会被过滤掉。
- L79-L80 `if short_name(producer.target) != "conv2d": continue`：只保留卷积作为前驱。
- L81 `chain = [producer.name, node.name]`：记录 `conv -> relu`。
- L82-L89 `pool = next((user for user in node.users if user.op == "call_function" and short_name(user.target) in POOL_NAMES), None)`：在 ReLU 的消费者里找池化节点。用生成器 + `next(..., None)` 找到第一个或返回 `None`。
- L90-L91 `if pool is not None: chain.append(pool.name)`：若池化存在，把链延长为三段。
- L92 `candidates.append(" -> ".join(chain))`。
- L93 `return candidates`。

### 3.6 单张图的完整报告 describe（L96-L112）

- L96 `def describe(title, gm, log) -> str:`：打印标题、forward 代码、汇总、候选融合，并返回 forward 代码。
- L97-L99：打印分隔线标题。
- L100-L101 `log("GraphModule forward 代码:"); log(gm.code)`：`gm.code` 是等价的可执行 Python 前向代码，能直观看到图被翻译回源码的样子。
- L103 `summarize(gm, log)`：节点、边、直方图。
- L104 `candidates = find_fusion_candidates(gm)`。
- L105-L110：打印候选融合，若无则提示该粒度下看不到内部算子。
- L112 `return gm.code`。

### 3.7 主流程 main（L115-L149）

- L116-L118：命令行参数 `--out`，默认输出到 `artifacts/lenet_graph.txt`。
- L120-L122：设置种子、实例化 LeNet、切评估模式（无需梯度）。
- L124-L128：`log` 闭包，同时打印并收集到 `lines`。
- L130 `module_gm = fx.symbolic_trace(model)`：默认 Tracer 抓图，`nn.Conv2d`、`nn.Linear` 作为叶子，得到 module 级图。
- L131 `expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))`：用自定义 Tracer 抓图得到 operator 级 `Graph`，再包成 `GraphModule`。传 `model` 是为了让 `get_attr` 能解析到真实权重。
- L133-L134：分别输出两张图的报告并保留 forward 代码。
- L136-L139：打印节点数对比。
- L141-L144：把完整报告写入 `artifacts/lenet_graph.txt`，两份 forward 代码分别写入 `lenet_graphmodule_module_level.py` 与 `lenet_graphmodule_operator_level.py`。
- L145 `print(f"saved {args.out}")`。

## 4. 运行结果

运行方式：

```bash
python3 -m src.stage3_fx.inspect_graph
```

### 4.1 两种粒度的对比

| 粒度 | 节点数 | `nn.Conv2d` / `nn.Linear` | 权重表示 | 能看到 Conv+ReLU 融合机会 |
|------|--------|---------------------------|----------|--------------------------|
| module 级 | 14 | `call_module` 节点 | 藏在模块内部 | 否 |
| operator 级 | 24 | 展开为 `F.conv2d` / `F.linear` | `get_attr` 节点 | 是 |

module 级直方图：

```
call_function    6
call_method      1
call_module      5
output           1
placeholder      1
```

operator 级节点共 24 个，其中 `get_attr` 10 个（5 层参数层的权重与偏置），`call_function` 12 个（2 个 conv2d、2 个 max_pool2d、4 个 relu、3 个 linear、1 个 flatten 中的 flatten 属于 `call_method`）。

### 4.2 operator 级图的部分节点

```
#0   placeholder    x                target=x
        args=()  users=['conv2d']
#1   get_attr       conv1_weight     target=conv1.weight
        args=()  users=['conv2d']
#2   get_attr       conv1_bias       target=conv1.bias
        args=()  users=['conv2d']
#3   call_function  conv2d           target=conv2d
        args=(x, conv1_weight, conv1_bias, (1, 1), (2, 2), (1, 1), 1)
        users=['relu']
#4   call_function  relu             target=relu
        args=(conv2d)  users=['max_pool2d']
#5   call_function  max_pool2d       target=max_pool2d
        args=(relu, 2)  users=['conv2d_1']
```

注意 `conv2d` 的 `args` 里直接出现了 `conv1_weight`、`conv1_bias` 两个 `get_attr` 节点，以及步长、填充、膨胀、分组等参数。这正是编译器做常量折叠时需要的「算子 + 常量」形态。

### 4.3 依赖边

```
x -> conv2d
conv1_weight -> conv2d
conv1_bias -> conv2d
conv2d -> relu
relu -> max_pool2d
max_pool2d -> conv2d_1
...
linear_2 -> output
```

边就是节点之间的数据流。`graph.nodes` 的迭代顺序本身即拓扑序，因此从头到尾打印即可得到合法的执行顺序。

### 4.4 候选融合模式

```
conv2d -> relu -> max_pool2d
conv2d_1 -> relu_1 -> max_pool2d_1
```

两个卷积块各形成一条候选链。Stage 4 会真正把这类链替换为单个算子并验证等价性。

## 5. 观察与对编译器学习的意义

1. 图粒度决定可优化空间。module 级图只有 14 个节点，`conv1` 是一个黑盒；operator 级图把黑盒打开成 24 个节点，`conv2d`、`relu`、`max_pool2d` 都可被单独改写。编译器通常工作在 operator 级图上，因为融合需要看到边界。
2. `get_attr` 是常量折叠的入口。权重以 `get_attr` 出现在算子参数里，编译器可以在编译期把它们当作已知常量参与计算，例如把 `Conv` 的权重与后续 `BatchNorm` 的缩放合并成一个新权重。
3. 拓扑序是 pass 的遍历基础。`graph.nodes` 天然按依赖顺序排列，pass 遍历一次即可保证「用到某节点时它的输入已被处理」。
4. `users` 是安全改写的判据。替换一个节点前必须检查它的消费者，只有确认所有 `users` 都能接受新算子时才能改写，否则会破坏其他路径。
5. `gm.code` 说明 GraphModule 是可执行、可回读的 IR。这一点与 TVM Relax、MLIR 的文本 IR 一致：图既能在内存中以对象形式变换，又能打印成人类可读文本做审查。

## 6. tests/test_stage3_fx.py 逐行说明

代码位置：`tests/test_stage3_fx.py`

```python
import torch.fx as fx

from src.stage1_lenet.model import LeNet
from src.stage3_fx.inspect_graph import FunctionalTracer, find_fusion_candidates


def _ops(gm):
    return {getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes}


def test_expanded_graph_exposes_primitive_ops():
    model = LeNet()
    module_gm = fx.symbolic_trace(model)
    expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))

    assert "conv2d" not in _ops(module_gm)
    assert "conv2d" in _ops(expanded_gm)
    assert "linear" in _ops(expanded_gm)
    assert len(list(expanded_gm.graph.nodes)) > len(list(module_gm.graph.nodes))


def test_fusion_candidates_count():
    model = LeNet()
    expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))
    candidates = find_fusion_candidates(expanded_gm)
    assert len(candidates) == 2
    assert all("conv2d" in candidate and "relu" in candidate for candidate in candidates)
```

- L1 `import torch.fx as fx`：默认 Tracer 与 `GraphModule`。
- L3-L4：导入 LeNet 与待测函数。
- L7-L8 `_ops(gm)`：收集一张图里所有 target 的短名，用于断言算子层级。
- L11 `test_expanded_graph_exposes_primitive_ops`：验证 operator 级图暴露了基础算子。
- L12-L14：构造模型、module 级图、operator 级图。
- L16 `assert "conv2d" not in _ops(module_gm)`：module 级图里 `conv2d` 藏在 `call_module` 内，不应出现。
- L17-L18 `assert "conv2d" in _ops(expanded_gm)` 与 `assert "linear" in _ops(expanded_gm)`：operator 级图应出现这两种算子。
- L19 `assert len(...) > len(...)`：展开后节点更多。
- L22 `test_fusion_candidates_count`：LeNet 有两个卷积块，应找出 2 条候选链。
- L23-L25：构造 operator 级图并断言候选数量。
- L26 `assert all("conv2d" in candidate and "relu" in candidate ...)`：每条候选链都包含卷积与 ReLU。

## 7. 相关文件

| 文件 | 作用 |
|------|------|
| `src/stage3_fx/inspect_graph.py` | 图捕获、节点与依赖打印、融合候选识别 |
| `tests/test_stage3_fx.py` | 两种粒度的结构测试 |
| `artifacts/lenet_graph.txt` | 完整图报告 |
| `artifacts/lenet_graphmodule_module_level.py` | module 级 GraphModule 的 forward 代码 |
| `artifacts/lenet_graphmodule_operator_level.py` | operator 级 GraphModule 的 forward 代码 |
