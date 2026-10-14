# Stage 2：torch.fx 图捕获与依赖分析

本阶段把 LeNet-BN 转换成 GraphModule，打印节点、算子、依赖边与算子直方图，并对两种图粒度做对比。这是整个项目里最接近「AI 编译器输入端」的一步：编译器拿到的就是这种计算图，Stage 3 的折叠 pass 也从这里定位落点。

## 1. 验收标准

| 项目 | 标准 | 实际结果 |
|------|------|----------|
| 打印完整计算图与依赖边 | 节点、args、users、边齐全 | 已生成 `artifacts/lenet_bn_graph.txt`（301 行） |
| 标出候选融合模式 | 找出 2 条 `Conv -> BN -> ReLU` 链 | 找到 2 条 `conv2d -> batch_norm -> relu -> max_pool2d` |
| 单元测试 | 通过 | 2 passed（全量 9 passed） |

## 2. torch.fx 的核心概念

| 概念 | 含义 |
|------|------|
| `Node` | 图中的一个节点，有 `op`、`name`、`target`、`args`、`users` 等属性 |
| `Graph` | 节点的有序集合，迭代 `graph.nodes` 得到的就是拓扑序 |
| `GraphModule` | 由 `Graph` 生成的 `nn.Module` 子类，`gm.code` 可看到等价 Python 前向代码 |
| `op` | 节点类型，常见有 `placeholder`、`get_attr`、`call_function`、`call_method`、`call_module`、`output` |
| `get_attr` | 从模块上取属性（权重、偏置、BN 的 running 统计等），是常量来源 |
| `Tracer.is_leaf_module` | 决定哪些 `nn.Module` 不再展开、直接作为一个 `call_module` 节点。默认对 `torch.nn` 模块返回 True。 |

## 3. BN 为什么需要额外处理

`_BatchNorm.forward` 的第一行是 `self._check_input_dim(input)`（见 `torch/nn/modules/batchnorm.py`），其中对 `input.dim()` 做 Python 控制流判断。默认 Tracer 把 BN 当叶子、不进入其 `forward`，所以 module 级图不受影响；但 `FunctionalTracer` 要穿透 BN，此时 `input` 是 `Proxy`，无法参与 `if input.dim() != 4` 这类控制流，会抛：

```
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
```

该检查只做张量维数合法性校验，与图结构无关。本阶段的做法是在追踪期间把 `BatchNorm1d/2d/3d._check_input_dim` 临时替换为 no-op（`disable_bn_input_check`），追踪结束后恢复。这是使用 fx 时一个典型且可复用的坑。

## 4. src/stage2_fx/inspect_graph.py 逐行说明

代码位置：`src/stage2_fx/inspect_graph.py`（共 218 行）

可视化（可选）：加 `--draw-svg` 后，脚本把两张图分别导出为 Graphviz SVG；再加 `--draw-getattr` 会多导出一张保留常量节点的 operator 级图，见第 5.5 节与 `src/common/graph_draw.py`。

### 4.1 文件头与导入（L1-L27）

- L1-L9 文档字符串：说明脚本职责、要对比的两种粒度，以及在 operator 级图上标出 `Conv -> BN -> ReLU` 候选链。
- L10 `import argparse`：解析命令行参数。
- L11 `import contextlib`：实现临时替换 BN 维数检查的上下文管理器。
- L12 `from collections import Counter`：统计各类节点数量。
- L13-L14 `Path` / `Any`。
- L16 `import torch.fx as fx`。
- L17 `from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d, BatchNorm3d`：拿到三个具体 BN 类，它们各自覆盖了 `_check_input_dim`。
- L19 `from src.common.graph_draw import draw_graph`：可选的图可视化工具，缺 pydot/graphviz 时会安全跳过。
- L20-L21：导入公共种子工具与 LeNet-BN。
- L23-L24：仓库根目录与产物目录。
- L26 `POOL_NAMES = {...}`：池化算子名集合，用于识别池化节点。
- L27 `BN_NAMES = {"batch_norm"}`：BN 在 operator 级图中的算子名，用于识别 BN 节点。

### 4.2 屏蔽 BN 维数检查（L30-L47）

- L30 `@contextlib.contextmanager`：声明上下文管理器。
- L31 `def disable_bn_input_check():`。
- L32-L37 文档字符串：说明为什么需要。
- L38-L39：记录三个类原始的 `_check_input_dim`，并临时替换为 `lambda self, input: None`。
- L42-L43 `try: yield`：进入 `with` 块期间生效。
- L44-L46 `finally:`：无论是否异常，都按原顺序恢复三个方法，避免污染全局状态。

### 4.3 自定义 Tracer（L49-L57）

- L49 `class FunctionalTracer(fx.Tracer):`：继承默认 Tracer。
- L50 文档字符串。
- L52-L53 `is_leaf_module(...) -> False`：对所有模块都返回「不是叶子」，于是 `nn.Conv2d`、`nn.BatchNorm2d`、`nn.Linear` 会被穿透，其内部的 `F.conv2d`、`F.batch_norm`、`F.linear` 直接进入图中，权重变成 `get_attr`。
- L55-L57 `trace(root, concrete_args=None)`：重写 `trace`，先进入 `disable_bn_input_check()` 再调用父类实现。脚本与测试都通过这个入口抓图，因此补丁对两条调用路径都生效。

### 4.4 辅助格式化（L60-L71）

- L60-L61 `short_name(target)`：取可调用对象的名字。`torch.nn.functional.relu` 得到 `"relu"`，`F.batch_norm` 得到 `"batch_norm"`；非函数目标退化为 `str(target)`。
- L64 `def fmt(value):`：把节点参数格式化成可读字符串。
- L65-L66 `if isinstance(value, fx.Node): return value.name`：参数若是节点，输出节点名。
- L67-L68 元组/列表：递归格式化。
- L69-L70 字典：递归格式化。
- L71 `return repr(value)`：其余标量用 `repr`。

### 4.5 汇总 summarize（L74-L96）

- L74 `def summarize(gm, log) -> tuple[list, Counter]:`：输出节点、边、直方图，并返回边与直方图。
- L75 `nodes = list(gm.graph.nodes)`：把拓扑序节点链表转成 list。
- L76 `edges = [(producer.name, node.name) for node in nodes for producer in node.all_input_nodes]`：依赖边。`node.all_input_nodes` 给出该节点直接消费的所有生产者节点，包括 `args` 与 `kwargs` 中的节点。
- L77 `histogram = Counter(node.op for node in nodes)`。
- L79-L83：打印节点数、边数与按 `op` 排序的直方图。
- L84-L90 节点列表：L88 带序号打印 `op / name / target`，L89-L90 单独打印 `args` 与 `users` 以避免行过长。
- L92-L94：逐条打印 `producer -> consumer`。
- L96 `return edges, histogram`。

### 4.6 融合候选查找 find_fusion_candidates（L99-L127）

`Conv -> BN -> ReLU` 链比经典 LeNet 多一层 BN，因此从 ReLU 反查两层。

- L99 `def find_fusion_candidates(gm) -> list[str]:`：返回形如 `conv2d -> batch_norm -> relu -> max_pool2d` 的候选链。
- L101 `candidates = []`。
- L102-L103：遍历所有节点，只从 `call_function` 的 `relu` 出发。
- L105-L106 `norm = node.args[0]`：ReLU 的输入应是 BN 节点；否则跳过。
- L108 `if short_name(norm.target) not in BN_NAMES: continue`：确认是 `batch_norm`。
- L110-L111 `conv = norm.args[0]`：BN 的输入应是 `conv2d`；否则跳过。
- L115 `chain = [conv.name, norm.name, node.name]`：记录 `conv -> bn -> relu`。
- L116-L123 `pool = next((... in POOL_NAMES), None)`：在 ReLU 的消费者里找池化节点，找到第一个或返回 `None`。
- L124-L125 `if pool is not None: chain.append(pool.name)`：若池化存在，链延长为四段。
- L126-L127：拼接并返回。

过滤条件逐层校验 `op == "call_function"`，正是为了让 module 级图（`call_module` 节点）不被误判。

### 4.7 单张图的完整报告 describe（L130-L146）

- L130 `def describe(title, gm, log) -> str`。
- L131-L133：打印分隔线标题。
- L134-L135 `log(gm.code)`：`gm.code` 是等价的可执行 Python 前向代码。
- L137 `summarize(gm, log)`。
- L138-L144：打印候选融合，若无则提示该粒度下看不到内部算子。
- L146 `return gm.code`。

### 4.8 主流程 main（L149-L218）

- L150-L162：命令行参数 `--out`（默认 `artifacts/lenet_bn_graph.txt`）、`--draw-svg`（是否额外导出 SVG）、`--draw-getattr`（配合 `--draw-svg`，额外导出一张保留 `get_attr` 常量节点的 operator 级图）。
- L164-L166：设置种子、实例化 LeNet-BN、切评估模式（BN 用 running 统计，图与训练无关）。
- L168-L172：`log` 闭包，同时打印并收集到 `lines`。
- L174 `module_gm = fx.symbolic_trace(model)`：默认 Tracer 抓图，得到 module 级图。
- L175 `expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))`：自定义 Tracer 抓图得到 operator 级 `Graph`，再包成 `GraphModule`。传 `model` 是为了让 `get_attr` 能解析到真实权重与 BN 统计。
- L177-L178：分别输出两张图的报告并保留 forward 代码。
- L180-L203：`--draw-svg` 打开时导出图。module 级导出为 `lenet_bn_module_level.svg`；operator 级默认导出为 `lenet_bn_operator_level.svg`（置 `ignore_getattr=True` 降噪，隐藏 18 个常量节点）；再加 `--draw-getattr` 时，额外导出 `lenet_bn_operator_level_full.svg`（`ignore_getattr=False`，保留全部 `get_attr` 常量节点）。缺依赖时 `draw_graph` 打印提示并跳过。
- L205-L208：打印节点数对比。
- L210-L213：把完整报告写入 `lenet_bn_graph.txt`，两份 forward 代码分别写入 `lenet_bn_graphmodule_module_level.py` 与 `lenet_bn_graphmodule_operator_level.py`。
- L214 `print(f"saved {args.out}")`。

## 5. 运行结果

运行方式：

```bash
python3 -m src.stage2_fx.inspect_graph
```

需要出图时加 `--draw-svg`：

```bash
python3 -m src.stage2_fx.inspect_graph --draw-svg
```

### 5.1 两种粒度的对比

| 粒度 | 节点数 | 依赖边数 | `nn.Conv2d` / `nn.BatchNorm2d` / `nn.Linear` | 权重/BN 统计 | 能看到 Conv+BN 融合机会 |
|------|--------|----------|---------------------------------------------|--------------|--------------------------|
| module 级 | 16 | 15 | `call_module` 节点（7 个） | 藏在模块内部 | 否 |
| operator 级 | 34 | 33 | 展开为 `conv2d` / `batch_norm` / `linear` | 18 个 `get_attr` | 是 |

module 级直方图：

```
call_function    6
call_method      1
call_module      7
output           1
placeholder      1
```

operator 级直方图：

```
call_function    13
call_method      1
get_attr         18
output           1
placeholder      1
```

`get_attr` 18 个的构成：三个全连接各 2 个（权重、偏置）共 6 个；两个卷积各 2 个共 4 个；两个 BN 各 4 个（`weight`、`bias`、`running_mean`、`running_var`）共 8 个。合计 6+4+8=18。`call_function` 13 个：2 个 `conv2d`、2 个 `batch_norm`、4 个 `relu`、2 个 `max_pool2d`、3 个 `linear`。

### 5.2 operator 级图的 BN 部分

```
#8   call_function  batch_norm       target=batch_norm
        args=(conv2d, bn1_running_mean, bn1_running_var)
        users=['relu']
#9   call_function  relu             target=relu
        args=(batch_norm)  users=['max_pool2d']
```

对应的生成代码：

```python
bn1_weight = self.bn1.weight
bn1_bias = self.bn1.bias
bn1_running_mean = self.bn1.running_mean
bn1_running_var = self.bn1.running_var
batch_norm = torch.nn.functional.batch_norm(
    conv2d, bn1_running_mean, bn1_running_var,
    weight=bn1_weight, bias=bn1_bias,
    training=False, momentum=0.1, eps=1e-05,
)
```

注意 `training=False` 与四个 `get_attr` 一起出现：这正是 Stage 3 折叠 BN 所需的全部素材——`gamma`、`beta`、`mean`、`var`、`eps` 都是编译期已知常量。

### 5.3 依赖边（BN 段）

```
conv2d -> batch_norm
bn1_running_mean -> batch_norm
bn1_running_var -> batch_norm
bn1_weight -> batch_norm
bn1_bias -> batch_norm
batch_norm -> relu
relu -> max_pool2d
```

`graph.nodes` 的迭代顺序本身即拓扑序，因此从头到尾打印即可得到合法的执行顺序。

### 5.4 候选融合模式

```
conv2d -> batch_norm -> relu -> max_pool2d
conv2d_1 -> batch_norm_1 -> relu_1 -> max_pool2d_1
```

两个卷积块各形成一条候选链。Stage 3 会真正把 `Conv+BN` 折叠为单个卷积（`relu` 作为激活保留），并验证等价性。

### 5.5 Graphviz 可视化（`--draw-svg`）

`--draw-svg` 会额外导出两张 SVG，直观对比两种粒度；再加 `--draw-getattr` 会多导出一张保留全部常量节点的 operator 级图：

| 产物 | 内容 | 说明 |
|------|------|------|
| `artifacts/lenet_bn_module_level.svg` | module 级图 | 16 个节点，`conv1`/`bn1` 等是黑盒方框 |
| `artifacts/lenet_bn_operator_level.svg` | operator 级图（降噪） | 置 `ignore_getattr=True`，隐藏 18 个常量节点，只看数据流主线 |
| `artifacts/lenet_bn_operator_level_full.svg` | operator 级图（完整） | 加 `--draw-getattr` 生成，`ignore_getattr=False`，18 个 `get_attr` 常量节点全部显示 |

实现放在 `src/common/graph_draw.py`，核心是 `torch.fx.passes.graph_drawer.FxGraphDrawer`：

- `is_available()`：同时检查 `pydot` 是否可导入、系统是否有 `dot` 可执行文件。
- `draw_graph(gm, path, name=..., ignore_getattr=False, ...)`：用 `FxGraphDrawer` 生成 DOT 再写 SVG，失败时打印提示并返回 `False`，不会中断主流程。
- operator 级图节点多、`get_attr` 密集：默认导出置 `ignore_getattr=True` 保持可读；需要看权重/BN 统计等常量节点时，用 `--draw-getattr` 导出完整版。

导出为可选项（默认关闭），因此即使环境没有 `pydot`/`graphviz`，`inspect_graph` 仍可正常运行。

## 6. 观察与对编译器学习的意义

1. 图粒度决定可优化空间。module 级图只有 16 个节点，`conv1`/`bn1` 是黑盒；operator 级图把它们打开成 34 个节点，`conv2d`、`batch_norm`、`relu` 都可被单独改写。编译器通常工作在 operator 级图上，因为融合需要看到边界。
2. `get_attr` 是常量折叠的入口。BN 的 `weight/bias/running_mean/running_var` 都以 `get_attr` 出现在 `batch_norm` 的参数里，编译器可在编译期把它们当作已知常量参与代数化简，这正是 Stage 3 的做法。
3. 拓扑序是 pass 的遍历基础。`graph.nodes` 天然按依赖顺序排列，pass 遍历一次即可保证「用到某节点时它的输入已被处理」。
4. `users` 是安全改写的判据。替换节点前必须检查消费者，只有确认所有 `users` 都能接受新算子时才能改写。
5. `gm.code` 说明 GraphModule 是可执行、可回读的 IR，与 TVM Relax、MLIR 的文本 IR 一致。
6. 框架的 `nn.Module` 前向里可能夹杂与图无关的 Python 控制流（BN 的 `_check_input_dim`），追踪时需要显式处理。这类「框架实现细节」是接入任何图系统都要面对的现实问题。

## 7. tests/test_stage2_fx.py 逐行说明

代码位置：`tests/test_stage2_fx.py`

```python
import torch.fx as fx

from src.stage1_lenet.model import LeNetBN
from src.stage2_fx.inspect_graph import FunctionalTracer, find_fusion_candidates


def _ops(gm):
    return {getattr(node.target, "__name__", str(node.target)) for node in gm.graph.nodes}


def test_expanded_graph_exposes_primitive_ops():
    model = LeNetBN().eval()
    module_gm = fx.symbolic_trace(model)
    expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))

    assert "conv2d" not in _ops(module_gm)
    assert "conv2d" in _ops(expanded_gm)
    assert "batch_norm" in _ops(expanded_gm)
    assert "linear" in _ops(expanded_gm)
    assert len(list(expanded_gm.graph.nodes)) > len(list(module_gm.graph.nodes))


def test_fusion_candidates_count():
    model = LeNetBN().eval()
    expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))
    candidates = find_fusion_candidates(expanded_gm)
    assert len(candidates) == 2
    assert all(
        "conv2d" in candidate and "batch_norm" in candidate and "relu" in candidate
        for candidate in candidates
    )
```

- L1 `import torch.fx as fx`：默认 Tracer 与 `GraphModule`。
- L3-L4：导入 LeNet-BN 与待测对象。
- L7-L8 `_ops(gm)`：收集一张图里所有 target 的短名，用于断言算子层级。
- L11-L19 `test_expanded_graph_exposes_primitive_ops`：验证 operator 级图暴露了基础算子。`LeNetBN().eval()` 保证 BN 走推理分支；module 级图里没有 `conv2d`，operator 级图里有 `conv2d`、`batch_norm`、`linear`，且节点更多。
- L22-L29 `test_fusion_candidates_count`：LeNet-BN 有两个卷积块，应找出 2 条候选链；每条链同时含 `conv2d`、`batch_norm`、`relu`。

## 8. 相关文件

| 文件 | 作用 |
|------|------|
| `src/stage2_fx/inspect_graph.py` | 图捕获、节点与依赖打印、融合候选识别 |
| `src/common/graph_draw.py` | 可选 Graphviz SVG 导出（`--draw-svg` 使用） |
| `tests/test_stage2_fx.py` | 两种粒度的结构测试 |
| `artifacts/lenet_bn_graph.txt` | 完整图报告 |
| `artifacts/lenet_bn_graphmodule_module_level.py` | module 级 GraphModule 的 forward 代码 |
| `artifacts/lenet_bn_graphmodule_operator_level.py` | operator 级 GraphModule 的 forward 代码 |
| `artifacts/lenet_bn_module_level.svg` | module 级图（`--draw-svg` 生成） |
| `artifacts/lenet_bn_operator_level.svg` | operator 级图（`--draw-svg` 生成，隐藏常量节点） |
| `artifacts/lenet_bn_operator_level_full.svg` | operator 级完整图（`--draw-svg --draw-getattr` 生成，保留 `get_attr`） |

## 9. 与上下阶段的衔接

- 上游 Stage 1 定义了 LeNet-BN；本阶段把它的 `forward` 展开为图。
- 下游 Stage 3 复用本阶段的结构：`find_fusion_candidates` 给出 2 条 `conv2d -> batch_norm -> relu` 链，`get_attr` 提供折叠所需的常量，`gm.code` 与 `graph.lint()` 用于验证改写后的图仍然合法。
