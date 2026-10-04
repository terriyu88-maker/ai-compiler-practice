"""torch.fx 图捕获：把 LeNet 转成 GraphModule，打印节点、依赖边与算子直方图。

对比两种粒度：
- module 级：默认 Tracer 把 nn.Conv2d / nn.Linear 当作叶子，图中是 call_module 节点。
- operator 级：自定义 Tracer 穿透 nn.Module，图中是 F.conv2d / F.linear 等算子，权重变成 get_attr。
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch.fx as fx

from src.common.seed import set_seed
from src.stage1_lenet.model import LeNet

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

POOL_NAMES = {"max_pool2d", "avg_pool2d", "adaptive_avg_pool2d"}


class FunctionalTracer(fx.Tracer):
    """不把任何 nn.Module 当作叶子，从而展开成算子级计算图。"""

    def is_leaf_module(self, module, qualname: str) -> bool:
        return False


def short_name(target: Any) -> str:
    return getattr(target, "__name__", str(target))


def fmt(value: Any) -> str:
    if isinstance(value, fx.Node):
        return value.name
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(fmt(v) for v in value) + ")"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {fmt(v)}" for k, v in value.items()) + "}"
    return repr(value)


def summarize(gm: fx.GraphModule, log) -> tuple[list, Counter]:
    nodes = list(gm.graph.nodes)
    edges = [(producer.name, node.name) for node in nodes for producer in node.all_input_nodes]
    histogram = Counter(node.op for node in nodes)

    log(f"节点数={len(nodes)} 依赖边数={len(edges)}")
    log("算子直方图:")
    for op, count in sorted(histogram.items()):
        log(f"  {op:<16} {count}")
    log("")
    log("节点列表（拓扑序）:")
    for index, node in enumerate(nodes):
        target = short_name(node.target) if node.op != "placeholder" else str(node.target)
        users = [user.name for user in node.users]
        log(f"  #{index:<3} {node.op:<14} {node.name:<16} target={target}")
        log(f"        args={fmt(node.args)}")
        log(f"        users={users}")
    log("")
    log("依赖边 (producer -> consumer):")
    for producer, consumer in edges:
        log(f"  {producer} -> {consumer}")
    log("")
    return edges, histogram


def find_fusion_candidates(gm: fx.GraphModule) -> list[str]:
    """找出 Conv -> ReLU -> MaxPool 形式的融合候选链。"""
    candidates = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or short_name(node.target) != "relu":
            continue
        producer = node.args[0] if node.args else None
        if not isinstance(producer, fx.Node) or producer.op != "call_function":
            continue
        if short_name(producer.target) != "conv2d":
            continue
        chain = [producer.name, node.name]
        pool = next(
            (
                user
                for user in node.users
                if user.op == "call_function" and short_name(user.target) in POOL_NAMES
            ),
            None,
        )
        if pool is not None:
            chain.append(pool.name)
        candidates.append(" -> ".join(chain))
    return candidates


def describe(title: str, gm: fx.GraphModule, log) -> str:
    log("=" * 72)
    log(title)
    log("=" * 72)
    log("GraphModule forward 代码:")
    log(gm.code)
    log("")
    summarize(gm, log)
    candidates = find_fusion_candidates(gm)
    log("候选融合模式 (Conv -> ReLU -> MaxPool):")
    if candidates:
        for candidate in candidates:
            log(f"  {candidate}")
    else:
        log("  (无：该粒度下看不到内部算子)")
    log("")
    return gm.code


def main():
    parser = argparse.ArgumentParser(description="Inspect LeNet as a torch.fx graph")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "lenet_graph.txt")
    args = parser.parse_args()

    set_seed()
    model = LeNet()
    model.eval()

    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    module_gm = fx.symbolic_trace(model)
    expanded_gm = fx.GraphModule(model, FunctionalTracer().trace(model))

    module_code = describe("module 级图（默认 Tracer，nn.Module 作为叶子）", module_gm, log)
    expanded_code = describe("operator 级图（自定义 Tracer，穿透 nn.Module）", expanded_gm, log)

    log("对比结论:")
    log(f"  module 级节点数   = {len(list(module_gm.graph.nodes))}")
    log(f"  operator 级节点数 = {len(list(expanded_gm.graph.nodes))}")
    log(f"  operator 级比 module 级多出 {len(list(expanded_gm.graph.nodes)) - len(list(module_gm.graph.nodes))} 个节点")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (ARTIFACTS / "lenet_graphmodule_module_level.py").write_text(module_code)
    (ARTIFACTS / "lenet_graphmodule_operator_level.py").write_text(expanded_code)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
