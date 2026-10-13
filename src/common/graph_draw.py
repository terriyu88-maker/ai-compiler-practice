"""可选的 fx 图可视化工具：装了 pydot + graphviz 就把 GraphModule 导出成 SVG，否则优雅跳过。

依赖两项：
- Python 包 pydot：`torch.fx.passes.graph_drawer` 依赖它，缺了它绘图类只是空壳。
- 系统包 graphviz：提供 `dot` 二进制，用于把 DOT 文本渲染成 SVG/PNG。

用法：
    from src.common.graph_draw import draw_graph
    draw_graph(gm, ARTIFACTS / "lenet_bn_module_level.svg", name="lenet_bn_module_level")
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

import torch.fx as fx


def is_available() -> bool:
    """pydot 与 graphviz 的 dot 都在位时返回 True。"""
    try:
        from torch.fx.passes.graph_drawer import HAS_PYDOT
    except Exception:
        return False
    return bool(HAS_PYDOT) and shutil.which("dot") is not None


def draw_graph(
    gm: fx.GraphModule,
    path: Path,
    *,
    name: str | None = None,
    ignore_getattr: bool = False,
    ignore_parameters_and_buffers: bool = False,
    log: Callable[[str], None] = print,
) -> bool:
    """把 GraphModule 渲染成 SVG，返回是否成功。

    缺依赖或渲染失败时返回 False，不抛异常，调用方可以安全地忽略结果。
    """
    if not is_available():
        log(f"  跳过绘图 {path.name}：需要 pydot 与 graphviz(dot)")
        return False

    from torch.fx.passes.graph_drawer import FxGraphDrawer

    try:
        drawer = FxGraphDrawer(
            gm,
            name or path.stem,
            ignore_getattr=ignore_getattr,
            ignore_parameters_and_buffers=ignore_parameters_and_buffers,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        drawer.get_dot_graph().write_svg(str(path))
    except Exception as exc:  # 渲染失败不应打断主流程
        log(f"  绘图失败 {path.name}：{exc}")
        return False

    log(f"  已导出 {path}")
    return True
