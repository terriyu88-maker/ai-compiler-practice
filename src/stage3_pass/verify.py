"""验证 Conv+BN+ReLU 折叠 pass：数值对齐、BatchNorm 消除、节点数与 CPU 延迟对比。

在 Stage 1 的 LeNet-BN 上运行：symbolic_trace 得到 module 级图，两个
Conv2d -> BatchNorm2d -> ReLU 三元组被各替换成一个融合节点。
"""

import argparse
from pathlib import Path

import torch
import torch.fx as fx
from torch import nn

from src.common.graph_draw import draw_graph
from src.common.seed import set_seed
from src.common.timer import benchmark
from src.stage1_lenet.model import LeNetBN
from src.stage3_pass.fusion_pass import fuse_conv_bn_relu

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


@torch.no_grad()
def randomize_bn_stats(model: nn.Module) -> None:
    """把 BN 的 running 统计与仿射参数设为非平凡值。

    默认的 running_mean=0 / running_var=1 会让 BN 近似恒等，折叠后权重几乎不变，
    验证会失去意义。随机化后，折叠是否算对才真正经受考验。
    """
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            channels = module.num_features
            module.weight.copy_(torch.randn(channels) * 0.2 + 1.0)
            module.bias.copy_(torch.randn(channels) * 0.2)
            module.running_mean.copy_(torch.randn(channels) * 0.5)
            module.running_var.copy_(torch.rand(channels) * 1.5 + 0.5)


def count_modules(model: nn.Module, cls: type) -> int:
    return sum(1 for module in model.modules() if isinstance(module, cls))


def main():
    parser = argparse.ArgumentParser(description="Verify Conv+BN+ReLU fusion pass")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--report", type=Path, default=ARTIFACTS / "stage3_fusion_report.txt")
    parser.add_argument(
        "--draw-svg",
        action="store_true",
        help="额外导出折叠前/后的 Graphviz SVG（需 pydot + graphviz 的 dot）",
    )
    args = parser.parse_args()

    set_seed()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    model = LeNetBN().eval()
    randomize_bn_stats(model)
    x = torch.randn(args.batch_size, 1, 28, 28)

    with torch.no_grad():
        reference = model(x)

    gm = fx.symbolic_trace(model)
    gm.eval()
    before_code = gm.code
    before_nodes = len(list(gm.graph.nodes))
    before_bn = count_modules(gm, nn.BatchNorm2d)
    if args.draw_svg:
        draw_graph(
            gm,
            ARTIFACTS / "stage3_model_before.svg",
            name="stage3_model_before",
            log=log,
        )

    rewritten = fuse_conv_bn_relu(gm)
    after_code = gm.code
    after_nodes = len(list(gm.graph.nodes))
    after_bn = count_modules(gm, nn.BatchNorm2d)
    if args.draw_svg:
        draw_graph(
            gm,
            ARTIFACTS / "stage3_model_after.svg",
            name="stage3_model_after",
            log=log,
        )

    with torch.no_grad():
        fused_out = gm(x)
    max_err = (fused_out - reference).abs().max().item()

    with torch.no_grad():
        eager = benchmark(lambda: model(x), iters=args.iters)
        fused = benchmark(lambda: gm(x), iters=args.iters)

    log("Conv+BN+ReLU 折叠 pass 验证（LeNet-BN）")
    log("=" * 60)
    log(f"输入 shape: {tuple(x.shape)}")
    log("")
    log("折叠改写:")
    for item in rewritten:
        log(f"  {item}")
    log("")
    log(f"BatchNorm2d 模块数: {before_bn} -> {after_bn}")
    log(f"图节点数         : {before_nodes} -> {after_nodes}")
    log(f"最大绝对误差     : {max_err:.3e}")
    log("")
    log(f"原始模型 : {eager}")
    log(f"折叠模型 : {fused}")
    speedup = eager.median_ms / fused.median_ms if fused.median_ms > 0 else float("nan")
    log(f"加速比(median): {speedup:.2f}x")
    log("")
    log("折叠后 forward 代码:")
    log(after_code)

    args.report.write_text("\n".join(lines) + "\n")
    (ARTIFACTS / "stage3_model_before.py").write_text(before_code)
    (ARTIFACTS / "stage3_model_after.py").write_text(after_code)
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
