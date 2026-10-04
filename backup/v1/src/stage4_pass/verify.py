"""验证 Conv+BN+ReLU 折叠 pass：数值对齐、BatchNorm 消除、节点数与 CPU 延迟对比。"""

import argparse
from pathlib import Path

import torch
import torch.fx as fx
from torch import nn

from src.common.seed import set_seed
from src.common.timer import benchmark
from src.stage4_pass.fusion_pass import fuse_conv_bn_relu
from src.stage4_pass.model import ConvBNNet, randomize_bn_stats

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def count_modules(model: nn.Module, cls: type) -> int:
    return sum(1 for module in model.modules() if isinstance(module, cls))


def main():
    parser = argparse.ArgumentParser(description="Verify Conv+BN+ReLU fusion pass")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--report", type=Path, default=ARTIFACTS / "stage4_fusion_report.txt")
    args = parser.parse_args()

    set_seed()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    model = ConvBNNet().eval()
    randomize_bn_stats(model)
    x = torch.randn(args.batch_size, 1, 28, 28)

    with torch.no_grad():
        reference = model(x)

    gm = fx.symbolic_trace(model)
    gm.eval()
    before_code = gm.code
    before_nodes = len(list(gm.graph.nodes))
    before_bn = count_modules(gm, nn.BatchNorm2d)

    rewritten = fuse_conv_bn_relu(gm)
    after_code = gm.code
    after_nodes = len(list(gm.graph.nodes))
    after_bn = count_modules(gm, nn.BatchNorm2d)

    with torch.no_grad():
        fused_out = gm(x)
    max_err = (fused_out - reference).abs().max().item()

    with torch.no_grad():
        eager = benchmark(lambda: model(x), iters=args.iters)
        fused = benchmark(lambda: gm(x), iters=args.iters)

    log("Conv+BN+ReLU 折叠 pass 验证")
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
    (ARTIFACTS / "stage4_model_before.py").write_text(before_code)
    (ARTIFACTS / "stage4_model_after.py").write_text(after_code)
    print(f"saved {args.report}")


if __name__ == "__main__":
    main()
