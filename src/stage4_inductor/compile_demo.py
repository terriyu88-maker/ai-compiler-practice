"""torch.compile / Inductor 观察：对比 eager 与编译后的延迟，并导出生成的算子代码。

本机为 CPU-only，Inductor 会生成 C++ / OpenMP 代码而非 Triton kernel。
观察重点一致：哪些算子被融合进同一个 kernel，以及融合为什么减少访存。
"""

import os

os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

import argparse
import re
import shutil
import time
from pathlib import Path

import torch
import torch._dynamo

torch._dynamo.config.debug_dir_root = "/tmp/torch_compile_debug"

from src.common.seed import set_seed
from src.common.timer import benchmark
from src.stage1_lenet.model import LeNetBN

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

KERNEL_DEF_RE = re.compile(
    r"^(\w+) = async_compile\.(?:cpp_pybinding|triton)", re.MULTILINE
)


def newest_output_code(root: Path) -> Path | None:
    files = list(root.rglob("output_code.py"))
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description="Observe torch.compile / Inductor codegen")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cpu")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    model = LeNetBN().eval()
    x = torch.randn(args.batch_size, 1, 28, 28)

    with torch.no_grad():
        reference = model(x)
        eager = benchmark(lambda: model(x), iters=args.iters, device=device)
    log(f"eager   : {eager}")

    compiled = torch.compile(model)
    log("compiling (first call triggers codegen) ...")
    start = time.time()
    with torch.no_grad():
        out = compiled(x)
    compile_seconds = time.time() - start
    max_err = (out - reference).abs().max().item()
    log(f"compile wall time: {compile_seconds:.1f}s, max abs err vs eager: {max_err:.3e}")

    with torch.no_grad():
        compiled_result = benchmark(lambda: compiled(x), iters=args.iters, device=device)
    log(f"compiled: {compiled_result}")
    log(f"speedup (median): {eager.median_ms / compiled_result.median_ms:.2f}x")

    debug_root = Path(getattr(torch._dynamo.config, "debug_dir_root", "/tmp/torch_compile_debug"))
    code_path = newest_output_code(debug_root)
    if code_path is not None:
        destination = ARTIFACTS / "inductor_generated_code.py"
        shutil.copyfile(code_path, destination)
        code = code_path.read_text()
        kernels = sorted(set(KERNEL_DEF_RE.findall(code)))
        log("")
        log(f"generated code -> {destination}")
        log(f"source debug   -> {code_path}")
        log(f"generated lines: {len(code.splitlines())}")
        log(f"defined kernels ({len(kernels)}):")
        for kernel in kernels:
            log(f"  {kernel}")
    else:
        log(f"no output_code.py found under {debug_root}")

    (ARTIFACTS / "stage4_inductor_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
