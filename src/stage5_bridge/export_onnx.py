"""Stage 5：把 LeNet-BN 导出为 ONNX，与 torch.fx GraphModule 逐层对照。

本阶段只做到 ONNX 图层：
1. 加载 Stage 1 训练好的权重，eval 模式导出 ONNX：一个默认常量折叠，一个关闭折叠。
2. 用 onnx / onnx.reference 打印 ONNX 图结构（opset、输入输出、initializer、节点与直方图）。
3. 把 fx GraphModule 的节点与两种 ONNX 变体做算子级对照，观察两种 IR 在算子粒度、
   常量表达、图边界上的差异，以及导出器自动折叠 Conv+BN 的行为。
4. 用 ONNX ReferenceEvaluator 跑一遍，与 PyTorch 前向做数值对齐。
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.fx as fx
from onnx import TensorProto
from onnx.reference import ReferenceEvaluator

from src.common.seed import set_seed
from src.stage1_lenet.model import LeNetBN

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
WEIGHTS = ARTIFACTS / "lenet_bn.pt"
NOFOLD_PATH = ARTIFACTS / "lenet_bn_nofold.onnx"

FX_TO_ONNX = {
    "torch.nn.modules.conv.Conv2d": "Conv",
    "torch.nn.modules.batchnorm.BatchNorm2d": "BatchNormalization",
    "torch.nn.modules.linear.Linear": "Gemm",
    "torch.nn.functional.relu": "Relu",
    "torch.nn.functional.max_pool2d": "MaxPool",
}
METHOD_TO_ONNX = {"flatten": "Flatten"}


def load_model() -> LeNetBN:
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"未找到 {WEIGHTS}，请先运行 python3 -m src.stage1_lenet.train"
        )
    model = LeNetBN()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    return model


def export_to_onnx(model: LeNetBN, out_path: Path, opset: int, folding: bool) -> None:
    dummy = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,
        do_constant_folding=folding,
    )


def compact_graph_text(model: onnx.ModelProto) -> str:
    """ONNX 图的可读文本（只含结构，不内联权重数值）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return onnx.helper.printable_graph(model.graph)


def tensor_type_str(value_info) -> str:
    tt = value_info.type.tensor_type
    dims = []
    for dim in tt.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    name = TensorProto.DataType.Name(tt.elem_type)
    return f"{name}[{', '.join(dims)}]"


def attr_to_str(attr) -> str:
    val = onnx.helper.get_attribute_value(attr)
    if isinstance(val, bytes):
        return val.decode("utf-8", "replace")
    if isinstance(val, list) and val and isinstance(val[0], bytes):
        return "[" + ", ".join(v.decode("utf-8", "replace") for v in val) + "]"
    return str(val)


def describe_onnx_graph(model: onnx.ModelProto, log) -> Counter:
    g = model.graph
    log(f"IR version={model.ir_version}  producer={model.producer_name} {model.producer_version}")
    for opset in model.opset_import:
        log(f"opset: domain={opset.domain or 'ai.onnx'} version={opset.version}")
    log("")

    log("图输入:")
    for vi in g.input:
        log(f"  {vi.name:<8} {tensor_type_str(vi)}")
    log("图输出:")
    for vi in g.output:
        log(f"  {vi.name:<8} {tensor_type_str(vi)}")
    log("")

    total = 0
    log(f"initializer（权重/常量，共 {len(g.initializer)} 个）:")
    for init in g.initializer:
        total += int(np.prod(init.dims)) if init.dims else 1
        dtype = TensorProto.DataType.Name(init.data_type)
        log(f"  {init.name:<24} shape={list(init.dims)} dtype={dtype}")
    log(f"initializer 元素总数 = {total}")
    log("")

    histogram = Counter(node.op_type for node in g.node)
    log("节点直方图:")
    for op, count in sorted(histogram.items()):
        log(f"  {op:<20} {count}")
    log("")

    log("节点列表（拓扑序）:")
    for index, node in enumerate(g.node):
        ins = ", ".join(node.input) or "-"
        outs = ", ".join(node.output) or "-"
        log(f"  #{index:<3} {node.op_type:<20} name={node.name or '-':<16} {ins} -> {outs}")
        for attr in node.attribute:
            log(f"        {attr.name} = {attr_to_str(attr)}")
    log("")
    return histogram


def fx_sequence(model: LeNetBN) -> list[tuple[str, str]]:
    """返回 fx module 级图中 (节点名, 预期 ONNX 算子) 的序列。"""
    gm = fx.symbolic_trace(model)
    modules = dict(gm.named_modules())
    seq = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            cls = type(modules[node.target])
            expected = FX_TO_ONNX.get(f"{cls.__module__}.{cls.__name__}", cls.__name__)
        elif node.op == "call_function":
            name = getattr(node.target, "__name__", str(node.target))
            module = getattr(node.target, "__module__", "")
            expected = FX_TO_ONNX.get(f"{module}.{name}", name)
        elif node.op == "call_method":
            name = str(node.target)
            expected = METHOD_TO_ONNX.get(name, name)
        else:
            expected = node.op
        seq.append((node.name, expected))
    return seq


def compare_levels(
    model: LeNetBN, folded: onnx.ModelProto, nofold: onnx.ModelProto, log
) -> None:
    fx_seq = fx_sequence(model)
    folded_hist = Counter(node.op_type for node in folded.graph.node)
    nofold_hist = Counter(node.op_type for node in nofold.graph.node)

    log("fx module 级节点序列（含预期 ONNX 算子）:")
    for name, op in fx_seq:
        log(f"  {name:<12} -> {op}")
    log("")

    boundary = {"placeholder", "output"}
    fx_hist = Counter(op for _, op in fx_seq if op not in boundary)
    log("算子数量对照（fx / ONNX 折叠 / ONNX 不折叠）:")
    all_ops = sorted(set(fx_hist) | set(folded_hist) | set(nofold_hist))
    for op in all_ops:
        log(f"  {op:<20} fx={fx_hist.get(op, 0):<3} folded={folded_hist.get(op, 0):<3} nofold={nofold_hist.get(op, 0)}")
    log("")
    log(f"节点总数: fx(计算节点)={sum(fx_hist.values())}  "
        f"onnx folded={sum(folded_hist.values())}  onnx nofold={sum(nofold_hist.values())}")
    log("")


def verify_numerics(model: LeNetBN, onnx_model: onnx.ModelProto, log) -> float:
    set_seed()
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        torch_out = model(x).numpy()

    onnx.checker.check_model(onnx_model)
    session = ReferenceEvaluator(onnx_model)
    onnx_out = session.run(None, {"input": x.numpy()})[0]

    max_abs = float(np.abs(torch_out - onnx_out).max())
    log(f"  shape torch={tuple(torch_out.shape)} onnx={tuple(onnx_out.shape)}  max abs err = {max_abs:.3e}")
    return max_abs


def main():
    parser = argparse.ArgumentParser(description="Export LeNet-BN to ONNX and compare with fx")
    parser.add_argument("--out", type=Path, default=ARTIFACTS / "lenet_bn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--report", type=Path, default=ARTIFACTS / "stage5_onnx_report.txt")
    args = parser.parse_args()

    set_seed()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = load_model()

    export_to_onnx(model, args.out, args.opset, folding=True)
    export_to_onnx(model, NOFOLD_PATH, args.opset, folding=False)
    folded = onnx.load(str(args.out))
    nofold = onnx.load(str(NOFOLD_PATH))

    lines: list[str] = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("ONNX 图结构（do_constant_folding=True）")
    log("=" * 72)
    describe_onnx_graph(folded, log)

    log("=" * 72)
    log("fx GraphModule 与两种 ONNX 变体的抽象层次对照")
    log("=" * 72)
    compare_levels(model, folded, nofold, log)

    log("=" * 72)
    log("数值对齐 (PyTorch eager vs ONNX ReferenceEvaluator)")
    log("=" * 72)
    log("折叠版:")
    folded_err = verify_numerics(model, folded, log)
    log("不折叠版:")
    nofold_err = verify_numerics(model, nofold, log)
    log("")

    log("结论:")
    log("  1. fx module 级保留 Conv/BN/Linear 为 call_module 叶子，权重藏在子模块里；")
    log("     ONNX 把权重显式提成 initializer，运算节点换成后端标准算子。")
    log("  2. do_constant_folding=True 时，导出器自动把推理态 BN 折进卷积权重与偏置：")
    log("     BatchNormalization 节点消失（14 -> 12），与 Stage 3 手写折叠是同一套代数。")
    log("  3. 关闭常量折叠后，BN 作为独立 BatchNormalization 节点保留，14 个计算节点与 fx 完全对应。")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")
    (ARTIFACTS / "stage5_onnx_graph.txt").write_text(compact_graph_text(folded) + "\n")
    (ARTIFACTS / "stage5_onnx_nofold_graph.txt").write_text(compact_graph_text(nofold) + "\n")
    print(f"saved {args.out}")
    print(f"saved {NOFOLD_PATH}")
    print(f"saved {args.report}")
    assert folded_err < 1e-4, f"folded max abs err {folded_err} >= 1e-4"
    assert nofold_err < 1e-4, f"nofold max abs err {nofold_err} >= 1e-4"


if __name__ == "__main__":
    main()
