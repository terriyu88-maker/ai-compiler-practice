from collections import Counter

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

from src.stage1_lenet.manual_backprop import randomize_bn_buffers
from src.stage1_lenet.model import LeNetBN
from src.stage5_bridge.export_onnx import export_to_onnx

OPSET = 17


def _prepare(tmp_path):
    model = LeNetBN().eval()
    randomize_bn_buffers(model)
    return model, tmp_path / "folded.onnx", tmp_path / "nofold.onnx"


def test_constant_folding_folds_bn_into_conv(tmp_path):
    model, folded_path, nofold_path = _prepare(tmp_path)
    export_to_onnx(model, folded_path, OPSET, folding=True)
    export_to_onnx(model, nofold_path, OPSET, folding=False)

    folded = onnx.load(str(folded_path))
    nofold = onnx.load(str(nofold_path))
    folded_hist = Counter(node.op_type for node in folded.graph.node)
    nofold_hist = Counter(node.op_type for node in nofold.graph.node)

    assert folded_hist.get("BatchNormalization", 0) == 0
    assert nofold_hist.get("BatchNormalization", 0) == 2
    assert len(folded.graph.node) == 12
    assert len(nofold.graph.node) == 14
    assert folded_hist["Conv"] == nofold_hist["Conv"] == 2


def test_onnx_matches_pytorch(tmp_path):
    model, folded_path, nofold_path = _prepare(tmp_path)
    x = torch.randn(4, 1, 28, 28)
    with torch.no_grad():
        reference = model(x).numpy()

    for path, folding in ((folded_path, True), (nofold_path, False)):
        export_to_onnx(model, path, OPSET, folding=folding)
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)
        out = ReferenceEvaluator(onnx_model).run(None, {"input": x.numpy()})[0]
        assert np.abs(reference - out).max() < 1e-4
