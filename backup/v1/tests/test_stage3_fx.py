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
