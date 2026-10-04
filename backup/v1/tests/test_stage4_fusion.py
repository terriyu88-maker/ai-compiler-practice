import torch
import torch.fx as fx
import torch.nn as nn

from src.stage4_pass.fusion_pass import fold_conv_bn, fuse_conv_bn_relu
from src.stage4_pass.model import ConvBNNet, randomize_bn_stats


def test_fold_conv_bn_matches_eval_formula():
    conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
    bn = nn.BatchNorm2d(4)
    with torch.no_grad():
        bn.weight.copy_(torch.randn(4).abs() + 0.5)
        bn.bias.copy_(torch.randn(4))
        bn.running_mean.copy_(torch.randn(4))
        bn.running_var.copy_(torch.rand(4) + 0.3)
    conv.eval()
    bn.eval()

    x = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        expected = bn(conv(x))
        got = fold_conv_bn(conv, bn)(x)

    assert torch.allclose(got, expected, atol=1e-5)


def test_pass_is_numerically_equivalent():
    model = ConvBNNet().eval()
    randomize_bn_stats(model)
    x = torch.randn(4, 1, 28, 28)

    gm = fx.symbolic_trace(model)
    gm.eval()
    fuse_conv_bn_relu(gm)
    gm.eval()

    with torch.no_grad():
        assert torch.allclose(gm(x), model(x), atol=1e-5)


def test_pass_removes_bn_and_shrinks_graph():
    model = ConvBNNet().eval()
    randomize_bn_stats(model)

    gm = fx.symbolic_trace(model)
    before = len(list(gm.graph.nodes))
    fuse_conv_bn_relu(gm)
    after = len(list(gm.graph.nodes))

    assert after == before - 4
    assert not any(isinstance(module, nn.BatchNorm2d) for module in gm.modules())
