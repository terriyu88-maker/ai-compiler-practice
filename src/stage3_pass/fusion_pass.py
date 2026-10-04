"""手写 torch.fx pass：把推理态的 Conv2d -> BatchNorm2d -> ReLU 折叠成一个节点。

BatchNorm 在推理态是逐通道仿射变换，可以精确折进前置卷积的权重与偏置：

    scale = gamma / sqrt(running_var + eps)
    W' = W * scale
    b' = (b - running_mean) * scale + beta

折叠后 BN 从图中消失，卷积直接输出等价结果；ReLU 无法折进权重，作为卷积的尾随
逐元素算子保留在同一个融合模块里，对应图中的一个 call_module 节点。
"""

import torch
import torch.fx as fx
from torch import nn
from torch.fx import GraphModule
from torch.nn import functional as F


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """把 bn 的推理态仿射变换折进 conv，返回带折叠权重与偏置的等价卷积。"""
    weight = conv.weight.detach()
    bias = conv.bias.detach() if conv.bias is not None else torch.zeros(conv.out_channels)

    scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = (bias - bn.running_mean.detach()) * scale + bn.bias.detach()

    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )
    with torch.no_grad():
        fused.weight.copy_(folded_weight)
        fused.bias.copy_(folded_bias)
    return fused


class FusedConvBNReLU(nn.Module):
    """Conv+BN 折叠后的卷积，外接 ReLU，整体对应图中的一个节点。"""

    def __init__(self, conv: nn.Conv2d, bn: nn.BatchNorm2d):
        super().__init__()
        self.conv = fold_conv_bn(conv, bn)

    def forward(self, x):
        return F.relu(self.conv(x))


def _match_triples(
    gm: GraphModule, modules: dict[str, nn.Module]
) -> list[tuple[fx.Node, fx.Node, fx.Node]]:
    """找出所有 (conv, bn, relu) 三元组，且 conv/bn 都只被这条链消费。"""
    triples = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not F.relu:
            continue
        bn_node = node.args[0] if node.args else None
        if not isinstance(bn_node, fx.Node) or bn_node.op != "call_module":
            continue
        if not isinstance(modules.get(bn_node.target), nn.BatchNorm2d):
            continue
        conv_node = bn_node.args[0] if bn_node.args else None
        if not isinstance(conv_node, fx.Node) or conv_node.op != "call_module":
            continue
        if not isinstance(modules.get(conv_node.target), nn.Conv2d):
            continue
        if len(bn_node.users) != 1 or len(conv_node.users) != 1:
            continue
        triples.append((conv_node, bn_node, node))
    return triples


def fuse_conv_bn_relu(gm: GraphModule) -> list[str]:
    """就地改写 gm，把每个 Conv-BN-ReLU 三元组替换为一个融合节点，返回改写说明。"""
    modules = dict(gm.named_modules())
    graph = gm.graph
    triples = _match_triples(gm, modules)

    rewritten: list[str] = []
    for index, (conv_node, bn_node, relu_node) in enumerate(triples):
        conv = modules[conv_node.target]
        bn = modules[bn_node.target]
        name = f"fused_conv_bn_relu_{index}"
        gm.add_submodule(name, FusedConvBNReLU(conv, bn))

        with graph.inserting_before(conv_node):
            fused_node = graph.call_module(name, args=(conv_node.args[0],))
        relu_node.replace_all_uses_with(fused_node)
        graph.erase_node(relu_node)
        graph.erase_node(bn_node)
        graph.erase_node(conv_node)

        rewritten.append(
            f"{conv_node.name} + {bn_node.name} + {relu_node.name} -> {name}"
        )

    gm.delete_all_unused_submodules()
    graph.lint()
    gm.recompile()
    return rewritten
