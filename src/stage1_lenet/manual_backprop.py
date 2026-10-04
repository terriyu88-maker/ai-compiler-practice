"""手写 LeNet-BN 的前向 / 反向传播，与 autograd 逐参数对比。

这里不使用 autograd：所有算子（卷积、BatchNorm、池化、线性、ReLU、交叉熵）的
前向与梯度都由本文件显式计算，用于建立对反向传播的实际认知。

BatchNorm 按方案 B 处理：梯度检查在 eval 模式下进行，running_mean / running_var
视为固定常量（它们是 buffer，autograd 也不为其产生梯度）。在这个前提下，
y = gamma * x_hat + beta 仍是 x 的仿射变换，因此 dx、dgamma、dbeta 都有闭式解，
梯度检查覆盖全部可学习参数。
"""

import torch
import torch.nn.functional as F

from src.common.seed import set_seed
from src.stage1_lenet.model import LeNetBN

TOL = 1e-4


def conv2d_forward(x, weight, bias, padding=0, stride=1):
    """im2col 卷积：unfold 展开后用矩阵乘法完成。"""
    cols = F.unfold(x, kernel_size=weight.shape[2:], padding=padding, stride=stride)
    w_mat = weight.reshape(weight.shape[0], -1)
    h_out = (x.shape[2] + 2 * padding - weight.shape[2]) // stride + 1
    w_out = (x.shape[3] + 2 * padding - weight.shape[3]) // stride + 1
    out = (w_mat @ cols + bias.view(1, -1, 1)).reshape(
        x.shape[0], weight.shape[0], h_out, w_out
    )
    return out, (cols, weight, padding, stride, x.shape)


def conv2d_backward(grad_out, cache):
    cols, weight, padding, stride, in_shape = cache
    w_mat = weight.reshape(weight.shape[0], -1)
    g = grad_out.reshape(grad_out.shape[0], weight.shape[0], -1)
    grad_w = (g @ cols.transpose(1, 2)).sum(dim=0).reshape_as(weight)
    grad_b = g.sum(dim=(0, 2))
    grad_cols = w_mat.t() @ g
    grad_x = F.fold(
        grad_cols,
        output_size=(in_shape[2], in_shape[3]),
        kernel_size=weight.shape[2:],
        padding=padding,
        stride=stride,
    )
    return grad_x, grad_w, grad_b


def batchnorm2d_forward(x, weight, bias, running_mean, running_var, eps):
    """eval 模式 BN：y = gamma * (x - mean) / sqrt(var + eps) + beta，mean/var 固定。"""
    inv_std = torch.rsqrt(running_var + eps)
    mean = running_mean.reshape(1, -1, 1, 1)
    scale = (weight * inv_std).reshape(1, -1, 1, 1)
    shift = (bias.reshape(1, -1, 1, 1) - scale * mean)
    x_hat = (x - mean) * inv_std.reshape(1, -1, 1, 1)
    return scale * x + shift, (x_hat, scale)


def batchnorm2d_backward(grad_out, cache):
    x_hat, scale = cache
    grad_gamma = (grad_out * x_hat).sum(dim=(0, 2, 3))
    grad_beta = grad_out.sum(dim=(0, 2, 3))
    grad_x = grad_out * scale
    return grad_x, grad_gamma, grad_beta


def relu_forward(x):
    return x.clamp(min=0), x


def relu_backward(grad_out, x):
    return grad_out * (x > 0).to(grad_out.dtype)


def maxpool2d_forward(x, kernel_size=2):
    n, c, h, w = x.shape
    kh = kw = kernel_size
    oh, ow = h // kh, w // kw
    x_r = x.reshape(n, c, oh, kh, ow, kw).permute(0, 1, 2, 4, 3, 5)
    x_flat = x_r.reshape(n, c, oh, ow, kh * kw)
    out, idx = x_flat.max(dim=-1)
    return out, (x.shape, idx, kernel_size)


def maxpool2d_backward(grad_out, cache):
    shape, idx, k = cache
    n, c, oh, ow = grad_out.shape
    grad_flat = torch.zeros(
        n, c, oh, ow, k * k, dtype=grad_out.dtype, device=grad_out.device
    )
    grad_flat.scatter_(-1, idx.unsqueeze(-1), grad_out.unsqueeze(-1))
    grad_r = grad_flat.reshape(n, c, oh, ow, k, k).permute(0, 1, 2, 4, 3, 5)
    return grad_r.reshape(shape)


def linear_forward(x, weight, bias):
    return F.linear(x, weight, bias), x


def linear_backward(grad_out, x, weight, bias):
    g = grad_out.reshape(-1, grad_out.shape[-1])
    grad_w = g.t() @ x.reshape(-1, x.shape[-1])
    grad_b = g.sum(dim=0)
    grad_x = grad_out @ weight
    return grad_x, grad_w, grad_b


def cross_entropy_forward(logits, target):
    shifted = logits - logits.max(dim=1, keepdim=True).values
    log_probs = shifted - shifted.exp().sum(dim=1, keepdim=True).log()
    n = logits.shape[0]
    loss = -log_probs[torch.arange(n), target].mean()
    return loss, (log_probs.exp(), target)


def cross_entropy_backward(cache):
    probs, target = cache
    n = probs.shape[0]
    grad = probs.clone()
    grad[torch.arange(n), target] -= 1.0
    return grad / n


class ManualLeNetBN:
    def __init__(self, model: LeNetBN):
        self.m = model
        self.cache = {}

    def forward(self, x):
        m, c = self.m, self.cache
        x, c["conv1"] = conv2d_forward(x, m.conv1.weight.detach(), m.conv1.bias.detach(), padding=2)
        x, c["bn1"] = batchnorm2d_forward(
            x, m.bn1.weight.detach(), m.bn1.bias.detach(),
            m.bn1.running_mean.detach(), m.bn1.running_var.detach(), m.bn1.eps,
        )
        x, c["relu1"] = relu_forward(x)
        x, c["pool1"] = maxpool2d_forward(x)
        x, c["conv2"] = conv2d_forward(x, m.conv2.weight.detach(), m.conv2.bias.detach(), padding=0)
        x, c["bn2"] = batchnorm2d_forward(
            x, m.bn2.weight.detach(), m.bn2.bias.detach(),
            m.bn2.running_mean.detach(), m.bn2.running_var.detach(), m.bn2.eps,
        )
        x, c["relu2"] = relu_forward(x)
        x, c["pool2"] = maxpool2d_forward(x)
        c["flat_shape"] = x.shape
        x = x.flatten(1)
        x, c["fc1"] = linear_forward(x, m.fc1.weight.detach(), m.fc1.bias.detach())
        x, c["relu3"] = relu_forward(x)
        x, c["fc2"] = linear_forward(x, m.fc2.weight.detach(), m.fc2.bias.detach())
        x, c["relu4"] = relu_forward(x)
        logits, c["fc3"] = linear_forward(x, m.fc3.weight.detach(), m.fc3.bias.detach())
        return logits

    def backward(self, grad_logits):
        m, c = self.m, self.cache
        grads = {}

        g, gw, gb = linear_backward(grad_logits, c["fc3"], m.fc3.weight.detach(), m.fc3.bias.detach())
        grads["fc3.weight"], grads["fc3.bias"] = gw, gb

        g = relu_backward(g, c["relu4"])
        g, gw, gb = linear_backward(g, c["fc2"], m.fc2.weight.detach(), m.fc2.bias.detach())
        grads["fc2.weight"], grads["fc2.bias"] = gw, gb

        g = relu_backward(g, c["relu3"])
        g, gw, gb = linear_backward(g, c["fc1"], m.fc1.weight.detach(), m.fc1.bias.detach())
        grads["fc1.weight"], grads["fc1.bias"] = gw, gb

        g = g.reshape(c["flat_shape"])
        g = maxpool2d_backward(g, c["pool2"])
        g = relu_backward(g, c["relu2"])
        g, gg, gb = batchnorm2d_backward(g, c["bn2"])
        grads["bn2.weight"], grads["bn2.bias"] = gg, gb
        g, gw, gb = conv2d_backward(g, c["conv2"])
        grads["conv2.weight"], grads["conv2.bias"] = gw, gb

        g = maxpool2d_backward(g, c["pool1"])
        g = relu_backward(g, c["relu1"])
        g, gg, gb = batchnorm2d_backward(g, c["bn1"])
        grads["bn1.weight"], grads["bn1.bias"] = gg, gb
        g, gw, gb = conv2d_backward(g, c["conv1"])
        grads["conv1.weight"], grads["conv1.bias"] = gw, gb
        return grads


def randomize_bn_buffers(model: LeNetBN) -> None:
    """把 BN 的 running 统计与仿射参数设为非平凡值，让仿射项真正参与梯度检查。"""
    with torch.no_grad():
        for bn in (model.bn1, model.bn2):
            n = bn.num_features
            bn.running_mean.copy_(torch.randn(n) * 0.5)
            bn.running_var.copy_(torch.rand(n) + 0.5)
            bn.weight.copy_(torch.randn(n) * 0.2 + 1.0)
            bn.bias.copy_(torch.randn(n) * 0.2)


def run_gradcheck(batch_size: int = 8):
    set_seed()
    model = LeNetBN().eval()
    randomize_bn_buffers(model)
    x = torch.randn(batch_size, 1, 28, 28)
    target = torch.randint(0, 10, (batch_size,))

    logits_ref = model(x)
    loss_ref = F.cross_entropy(logits_ref, target)
    loss_ref.backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}

    manual = ManualLeNetBN(model)
    with torch.no_grad():
        logits_manual = manual.forward(x)
        forward_err = (logits_manual - logits_ref.detach()).abs().max().item()
        loss_manual, ce_cache = cross_entropy_forward(logits_manual, target)
        grads = manual.backward(cross_entropy_backward(ce_cache))

    loss_err = abs(loss_manual.item() - loss_ref.item())
    reports = {}
    for name, ref in ref_grads.items():
        reports[name] = (grads[name] - ref).abs().max().item()
    grad_err = max(reports.values())

    print(f"forward  max abs err = {forward_err:.3e}")
    print(f"loss     abs err     = {loss_err:.3e}")
    for name, err in reports.items():
        print(f"  grad {name:14s} max abs err = {err:.3e}")
    print(f"grad     max abs err = {grad_err:.3e}")

    return {
        "forward_max_abs_err": forward_err,
        "loss_abs_err": loss_err,
        "grad_max_abs_err": grad_err,
        "per_param_grad_err": reports,
    }


def main():
    report = run_gradcheck()
    assert report["forward_max_abs_err"] < TOL, report["forward_max_abs_err"]
    assert report["grad_max_abs_err"] < TOL, report["grad_max_abs_err"]
    print(f"PASS: forward / grad 与 autograd 的最大误差均 < {TOL:g}")


if __name__ == "__main__":
    main()
