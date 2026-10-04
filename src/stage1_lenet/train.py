import argparse
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.common.seed import get_device, set_seed
from src.stage1_lenet.model import LeNetBN

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"


def build_loaders(batch_size: int):
    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0),
    )


def capture_layer_shapes(model: nn.Module, sample: torch.Tensor):
    shapes = {}
    handles = []

    def hook(_module, _inp, out, name):
        shapes[name] = tuple(out.shape)

    for name, module in model.named_modules():
        if not name:
            continue
        handles.append(module.register_forward_hook(lambda m, i, o, name=name: hook(m, i, o, name)))
    with torch.no_grad():
        model(sample)
    for h in handles:
        h.remove()
    return shapes


def grad_norms(model: nn.Module):
    return {
        n: float(p.grad.norm())
        for n, p in model.named_parameters()
        if p.grad is not None
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total, loss_sum / total


def main():
    parser = argparse.ArgumentParser(description="Train LeNet-BN on MNIST")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def log(msg: str):
        print(msg)
        lines.append(msg)

    train_loader, test_loader = build_loaders(args.batch_size)
    model = LeNetBN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log(f"device={device} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")
    log(f"params={sum(p.numel() for p in model.parameters())}")

    sample, _ = next(iter(train_loader))
    log("逐层输出 shape:")
    for name, shape in capture_layer_shapes(model, sample.to(device)).items():
        log(f"  {name:8s} -> {shape}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        running = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            loss.backward()
            if epoch == 1 and seen == 0:
                log("首个 batch 的逐层梯度 L2 范数:")
                for n, v in grad_norms(model).items():
                    log(f"  {n:14s} L2={v:.6f}")
            optimizer.step()
            running += loss.item() * y.numel()
            seen += y.numel()
        acc, test_loss = evaluate(model, test_loader, device)
        log(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={running / seen:.4f} test_loss={test_loss:.4f} "
            f"test_acc={acc * 100:.2f}% time={time.time() - start:.1f}s"
        )

    (ARTIFACTS / "lenet_bn.pt").parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACTS / "lenet_bn.pt")
    final_acc, _ = evaluate(model, test_loader, device)
    assert final_acc >= 0.98, f"test acc {final_acc:.4f} < 0.98"
    log(f"saved {ARTIFACTS / 'lenet_bn.pt'}")
    log(f"PASS: test_acc={final_acc * 100:.2f}% (>= 98%)")
    (ARTIFACTS / "stage1_train_log.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
