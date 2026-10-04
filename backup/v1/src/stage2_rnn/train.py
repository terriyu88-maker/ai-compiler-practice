import argparse
import time
from pathlib import Path

import torch
from torch import nn

from src.common.seed import get_device, set_seed
from src.stage2_rnn.data import build_loaders
from src.stage2_rnn.model import CELL_TYPES, SentimentClassifier

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


@torch.no_grad()
def evaluate(model, loader, device):
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


def train_one(cell_type, args, train_loader, test_loader, vocab_size, device, log):
    set_seed()
    model = SentimentClassifier(
        vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        cell_type=cell_type,
        dropout=args.dropout,
        pad_idx=0,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())

    log(f"=== {cell_type.upper()} (params={num_params}) ===")
    epoch_times = []
    final_acc = 0.0
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running += loss.item() * y.numel()
            seen += y.numel()
        dt = time.time() - start
        epoch_times.append(dt)
        final_acc, test_loss = evaluate(model, test_loader, device)
        log(
            f"epoch {epoch}/{args.epochs} train_loss={running / seen:.4f} "
            f"test_loss={test_loss:.4f} test_acc={final_acc * 100:.2f}% time={dt:.1f}s"
        )

    torch.save(model.state_dict(), ARTIFACTS / f"stage2_{cell_type}.pt")
    return {
        "cell_type": cell_type.upper(),
        "params": num_params,
        "final_acc": final_acc,
        "mean_epoch_s": sum(epoch_times) / len(epoch_times),
        "total_s": sum(epoch_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare RNN vs LSTM on IMDB subset")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--train-per-class", type=int, default=5000)
    parser.add_argument("--test-per-class", type=int, default=2500)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--max-vocab", type=int, default=20000)
    args = parser.parse_args()

    set_seed()
    device = get_device()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def log(msg: str):
        print(msg)
        lines.append(msg)

    train_loader, test_loader, vocab = build_loaders(
        batch_size=args.batch_size,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        max_len=args.max_len,
        max_vocab=args.max_vocab,
    )
    log(
        f"device={device} vocab={len(vocab)} "
        f"train={len(train_loader.dataset)} test={len(test_loader.dataset)} "
        f"max_len={args.max_len} epochs={args.epochs} batch_size={args.batch_size}"
    )

    results = [
        train_one(cell, args, train_loader, test_loader, len(vocab), device, log)
        for cell in CELL_TYPES
    ]

    header = f"{'cell':<6}{'params':>10}{'final_acc':>12}{'mean_epoch_s':>15}{'total_s':>10}"
    log("")
    log("对比结果:")
    log(header)
    log("-" * len(header))
    for r in results:
        log(
            f"{r['cell_type']:<6}{r['params']:>10}{r['final_acc'] * 100:>11.2f}%"
            f"{r['mean_epoch_s']:>15.1f}{r['total_s']:>10.1f}"
        )

    src = {r["cell_type"]: r for r in results}
    diff = (src["LSTM"]["final_acc"] - src["RNN"]["final_acc"]) * 100
    log(f"LSTM - RNN final_acc 差: {diff:+.2f} 个百分点")
    for r in results:
        assert r["final_acc"] > 0.75, f"{r['cell_type']} acc {r['final_acc']:.4f} <= 0.75"

    (ARTIFACTS / "stage2_rnn_compare.txt").write_text("\n".join(lines) + "\n")
    log("PASS: 两种单元均收敛（final_acc > 75%）")


if __name__ == "__main__":
    main()
