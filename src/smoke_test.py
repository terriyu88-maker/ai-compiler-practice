import torch
from torch import nn

from src.common.seed import get_device, set_seed
from src.common.timer import benchmark


class MLP(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128, out_features: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    set_seed()
    device = get_device()
    print(f"torch={torch.__version__} device={device}")

    model = MLP().to(device)
    x = torch.randn(32, 784, device=device)
    logits = model(x)
    assert logits.shape == (32, 10), logits.shape
    print(f"forward ok: input {tuple(x.shape)} -> logits {tuple(logits.shape)}")

    loss = nn.functional.cross_entropy(logits, torch.randint(0, 10, (32,), device=device))
    loss.backward()
    grad_norm = model.net[0].weight.grad.norm().item()
    assert grad_norm > 0
    print(f"backward ok: first linear grad norm = {grad_norm:.4f}")

    result = benchmark(lambda: model(x), device=device)
    print(f"benchmark ok: {result}")


if __name__ == "__main__":
    main()
