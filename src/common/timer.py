import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BenchResult:
    median_ms: float
    min_ms: float
    max_ms: float
    samples_ms: list[float]

    def __str__(self) -> str:
        return (
            f"median={self.median_ms:.3f}ms "
            f"min={self.min_ms:.3f}ms max={self.max_ms:.3f}ms"
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    iters: int = 10,
    device: torch.device | None = None,
) -> BenchResult:
    device = device or torch.device("cpu")
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        synchronize(device)
        samples: list[float] = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return BenchResult(
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        samples_ms=samples,
    )
