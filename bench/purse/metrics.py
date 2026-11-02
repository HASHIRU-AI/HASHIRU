"""Statistics for PURSE's results: bootstrap intervals, allocation efficiency, rank correlation, the declared against
hidden difficulty table, and how much of an allocator's work it bought from outside agents."""
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

from purse.market import Option
from purse.solvers import Attempt

__all__ = ["bootstrap_ci", "buy_share", "confusion", "efficiency", "paired_ci", "spearman"]


def bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """A 95% percentile interval for the mean of `xs`, resampling with replacement."""
    if not xs:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[min(n - 1, int(0.975 * n))]


def paired_ci(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """The interval for the mean difference a - b over paired runs (the same projects and seeds)."""
    return bootstrap_ci([x - y for x, y in zip(a, b, strict=True)], n, seed)


def efficiency(score: float, floor: float, oracle: float) -> float:
    """Where `score` falls between the cheapest fixed policy (0) and oracle-full (1)."""
    return (score - floor) / (oracle - floor) if oracle - floor > 1e-12 else 0.0


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        k = i
        while k + 1 < len(order) and xs[order[k + 1]] == xs[order[i]]:
            k += 1
        for m in range(i, k + 1):
            ranks[order[m]] = (i + k) / 2 + 1  # ties share their average rank
        i = k + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sd = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / sd if sd else 0.0


def confusion(declared: Mapping[str, str], hidden: Mapping[str, str]) -> dict[tuple[str, str], int]:
    """(declared difficulty, hidden bin) -> jobs, over jobs with both (a named model declares nothing)."""
    return dict(Counter((d, hidden[j]) for j, d in declared.items() if d and j in hidden))


def buy_share(attempts: Sequence[Attempt], market: Sequence[Option]) -> float:
    """Share of an allocator's attempts that went to outside agents."""
    agents = {o.name for o in market if o.kind == "agent"}
    tried = [a for a in attempts if a.outcome is None or not a.outcome.refused]
    return sum(a.option in agents for a in tried) / len(tried) if tried else 0.0
