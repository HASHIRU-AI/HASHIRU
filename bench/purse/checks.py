"""The validity checks (spec §12, V1 to V7): computed from the outcome table before any allocator runs, they hold the
benchmark to measuring allocation. V8 (replay matches live) needs live runs and is `purse run --live`."""
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from purse.jobs import Job
from purse.market import Option, priced
from purse.oracles import BUYABLE, c_full, expected_cost, expected_score, oracle_buys, oracle_full, sweep_prices, voi
from purse.projects import Project, fold_of, make_project, score, with_budget
from purse.solvers import SKIP, Allocation, fixed, hybrid
from purse.table import OutcomeTable, cell_stats

__all__ = ["Check", "run_checks"]

PER_RHO = 30
TAU = 0.5
SPARK, LUNA, QWEN = "meta/muse-spark-1.3", "gpt-6-luna", "ollama_chat/qwen3.5:4b"


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    threshold: str
    passed: bool


def _projects(pool: Sequence[Job], table: OutcomeTable, market: Sequence[Option], rho: float,
              seed: int) -> list[Project]:
    """PER_RHO projects at TAU, each drawn on its own (a job may recur across projects, never within one)."""
    out = []
    for i in range(PER_RHO):
        p = make_project(pool, i % 2, rho, seed * 10_000 + i, set())
        out.append(with_budget(p, TAU * c_full(p, table, market)))
    return out


def _within(p: Project, alloc: Allocation, table: OutcomeTable, market: Sequence[Option]) -> Allocation:
    """`alloc` cut to what its expected costs let it pay for, jobs in value order (as the executor runs them)."""
    out, spent = {}, 0.0
    for j in sorted(p.jobs, key=lambda x: (-x.value, x.id)):
        cost = expected_cost(p, {j.id: alloc.get(j.id, SKIP)}, table, market)
        out[j.id] = alloc.get(j.id, SKIP) if spent + cost <= p.budget_usd + 1e-12 else SKIP
        spent += cost if out[j.id] != SKIP else 0.0
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_checks(pool: Sequence[Job], table: OutcomeTable, market: Sequence[Option], seed: int = 0) -> list[Check]:
    """V1 to V7 on projects drawn from `pool`; deterministic in `seed`."""
    prices = sweep_prices(table, pool, market)
    market = priced(market, prices[1])
    by_rho = {rho: _projects(pool, table, market, rho, seed + k) for k, rho in enumerate((1.0, 0.5, 0.0))}
    mid = by_rho[0.5]
    checks = []

    oracle = _mean([expected_score(p, oracle_full(p, table, market, p.budget_usd), table, market) for p in mid])
    policies = {o.name: [fixed(p, o.name, market) for p in mid] for o in market if o.kind != "agent"}
    policies["hybrid"] = [hybrid(p) for p in mid]
    best = max(_mean([expected_score(p, _within(p, a, table, market), table, market)
                      for p, a in zip(mid, allocs, strict=True)]) for allocs in policies.values())
    checks.append(Check("V1 allocation matters", oracle - best, ">= 0.10 over the best fixed policy",
                        oracle - best >= 0.10))

    covered = _mean([sum(j.value for j in p.jobs if _within(p, fixed(p, SPARK), table, market)[j.id] != SKIP)
                     / sum(j.value for j in p.jobs) for p in mid])
    checks.append(Check("V2 the budget binds", covered, f"< 0.60 of the value affordable on {SPARK}", covered < 0.60))

    seen = cell_stats(table, pool, hidden=False, market=market)
    levels = sorted({j.level for j in pool})

    def rate(fam: str, level: int, option: str) -> float:
        s = seen.get((fam, level, None, option))
        return s.rate if s else 0.0
    families = sorted({j.family for j in pool})
    separated = sum(any(rate(f, lv, LUNA) >= 0.8 and rate(f, lv, QWEN) <= 0.4 for lv in levels)
                    and any(rate(f, lv, SPARK) - rate(f, lv, LUNA) >= 0.2 for lv in levels) for f in families)
    checks.append(Check("V3 tiers separate", separated, f"all {len(families)} families", separated == len(families)))

    hard = sum(rate(f, max(levels), SPARK) <= 0.6 for f in families)
    checks.append(Check("V4 headroom", hard, f"{SPARK} <= 60% at the top level in >= 2 families", hard >= 2))

    hidden = cell_stats(table, pool, hidden=True, market=market)
    at_one = _mean([voi(p, table, hidden, market, p.budget_usd) for p in by_rho[1.0]])
    at_zero = _mean([voi(p, table, hidden, market, p.budget_usd) for p in by_rho[0.0]])
    checks.append(Check("V5 observability works", at_zero, f">= 0.08 at rho 0, and <= 0.02 at rho 1 (got {at_one:.3f})",
                        at_zero >= 0.08 and at_one <= 0.02))

    def share(price: float) -> float:
        bought = buyable = 0
        for p in mid:
            mine = {j.id for j in p.jobs if j.family in BUYABLE}
            bought += len(oracle_buys(p, table, priced(market, price), p.budget_usd) & mine)
            buyable += len(mine)
        return bought / buyable if buyable else 0.0
    drop = share(prices[0]) - share(prices[-1])
    checks.append(Check("V6 make or buy responds", drop, ">= 0.40 fall in oracle buying across the price sweep",
                        drop >= 0.40))

    modes = {}
    for f in families:
        for fold in (0, 1):
            answers = Counter(j.answer for j in pool if j.family == f and fold_of(j) != fold)
            modes[(f, fold)] = answers.most_common(1)[0][0] if answers else ""
    guess = _mean([score(p, {j.id: modes[(j.family, fold_of(j))] for j in p.jobs}) for p in mid])
    checks.append(Check("V7 no shortcut", guess, "<= 0.10 for the most common answer per family", guess <= 0.10))
    return checks
