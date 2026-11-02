"""What the best possible split would score, from the outcome table: oracle-full knows each job's recorded rates;
the cell oracles know only cell statistics, with or without the hidden bins (their gap is the value of information).
Also the outside agents' price sweep and what oracle-full buys at each price."""
import statistics
from collections.abc import Mapping, Sequence

from purse.jobs import Job
from purse.market import Option, lookup, serves
from purse.projects import Project
from purse.solvers import SKIP, Allocation, knapsack
from purse.table import OutcomeTable, Stats, visible_stats

__all__ = ["MULTIPLIERS", "c_full", "expected_cost", "expected_score", "oracle_buys", "oracle_cells", "oracle_full",
           "sweep_prices", "truth", "voi"]

MULTIPLIERS = (0.25, 1.0, 4.0, 16.0)  # the outside agents' prices, as multiples of c_ref
BUYABLE = ("arithmetic", "grids")


def truth(table: OutcomeTable, job: Job, option: Option) -> tuple[float, float] | None:
    """A job's recorded success rate on an option, and its mean cost (an outside agent's is its current price)."""
    rate = table.rate(job.uid, option.name)
    if rate is None:
        return None
    cost = table.cost(job.uid, option.name) or 0.0
    return rate, option.per_message_usd if option.kind == "agent" else cost


Items = list[tuple[str, list[tuple[str, float, float]]]]


def _items(p: Project, table: OutcomeTable, market: Sequence[Option]) -> Items:
    return [(j.id, [(o.name, j.value * t[0], t[1]) for o in market if serves(o, j)
                    if (t := truth(table, j, o)) is not None]) for j in p.jobs]


def expected_score(p: Project, alloc: Mapping[str, str], table: OutcomeTable, market: Sequence[Option]) -> float:
    total, won = sum(j.value for j in p.jobs), 0.0
    for j in p.jobs:
        if (name := alloc.get(j.id, SKIP)) != SKIP and (t := truth(table, j, lookup(market, name))) is not None:
            won += j.value * t[0]
    return won / total if total else 0.0


def expected_cost(p: Project, alloc: Mapping[str, str], table: OutcomeTable, market: Sequence[Option]) -> float:
    return sum(t[1] for j in p.jobs if (name := alloc.get(j.id, SKIP)) != SKIP
               if (t := truth(table, j, lookup(market, name))) is not None)


def oracle_full(p: Project, table: OutcomeTable, market: Sequence[Option], budget_usd: float) -> Allocation:
    return knapsack(_items(p, table, market), budget_usd)


def c_full(p: Project, table: OutcomeTable, market: Sequence[Option]) -> float:
    """What oracle-full spends at its best score with no budget: per job the most reliable option, then the
    cheapest of those."""
    total = 0.0
    for _, choices in _items(p, table, market):
        if choices:
            total += max(choices, key=lambda c: (c[1], -c[2]))[2]
    return total


def oracle_cells(p: Project, stats: Stats, market: Sequence[Option], budget_usd: float, hidden: bool) -> Allocation:
    items = []
    for j in p.jobs:
        choices = []
        for o in market:
            s = stats.get((j.family, j.level, j.hidden if hidden else None, o.name))
            if serves(o, j) and s is not None:
                choices.append((o.name, j.value * s.rate, o.per_message_usd if o.kind == "agent" else s.cost_usd))
        items.append((j.id, choices))
    return knapsack(items, budget_usd)


def voi(p: Project, table: OutcomeTable, hidden: Stats, market: Sequence[Option], budget_usd: float) -> float:
    """Expected score of the best split knowing each job's hidden bin, minus knowing only what projects of this rho
    reveal (both judged on the jobs' recorded rates)."""
    seen = visible_stats(hidden, p.rho)
    return (expected_score(p, oracle_cells(p, hidden, market, budget_usd, True), table, market)
            - expected_score(p, oracle_cells(p, seen, market, budget_usd, False), table, market))


def sweep_prices(table: OutcomeTable, pool: Sequence[Job], market: Sequence[Option]) -> tuple[float, ...]:
    """c_ref x MULTIPLIERS, where c_ref is the median cost of gpt-6-luna on the jobs an outside agent can do."""
    costs = [c for j in pool if j.family in BUYABLE if (c := table.cost(j.uid, "gpt-6-luna")) is not None]
    if not costs:
        raise ValueError("no gpt-6-luna records for arithmetic or grid jobs: calibrate it first")
    c_ref = statistics.median(costs)
    return tuple(c_ref * m for m in MULTIPLIERS)


def oracle_buys(p: Project, table: OutcomeTable, market: Sequence[Option], budget_usd: float) -> set[str]:
    """The jobs oracle-full sends to an outside agent: rational demand at the market's prices."""
    return {jid for jid, name in oracle_full(p, table, market, budget_usd).items()
            if name != SKIP and lookup(market, name).kind == "agent"}
