"""Allocators that are not LLMs: a knapsack that splits the money up front, fixed policies, and a cascade that
adapts during the task. All of them run against a Market, live or replayed, under the same pool rule as the hub."""
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from purse.jobs import Job, extract_answer, family
from purse.market import Market, Option, Outcome, lookup, serves
from purse.projects import Project, score
from purse.table import Stats

__all__ = ["SKIP", "UNIT", "Allocation", "Attempt", "Pool", "RunResult", "cascade", "fit_theta", "fixed", "hybrid",
           "knapsack", "plan_knapsack", "random_allocation", "run_upfront", "uniform"]

Allocation = dict[str, str]  # job id -> option name, or SKIP
SKIP = "skip"
UNIT = 1e-4  # the knapsack counts dollars in units of $0.0001, costs rounded up
MAX_CHECKS = 3  # grid_check answers at most this many times per puzzle
HYBRID = {1: "ollama_chat/qwen3.5:4b", 2: "ollama_chat/qwen3.5:4b", 3: "gpt-6-luna", 4: "gpt-6-luna",
          5: "meta/muse-spark-1.3"}  # AgentCARD-style: the tier follows the visible level


class Pool:
    """The allocator's money, under the hub's rule: a cloud call needs money left, an outside agent its price, and
    a call's cost is charged in full afterwards (it may take the pool below zero, as in the hub)."""

    def __init__(self, budget_usd: float) -> None:
        self.budget_usd, self.spent_usd = budget_usd, 0.0

    @property
    def left(self) -> float:
        return self.budget_usd - self.spent_usd

    def allows(self, o: Option) -> bool:
        return o.kind == "local" or (self.left > 0 if o.kind == "cloud" else self.left >= o.per_message_usd)

    def charge(self, usd: float) -> None:
        self.spent_usd += usd


@dataclass(frozen=True)
class Attempt:
    job_id: str
    option: str
    outcome: Outcome


@dataclass(frozen=True)
class RunResult:
    answers: dict[str, str]
    spent_usd: float
    attempts: tuple[Attempt, ...]
    allocator_usd: float = 0.0  # what the allocator spent on its own model calls


def knapsack(items: Sequence[tuple[str, Sequence[tuple[str, float, float]]]], budget_usd: float) -> Allocation:
    """Multiple-choice knapsack: one (option, gain, cost) per job, or SKIP, for the most gain within the budget."""
    best: dict[int, tuple[float, Allocation]] = {0: (0.0, {})}
    for job_id, choices in items:
        nxt: dict[int, tuple[float, Allocation]] = {}
        for spent, (gain, alloc) in best.items():
            for option, g, c in [(SKIP, 0.0, 0.0), *choices]:
                s = spent + math.ceil(c / UNIT - 1e-9)
                if s * UNIT <= budget_usd + 1e-12:
                    cand = (gain + g, alloc | {job_id: option})
                    if s not in nxt or cand[0] > nxt[s][0] + 1e-12:
                        nxt[s] = cand
        best = nxt
    return max(best.items(), key=lambda kv: (kv[1][0], -kv[0]))[1][1]


def _cell(stats: Stats, j: Job, option: str, hidden: bool) -> tuple[float, float] | None:
    s = stats.get((j.family, j.level, j.hidden if hidden else None, option))
    return (s.rate, s.cost_usd) if s else None


def plan_knapsack(p: Project, stats: Stats, market: Sequence[Option], hidden: bool = False,
                  margin: float = 0.9) -> Allocation:
    """The solver's up-front split: expected points under `margin` of the budget, from cell statistics."""
    items = []
    for j in p.jobs:
        choices = [(o.name, j.value * c[0], c[1]) for o in market if serves(o, j)
                   if (c := _cell(stats, j, o.name, hidden)) is not None]
        items.append((j.id, choices))
    return knapsack(items, margin * p.budget_usd)


def fixed(p: Project, option: str, market: Sequence[Option] = ()) -> Allocation:
    o = next((x for x in market if x.name == option), None)
    return {j.id: option if o is None or serves(o, j) else SKIP for j in p.jobs}


def hybrid(p: Project) -> Allocation:
    return {j.id: HYBRID[j.level] for j in p.jobs}


def uniform(p: Project, stats: Stats, market: Sequence[Option]) -> Allocation:
    """Each job the most reliable option whose mean cost fits an equal share of the budget."""
    share, alloc = p.budget_usd / max(len(p.jobs), 1), {}
    for j in p.jobs:
        fits = [(c[0], -c[1], o.name) for o in market if serves(o, j)
                if (c := _cell(stats, j, o.name, False)) is not None and c[1] <= share]
        alloc[j.id] = max(fits)[2] if fits else SKIP
    return alloc


def random_allocation(p: Project, market: Sequence[Option], seed: int) -> Allocation:
    rng = random.Random(seed)
    return {j.id: rng.choice([o.name for o in market if serves(o, j)]) for j in p.jobs}


async def _try(j: Job, o: Option, pool: Pool, market: Market, log: list[Attempt]) -> Outcome:
    if not pool.allows(o):
        out = Outcome("", False, 0.0, 0.0, refused="the pool cannot pay for it")
    else:
        out = await market.attempt(j, o.name)
        pool.charge(out.cost_usd)
    log.append(Attempt(j.id, o.name, out))
    return out


async def run_upfront(p: Project, alloc: Mapping[str, str], market: Market, options: Sequence[Option]) -> RunResult:
    """Carry out a fixed allocation: jobs by value (then id), each once, nothing changed on the way."""
    pool, log, answers = Pool(p.budget_usd), list[Attempt](), dict[str, str]()
    for j in sorted(p.jobs, key=lambda x: (-x.value, x.id)):
        if (name := alloc.get(j.id, SKIP)) == SKIP:
            continue
        out = await _try(j, lookup(options, name), pool, market, log)
        if not out.refused:
            answers[j.id] = out.answer
    return RunResult(answers, pool.spent_usd, tuple(log))


async def cascade(p: Project, stats: Stats, market: Market, options: Sequence[Option], theta: float) -> RunResult:
    """The solver that adapts: per job, the options expected to succeed at least `theta` of the time, cheapest first.
    A checkable job moves up until the check passes (at most MAX_CHECKS tries); any other job asks the two cheapest
    and, if they disagree, a third. Without a qualifying option, the most reliable one answers alone."""
    pool, log, answers = Pool(p.budget_usd), list[Attempt](), dict[str, str]()

    def ranked(j: Job) -> list[tuple[Option, float, float]]:
        known = [(o, *c) for o in options if serves(o, j) if (c := _cell(stats, j, o.name, False)) is not None]
        good = sorted((x for x in known if x[1] >= theta), key=lambda x: (x[2], -x[1]))
        return good or sorted(known, key=lambda x: -x[1])[:1]

    def per_cost(j: Job) -> float:
        r = ranked(j)
        return j.value / max(r[0][2], UNIT) if r else 0.0

    for j in sorted(p.jobs, key=lambda x: (-per_cost(x), x.id)):
        route = [o for o, _, _ in ranked(j)]
        if family(j.family).checkable:
            for o in route[:MAX_CHECKS]:
                out = await _try(j, o, pool, market, log)
                if not out.refused:
                    answers[j.id] = out.answer
                    if family(j.family).correct(j.answer, out.answer):
                        break
            continue
        replies = []
        for o in route[:2]:
            out = await _try(j, o, pool, market, log)
            if not out.refused:
                replies.append(out.answer)
        if len(replies) == 2 and extract_answer(replies[0]) != extract_answer(replies[1]) and len(route) > 2:
            out = await _try(j, route[2], pool, market, log)
            if not out.refused:
                replies.append(out.answer)
        if replies:
            answers[j.id] = replies[-1]
    return RunResult(answers, pool.spent_usd, tuple(log))


async def fit_theta(projects: Sequence[Project], stats: Stats, market: Market, options: Sequence[Option],
                    grid: Sequence[float] = (0.3, 0.5, 0.7, 0.9)) -> float:
    """The threshold with the best mean score on `projects` (the other fold's), the smallest on a tie."""
    best, best_score = grid[0], -1.0
    for theta in grid:
        results = [await cascade(p, stats, market, options, theta) for p in projects]
        mean = sum(score(p, r.answers) for p, r in zip(projects, results, strict=True)) / max(len(projects), 1)
        if mean > best_score + 1e-12:
            best, best_score = theta, mean
    return best
