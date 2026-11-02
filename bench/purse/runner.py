"""One `purse run`: draw the condition's projects, give each its budget, run an allocator on each (replayed on the
outcome table, or live through a HASHIRU hub) for every seed, and write one result line per (project, seed)."""
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from purse import traces
from purse.jobs import Job
from purse.market import Market, Option
from purse.oracles import c_full, oracle_full
from purse.projects import Project, fold_of, make_project, project_text, score, with_budget
from purse.replay import ReplayMarket
from purse.solvers import (
    RunResult,
    cascade,
    fit_theta,
    fixed,
    hybrid,
    plan_knapsack,
    random_allocation,
    run_upfront,
    uniform,
)
from purse.table import OutcomeTable, Stats, cell_stats, visible_stats

__all__ = ["LIVE", "REPLAYED", "RunConfig", "is_live", "make_projects", "run", "stats_for"]

REPLAYED = ("knapsack", "cascade", "hybrid", "uniform", "random", "oracle")  # and fixed:<option>
LIVE = ("hashiru", "hashiru-names-models", "llm-split")


@dataclass(frozen=True)
class RunConfig:
    allocator: str
    rho: float = 0.5
    tau: float = 0.5
    projects: int = 10
    seeds: int = 3
    memory_mb: int = 12288
    ceo: str = "gpt-6-luna"
    informed: bool = True
    live: bool = False
    max_usd: float = 2.0
    out: Path = Path("bench/out")
    peers: Mapping[str, str] = field(default_factory=dict)
    port: int = 9250


def is_live(cfg: RunConfig) -> bool:
    return cfg.allocator in LIVE or cfg.live


def make_projects(pool: Sequence[Job], table: OutcomeTable, market: Sequence[Option], cfg: RunConfig) -> list[Project]:
    """The condition's projects (folds alternate; no job twice within a rho), each with budget tau x C_full."""
    used: set[str] = set()
    out = []
    for i in range(cfg.projects):
        p = replace(make_project(pool, i % 2, cfg.rho, seed=i, used=used), memory_mb=cfg.memory_mb)
        out.append(with_budget(p, cfg.tau * c_full(p, table, market)))
    return out


def stats_for(pool: Sequence[Job], table: OutcomeTable, market: Sequence[Option], fold: int, rho: float) -> Stats:
    """What an informed allocator is told for a project in `fold`: the other fold's cells, mixed by rho."""
    return visible_stats(cell_stats(table, [j for j in pool if fold_of(j) != fold], hidden=True, market=market), rho)


async def _replayed(cfg: RunConfig, p: Project, stats: Stats, market: Market, options: Sequence[Option],
                    table: OutcomeTable, seed: int, theta: float) -> RunResult:
    name = cfg.allocator
    if name == "cascade":
        return await cascade(p, stats, market, options, theta)
    alloc = (plan_knapsack(p, stats, options) if name == "knapsack" else hybrid(p) if name == "hybrid"
             else uniform(p, stats, options) if name == "uniform" else random_allocation(p, options, seed)
             if name == "random" else oracle_full(p, table, options, p.budget_usd) if name == "oracle"
             else fixed(p, name.removeprefix("fixed:"), options))
    return await run_upfront(p, alloc, market, options)


def _line(cfg: RunConfig, p: Project, seed: int, result: RunResult, **extra: Any) -> dict[str, Any]:
    return {"allocator": cfg.allocator, "project": p.id, "seed": seed, "rho": cfg.rho, "tau": cfg.tau,
            "budget_usd": p.budget_usd, "score": score(p, result.answers), "spent_usd": result.spent_usd,
            "allocator_usd": result.allocator_usd,
            "attempts": [[a.job_id, a.option, a.outcome.correct, a.outcome.cost_usd, a.outcome.refused]
                         for a in result.attempts]} | extra


LiveRunner = Callable[[RunConfig, Project, str, Sequence[Option], int], Any]  # async -> (extras, RunResult)


async def run(cfg: RunConfig, pool: Sequence[Job], table: OutcomeTable, options: Sequence[Option],
              live: LiveRunner | None = None, live_market: Callable[[Project], Any] | None = None) -> list[dict]:
    """Every (project, seed) of the condition; results are appended to cfg.out/results.jsonl as they finish."""
    projects = make_projects(pool, table, options, cfg)
    stats = {fold: stats_for(pool, table, options, fold, cfg.rho) for fold in (0, 1)}
    thetas: dict[int, float] = {}
    if cfg.allocator == "cascade":
        for fold in (0, 1):
            others = [p for p in projects if int(p.id.split("-f")[1][0]) != fold]
            thetas[fold] = await fit_theta(others, stats[1 - fold], ReplayMarket(table, options), options)
    cfg.out.mkdir(parents=True, exist_ok=True)
    lines = []
    for p in projects:
        fold = int(p.id.split("-f")[1][0])
        for seed in range(cfg.seeds):
            if cfg.allocator in LIVE:
                assert live is not None
                text = project_text(p, options, stats[fold] if cfg.informed else None)
                extra, result = await live(cfg, p, text, options, seed)
                line = _line(cfg, p, seed, result, **extra)
            elif cfg.live:
                assert live_market is not None
                async with live_market(p) as market:
                    line = _line(cfg, p, seed, await _replayed(cfg, p, stats[fold], market, options, table, seed,
                                                               thetas.get(fold, 0.7)), live=True)
            else:
                market = ReplayMarket(table, options, draw=seed)
                line = _line(cfg, p, seed, await _replayed(cfg, p, stats[fold], market, options, table, seed,
                                                           thetas.get(fold, 0.7)))
            lines.append(line)
            with (cfg.out / "results.jsonl").open("a") as f:
                f.write(json.dumps(line) + "\n")
    return lines


async def live_extras(p: Project, trace: Path, table: OutcomeTable, options: Sequence[Option],
                      seed: int) -> dict[str, Any]:
    """From a live HASHIRU run's trace: what it tried per job, declared difficulties, fidelity, refunds, and its
    decisions replayed on the table."""
    events = traces.read(trace)
    tried = traces.live_allocation(events, p)
    replayed = await traces.replay_live(p, tried, ReplayMarket(table, options, draw=seed), options)
    return {"trace": str(trace), "tried": tried, "declared": traces.declared(events, p),
            "hidden": {j.id: j.hidden for j in p.jobs}, "prompt_fidelity": traces.prompt_fidelity(events, p),
            "slack_usd": traces.slack(events), "replay_score": score(p, replayed.answers)}
