"""The open-weight pilot's report (plan/2026-11-02-finetuning-scope.md, §9), from its outcome table: no model calls.

It answers one question: with smaller open models, does a job's hidden difficulty change which model should get it?
For each three-tier ladder of the calibrated open models it runs the validity checks, and it measures the value of
knowing the hidden bin out of fold, with each project's statistics taken from the other fold as an allocator gets
them (V5 measures it in sample, which round 2 showed can hide noise). Go means some ladder passes V3, V4 and V5 and
the out-of-fold value is positive: then a full round with more jobs per cell is worth running."""
import itertools
import json
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path

from purse.checks import TAU, Check
from purse.jobs import BINS, Job
from purse.market import OPEN, Option, Tiers
from purse.oracles import c_full, voi
from purse.projects import fold_of, make_project, with_budget
from purse.table import OutcomeTable, cell_stats

__all__ = ["CAP_TOKENS", "bin_rates", "cap_hits", "go", "ladders", "merge", "out_of_fold_voi"]

CAP_TOKENS = 8192  # the open models' max_output_tokens in bench/prices.json
CHARS_PER_TOKEN = 4  # a record has its cost, not its token counts: the input is estimated from the prompt
PROJECTS = 30  # per rho, as the checks draw them


def merge(sources: Iterable[Path], uids: set[str], out: Path) -> int:
    """Write the records of `uids` in `sources` (missing files skipped) to `out`, afresh, each (uid, option, repeat)
    once. Returns how many it wrote."""
    seen: dict[tuple[str, str, int], str] = {}
    for src in sources:
        if not src.exists():
            continue
        for line in src.read_text().splitlines():
            r = json.loads(line)
            if r["uid"] in uids:
                seen.setdefault((r["uid"], r["option"], r["repeat"]), line)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(x + "\n" for x in seen.values()))
    return len(seen)


def ladders(calibrated: set[str], options: Sequence[Option] = OPEN) -> list[Tiers]:
    """Every small, middle and top tier the calibrated open models can form, cheapest first (OPEN is in price
    order)."""
    names = [o.name for o in options if o.name in calibrated]
    return [Tiers(*trio) for trio in itertools.combinations(names, 3)]


def out_of_fold_voi(pool: Sequence[Job], table: OutcomeTable, market: Sequence[Option], rho: float = 0.0,
                    seed: int = 0) -> float:
    """Mean value of knowing each job's hidden bin at `rho`, budget TAU x C_full, with the statistics of the other
    fold (positive: an allocator that could read the bin would score higher)."""
    other = {f: cell_stats(table, [j for j in pool if fold_of(j) != f], hidden=True, market=market) for f in (0, 1)}
    values = []
    for i in range(PROJECTS):
        p = make_project(pool, i % 2, rho, seed * 10_000 + i, set())
        p = with_budget(p, TAU * c_full(p, table, market))
        values.append(voi(p, table, other[i % 2], market, p.budget_usd))
    return statistics.mean(values)


def go(checks: Sequence[Check], oof: float) -> bool:
    """Whether a full round is worth running on this ladder."""
    passed = {c.name.split()[0] for c in checks if c.passed}
    return {"V3", "V4", "V5"} <= passed and oof > 0


def bin_rates(table: OutcomeTable, pool: Sequence[Job], option: str) -> dict[tuple[str, int], dict[str, float]]:
    """(family, level) -> hidden bin -> success rate of `option`."""
    stats = cell_stats(table, pool, hidden=True)
    out: dict[tuple[str, int], dict[str, float]] = {}
    for (fam, level, hidden, opt), s in stats.items():
        if opt == option and hidden in BINS:
            out.setdefault((fam, level), {})[hidden] = s.rate
    return out


def cap_hits(table: OutcomeTable, pool: Sequence[Job], option: Option) -> int:
    """Attempts on `option` that wrote about CAP_TOKENS of output: a thinking model that never stopped, cut off."""
    hits = 0
    for j in pool:
        cost_in = len(j.prompt) / CHARS_PER_TOKEN * option.input_usd_per_m / 1e6
        for r in table.records(j.uid, option.name):
            out_tokens = (r.cost_usd - cost_in) / (option.output_usd_per_m / 1e6) if option.output_usd_per_m else 0
            hits += out_tokens >= 0.95 * CAP_TOKENS
    return hits
