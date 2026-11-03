"""The trained splitter (plan/2026-11-02-finetuning-scope.md, §10): a model that reads a project's outline and writes,
in one reply, which option does each job, trained to give the knapsack's split. Here: the prompt, the training target,
reading a reply, training projects that share no job with the evaluation projects, and replay scoring that charges
the reply to the project's budget. No model calls."""
import asyncio
import json
import random
from collections.abc import Sequence
from typing import Any

from purse.jobs import Job
from purse.market import OPEN, Option, lookup, serves
from purse.oracles import c_full
from purse.projects import Project, _describe, make_project, parse_answers, score, with_budget
from purse.replay import ReplayMarket
from purse.runner import RunConfig, make_projects, stats_for
from purse.solvers import SKIP, Allocation, plan_knapsack, run_upfront
from purse.table import OutcomeTable, Stats

__all__ = ["PREVIEW_CHARS", "SPLITTER", "TARGET_TOKENS", "eval_projects", "own_cost", "parse_split", "reply_cost",
           "score_reply", "score_split", "split_text", "target", "training_rows"]

SPLITTER = "hosted_vllm/Qwen/Qwen3.5-9B"  # the model trained; its reply is charged at its list price
PREVIEW_CHARS = 160  # how much of a job's text the splitter sees (the knapsack reads none of it)
TARGET_TOKENS = 120  # about how long a split of eight jobs is, for the cost a target sets aside
CHARS_PER_TOKEN = 4
TAUS = (0.25, 0.5, 1.0, 2.0)  # budgets as multiples of C_full, as in round 2's sweep
RHOS = (0.0, 0.5, 1.0)
INSTRUCTION = ('Plan who does each job. Reply with only a JSON object that maps every job id to the name of one option '
               'above, or to "skip" to leave the job unanswered, e.g. {"j1": "gpt-6-luna", "j2": "skip"}. Each job '
               "then gets one attempt on its option, the most valuable jobs first, and an attempt that the money left "
               "cannot pay for is cut off and answers nothing.")


def _preview(prompt: str) -> str:
    flat = " ".join(prompt.split())
    return flat if len(flat) <= PREVIEW_CHARS else flat[: PREVIEW_CHARS - 1] + "…"


def split_text(p: Project, options: Sequence[Option], stats: Stats) -> str:
    """What the splitter reads: the budget, the market, the statistics an informed allocator is told, and each job's
    heading with the start of its text."""
    lines = [f"You have ${p.budget_usd:.4f} for this project. It pays for this reply first, then for every attempt "
             "you plan. Each job is worth its value in points; plan to earn as many as you can.", ""]
    lines += ["Who can do a job:", *(_describe(o) for o in options), ""]
    lines += ["Success rate and mean cost per attempt, measured on similar jobs:", "",
              "| family | level | option | success | mean cost |", "|---|---|---|---|---|"]
    lines += [f"| {f} | {lvl} | {o} | {s.rate:.0%} | ${s.cost_usd:.4f} |"
              for (f, lvl, _, o), s in sorted(stats.items())]
    lines += ["", INSTRUCTION, ""]
    for j in p.jobs:
        lines += [f"### {j.id} ({j.family}, level {j.level}, value {j.value})", _preview(j.prompt), ""]
    return "\n".join(lines).rstrip() + "\n"


def own_cost(text: str, out_tokens: int) -> float:
    """What the splitter's reply costs at its list price: reading `text` and writing `out_tokens`."""
    o = lookup(OPEN, SPLITTER)
    return (len(text) / CHARS_PER_TOKEN * o.input_usd_per_m + out_tokens * o.output_usd_per_m) / 1e6


def target(p: Project, options: Sequence[Option], stats: Stats, own_usd: float) -> str:
    """The knapsack's split of what the reply leaves, as the JSON the splitter should write."""
    return json.dumps(plan_knapsack(with_budget(p, max(0.0, p.budget_usd - own_usd)), stats, options))


def parse_split(reply: str, p: Project, options: Sequence[Option]) -> tuple[Allocation, int]:
    """The allocation in `reply`, and how many named options were not in the market or cannot do their job. A job
    the reply leaves out, or gives an unusable option, is skipped."""
    named = parse_answers(reply)
    by_name = {o.name: o for o in options}
    alloc, bad = {}, 0
    for j in p.jobs:
        name = str(named.get(j.id, SKIP)).strip()
        if name != SKIP and (name not in by_name or not serves(by_name[name], j)):
            bad, name = bad + 1, SKIP
        alloc[j.id] = name
    return alloc, bad


def eval_projects(pool: Sequence[Job], table: OutcomeTable, options: Sequence[Option], n: int = 20) -> list[Project]:
    """Round 2's projects (rho 0.5, tau 0.5): those the gpt-6-luna split and every replayed allocator were scored on."""
    return make_projects(pool, table, options, RunConfig("splitter", projects=n))


def training_rows(pool: Sequence[Job], table: OutcomeTable, options: Sequence[Option], held: Sequence[Project],
                  n: int, seed: int) -> list[dict[str, Any]]:
    """`n` distinct training examples from projects drawn without any job of `held`, at random budgets (TAUS) and
    rhos (RHOS), each the prompt and the knapsack's split."""
    held_uids = {j.uid for p in held for j in p.jobs}
    free = [j for j in pool if j.uid not in held_uids]
    rng = random.Random(seed)
    stats = {(f, rho): stats_for(pool, table, options, f, rho) for f in (0, 1) for rho in RHOS}
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _ in range(50 * n):
        if len(rows) == n:
            break
        fold, rho, tau = rng.randrange(2), rng.choice(RHOS), rng.choice(TAUS)
        try:
            p = make_project(free, fold, rho, rng.randrange(10**9), set())
        except ValueError:  # this fold has too few free jobs of some family
            continue
        p = with_budget(p, tau * c_full(p, table, options))
        text = split_text(p, options, stats[(fold, rho)])
        if p.budget_usd <= 0 or text in seen:
            continue
        seen.add(text)
        rows.append({"prompt": text, "completion": target(p, options, stats[(fold, rho)],
                                                         own_cost(text, TARGET_TOKENS)),
                     "uids": [j.uid for j in p.jobs], "fold": fold, "rho": rho, "tau": tau})
    return rows


def score_split(p: Project, alloc: Allocation, own_usd: float, table: OutcomeTable, options: Sequence[Option],
                seed: int) -> float:
    """The split carried out in replay under v2's pool rule, with the splitter's reply paid from the budget first."""
    if own_usd < 0:
        raise ValueError(f"the reply cannot cost less than nothing: {own_usd}")
    left = with_budget(p, max(0.0, p.budget_usd - own_usd))
    result = asyncio.run(run_upfront(left, alloc, ReplayMarket(table, options, draw=seed), options, bounded=True))
    return score(p, result.answers)


def reply_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """What a reply the server counted cost at the splitter's list price."""
    o = lookup(OPEN, SPLITTER)
    return (prompt_tokens * o.input_usd_per_m + completion_tokens * o.output_usd_per_m) / 1e6


def score_reply(p: Project, reply: str, prompt_tokens: int, completion_tokens: int, table: OutcomeTable,
                options: Sequence[Option], seeds: Sequence[int] = (0, 1, 2)) -> dict[str, Any]:
    """A served splitter's reply: its allocation, unusable names, cost, and mean replay score over `seeds`."""
    alloc, bad = parse_split(reply, p, options)
    own = reply_cost(prompt_tokens, completion_tokens)
    scores = [score_split(p, alloc, own, table, options, s) for s in seeds]
    return {"alloc": alloc, "bad": bad, "own_usd": own, "score": sum(scores) / len(scores)}
