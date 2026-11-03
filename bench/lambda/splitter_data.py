"""The trained splitter's data (plan/2026-11-02-finetuning-scope.md, §10), from round 2's table (no model calls):
train.jsonl and val.jsonl (prompt and the knapsack's split, from projects that share no job with the evaluation
projects) and eval.jsonl (round 2's 20 projects at each budget in the sweep). Also prints what the targets score in
replay, the ceiling a splitter that learns them exactly would reach.
Run from the repo root: PYTHONPATH=bench uv run python bench/lambda/splitter_data.py"""
import json
import statistics
from pathlib import Path
from typing import Any

from purse import splitter
from purse.cli import load_pool
from purse.market import calibrated, priced
from purse.oracles import c_full
from purse.projects import with_budget
from purse.runner import stats_for
from purse.solvers import plan_knapsack
from purse.table import OutcomeTable

R, OUT = Path("bench/out/r2"), Path("bench/out/splitter")
TRAIN, VAL, SEED = 3000, 200, 2026


def main() -> None:
    pool, table = load_pool(R / "pool.jsonl"), OutcomeTable(R / "table.jsonl")
    options = priced(calibrated(table.options()), float((R / "price").read_text()))
    held = splitter.eval_projects(pool, table, options)
    rows = splitter.training_rows(pool, table, options, held, TRAIN + VAL, SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", rows[:TRAIN]), ("val", rows[TRAIN:])):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    evals: list[dict[str, Any]] = []
    ceiling: dict[float, list[float]] = {}
    knap: dict[float, list[float]] = {}
    for p in held:
        fold = int(p.id.split("-f")[1][0])
        stats = stats_for(pool, table, options, fold, p.rho)
        for tau in splitter.TAUS:
            q = with_budget(p, tau * c_full(p, table, options))
            text = splitter.split_text(q, options, stats)
            own = splitter.own_cost(text, splitter.TARGET_TOKENS)
            split = splitter.target(q, options, stats, own)
            evals.append({"project": p.id, "tau": tau, "budget_usd": q.budget_usd, "prompt": text, "target": split})
            for seed in range(3):
                alloc = json.loads(split)
                ceiling.setdefault(tau, []).append(splitter.score_split(q, alloc, own, table, options, seed))
                knap.setdefault(tau, []).append(splitter.score_split(q, plan_knapsack(q, stats, options), 0.0, table,
                                                                     options, seed))
    (OUT / "eval.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evals))
    chars = [len(r["prompt"]) for r in rows]
    print(f"{len(rows[:TRAIN])} train, {len(rows[TRAIN:])} val, {len(evals)} eval prompts in {OUT}")
    print(f"prompt length: mean {statistics.mean(chars):.0f} chars, max {max(chars)} (about {max(chars) // 4} tokens)")
    print("tau   knapsack (free)   target, reply paid (a perfect splitter)")
    for tau in splitter.TAUS:
        print(f"{tau:<5} {statistics.mean(knap[tau]):.3f}             {statistics.mean(ceiling[tau]):.3f}")


if __name__ == "__main__":
    main()
