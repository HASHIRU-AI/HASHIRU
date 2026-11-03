"""Score splitters on round 2's 20 projects at every budget in the sweep (plan/2026-11-02-finetuning-scope.md, §10).

`ask` sends each prompt in bench/out/splitter/eval.jsonl to a served model (vLLM over the SSH tunnel) at temperature 0
and saves the replies with their token counts; a rerun asks only what is missing. `report` scores every saved variant
in replay against round 2's table, with each reply paid at the splitter's list price, beside the knapsack and the
stored gpt-6-luna split (no model calls).

  PYTHONPATH=bench uv run python bench/lambda/eval_splitter.py ask base-nothink Qwen/Qwen3.5-9B
  PYTHONPATH=bench uv run python bench/lambda/eval_splitter.py ask base-think Qwen/Qwen3.5-9B --think
  PYTHONPATH=bench uv run python bench/lambda/eval_splitter.py ask tuned splitter
  PYTHONPATH=bench uv run python bench/lambda/eval_splitter.py report"""
import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from purse import metrics, splitter
from purse.cli import load_pool
from purse.market import calibrated, priced
from purse.oracles import c_full
from purse.projects import with_budget
from purse.runner import stats_for
from purse.solvers import plan_knapsack
from purse.table import OutcomeTable

R, OUT = Path("bench/out/r2"), Path("bench/out/splitter")
URL = "http://127.0.0.1:18000/v1"
CONCURRENCY = 40


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


async def ask(variant: str, model: str, think: bool, url: str, max_tokens: int) -> None:
    rows, dest = read(OUT / "eval.jsonl"), OUT / f"replies-{variant}.jsonl"
    done = {(r["project"], r["tau"]) for r in read(dest)}
    todo = [r for r in rows if (r["project"], r["tau"]) not in done]
    client, gate = AsyncOpenAI(base_url=url, api_key="unused", timeout=1800), asyncio.Semaphore(CONCURRENCY)

    async def one(r: dict[str, Any]) -> None:
        async with gate:
            resp = await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": r["prompt"]}], temperature=0,
                max_tokens=max_tokens, extra_body={"chat_template_kwargs": {"enable_thinking": think}})
        if resp.usage is None:
            raise RuntimeError(f"the server counted no tokens for {r['project']} at tau {r['tau']}")
        line = {"project": r["project"], "tau": r["tau"], "reply": resp.choices[0].message.content or "",
                "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                "finish": resp.choices[0].finish_reason}
        with dest.open("a") as f:
            f.write(json.dumps(line) + "\n")
    await asyncio.gather(*(one(r) for r in todo))
    print(f"{variant}: {len(todo)} asked, {len(read(dest))} replies in {dest}")


def report() -> None:
    pool, table = load_pool(R / "pool.jsonl"), OutcomeTable(R / "table.jsonl")
    options = priced(calibrated(table.options()), float((R / "price").read_text()))
    held = {p.id: p for p in splitter.eval_projects(pool, table, options)}
    stats = {f: stats_for(pool, table, options, f, 0.5) for f in (0, 1)}
    targets = {(r["project"], r["tau"]): json.loads(r["target"]) for r in read(OUT / "eval.jsonl")}

    def project(pid: str, tau: float) -> Any:
        p = held[pid]
        return with_budget(p, tau * c_full(p, table, options))

    scores: dict[str, dict[tuple[str, float], float]] = {"knapsack (free)": {}}
    for pid, tau in targets:
        q = project(pid, tau)
        alloc = plan_knapsack(q, stats[int(pid.split("-f")[1][0])], options)
        scores["knapsack (free)"][(pid, tau)] = statistics.mean(
            splitter.score_split(q, alloc, 0.0, table, options, s) for s in range(3))
    extra: dict[str, str] = {}
    for path in sorted(OUT.glob("replies-*.jsonl")):
        variant, mine, bad, agree, cost, cut = path.stem.removeprefix("replies-"), {}, 0, 0, [], 0
        for r in read(path):
            key = (r["project"], r["tau"])
            out = splitter.score_reply(project(*key), r["reply"], r["prompt_tokens"], r["completion_tokens"], table,
                                       options)
            mine[key] = out["score"]
            bad += out["bad"]
            agree += sum(out["alloc"][j] == v for j, v in targets[key].items())
            cost.append(out["own_usd"] / project(*key).budget_usd)
            cut += r["finish"] == "length"
        scores[variant] = mine
        extra[variant] = (f"agrees with the knapsack on {agree / (8 * len(mine)):.0%} of jobs; unusable names {bad}; "
                          f"reply {statistics.mean(cost):.1%} of the budget; cut at max_tokens {cut}")
    luna: dict[str, list[float]] = {r["project"]: [] for r in read(R / "runs/llm-split/results.jsonl")}
    for r in read(R / "runs/llm-split/results.jsonl"):
        luna[r["project"]].append(r["score"])
    print("variant".ljust(28) + "".join(f"tau {t:<7}" for t in splitter.TAUS))
    for name, s in scores.items():
        cells = [statistics.mean(v for (_, t), v in s.items() if t == tau) if any(t == tau for _, t in s) else None
                 for tau in splitter.TAUS]
        print(name.ljust(28) + "".join(f"{c:<11.3f}" if c is not None else "-".ljust(11) for c in cells))
    stored = statistics.mean(map(statistics.mean, luna.values()))
    print("gpt-6-luna split (stored)".ljust(28) + "-".ljust(11) + f"{stored:<11.3f}")
    for name, note in extra.items():
        print(f"{name}: {note}")
    if "tuned" in scores:
        for other in [n for n in scores if n != "tuned"]:
            for tau in splitter.TAUS:
                keys = sorted(k for k in scores["tuned"] if k[1] == tau and k in scores[other])
                if len(keys) > 1:
                    a, b = [scores["tuned"][k] for k in keys], [scores[other][k] for k in keys]
                    lo, hi = metrics.paired_ci(a, b)
                    print(f"tuned minus {other}, tau {tau}: {statistics.mean(a) - statistics.mean(b):+.3f} "
                          f"[{lo:+.3f}, {hi:+.3f}] n={len(keys)}")
        keys = sorted(k for k in scores["tuned"] if k[1] == 0.5 and k[0] in luna)
        a, b = [scores["tuned"][k] for k in keys], [statistics.mean(luna[k[0]]) for k in keys]
        if len(keys) > 1:
            lo, hi = metrics.paired_ci(a, b)
            print(f"tuned minus gpt-6-luna split, tau 0.5: {statistics.mean(a) - statistics.mean(b):+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] n={len(keys)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("ask")
    a.add_argument("variant")
    a.add_argument("model", help="the name the server serves the model under")
    a.add_argument("--think", action="store_true", help="let the model think before it answers")
    a.add_argument("--url", default=URL)
    a.add_argument("--max-tokens", type=int, default=0, help="default: 512, or 16384 with --think")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "ask":
        asyncio.run(ask(args.variant, args.model, args.think, args.url, args.max_tokens or (16384 if args.think
                                                                                          else 512)))
    else:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
