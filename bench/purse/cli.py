"""The purse command. `pool` writes the jobs; `calibrate` sends each job to each option through a HASHIRU hub and
records the outcome table; `check` runs the validity checks on it; `run` runs an allocator on a condition's projects.
Commands that can reach a real model refuse unless --spend is given.

    PYTHONPATH=bench uv run python -m purse pool --out bench/out/pool.jsonl
    PYTHONPATH=bench uv run --env-file .env python -m purse calibrate --pool ... --table ... \\
        --options gpt-6-luna --repeats 2 --max-usd 5 --spend
    PYTHONPATH=bench uv run python -m purse check --pool ... --table ...
    PYTHONPATH=bench uv run python -m purse run --allocator knapsack --pool ... --table ... --rho 0.5 \\
        --tau 0.5 --out bench/out/k
"""
import argparse
import asyncio
import contextlib
import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from hashiru.config import Settings
from hashiru.hub import Hub
from hashiru.llm import LLM, Wallet, WalletExceeded
from purse.checks import run_checks
from purse.jobs import LEVELS, Job
from purse.market import CEO_MODEL, MARKET, Market, Option, calibrated, lookup, priced, serves
from purse.oracles import sweep_prices
from purse.projects import Project, build_pool, with_budget
from purse.replay import ReplayMarket
from purse.runner import LIVE, RunConfig, is_live, live_extras, run
from purse.solvers import RunResult, run_upfront
from purse.table import OutcomeTable, Record

__all__ = ["HubFactory", "calibrate", "environment", "load_pool", "main", "planned", "wallet_hit"]

ROOT = Path(__file__).resolve().parents[2]
RESERVE_USD = 0.25  # what each calibration hire reserves (firing refunds the rest)


def load_pool(path: Path) -> list[Job]:
    jobs = []
    for line in path.read_text().splitlines():
        d = json.loads(line)
        jobs.append(Job(**(d | {"features": tuple(tuple(f) for f in d["features"])})))
    return jobs


def planned(jobs: Sequence[Job], options: Sequence[str], repeats: int, table: OutcomeTable,
            market: Sequence[Option]) -> list[tuple[Job, str, int]]:
    """Every (job, option, repeat) the table does not have yet, for options that serve the job."""
    return [(j, o, r) for j in jobs for o in options if serves(lookup(market, o), j) for r in range(repeats)
            if not table.has(j.uid, o, r)]


async def calibrate(todo: Sequence[tuple[Job, str, int]], table: OutcomeTable, market: Market) -> int:
    """Attempt each and record it as it finishes. A refusal is recorded with its reason; an exception (a provider
    down, max_usd reached, Ctrl-C) stops the run, and a rerun resumes after the last record."""
    for j, option, repeat in todo:
        out = await market.attempt(j, option)
        table.add(Record(j.uid, option, repeat, out.answer, out.correct, out.cost_usd, out.seconds, out.refused))
    return len(todo)


def environment(argv: Sequence[str]) -> dict[str, str]:
    """What a result needs to be reproduced; never a key."""
    def run_git() -> str:
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"
    try:
        ollama = httpx.get("http://127.0.0.1:11434/api/version", timeout=1).json().get("version", "unknown")
    except (httpx.HTTPError, ValueError):
        ollama = "unreachable"
    catalog = ROOT / "model_catalog.json"
    lock = ROOT / "uv.lock"
    return {"git": run_git(), "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else "",
            "python": sys.version.split()[0], "platform": platform.platform(), "ollama": ollama,
            "price_map": datetime.fromtimestamp(catalog.stat().st_mtime, UTC).isoformat() if catalog.exists()
            else "bundled", "argv": " ".join(argv), "at": datetime.now(UTC).isoformat()}


def _real_hub(settings: Settings, wallet: Wallet) -> Hub:
    return Hub(settings, llm=LLM(settings.ollama_url, wallet, settings.ollama_num_ctx), wallet=wallet)


class HubFactory:
    """Builds every hub of one command around one wallet, so --max-usd caps the whole batch, not each project."""

    def __init__(self, max_usd: float, make: Callable[[Settings, Wallet], Hub] | None = None) -> None:
        self.wallet = Wallet(max_usd)
        self._make = make or _real_hub

    def __call__(self, settings: Settings) -> Hub:
        return self._make(settings, self.wallet)


def wallet_hit(exc: BaseException) -> bool:
    """Whether `exc` is, or wraps, the hub's WalletExceeded (task groups wrap what their tasks raise)."""
    if isinstance(exc, WalletExceeded):
        return True
    return isinstance(exc, BaseExceptionGroup) and any(wallet_hit(e) for e in exc.exceptions)


def _stopped(exc: BaseException, factory: HubFactory, where: str) -> int:
    if not wallet_hit(exc):
        raise exc
    print(f"stopped: --max-usd reached (${factory.wallet.spent:.4f} spent); {where} keeps everything that finished")
    return 3


def _peers(values: Sequence[str]) -> dict[str, str]:
    return {f"a2a/{k.removeprefix('a2a/')}": v for k, _, v in (x.partition("=") for x in values)}


@contextlib.asynccontextmanager
async def _hub_market(p: Project, options: Sequence[Option], out: Path, peers: dict[str, str],
                      reserve: float | None, factory: HubFactory, port: int = 9250) -> AsyncIterator[Market]:
    from purse.hashiru_run import HubMarket, bench_settings
    hub = factory(bench_settings(p, options, out_dir=out, max_usd=factory.wallet.max_usd, max_escalations=0,
                                 peer_urls=peers, turn_reserve_usd=0.0, port=port))
    async with hub.serving(), HubMarket(hub, hub.new_session("purse"), reserve) as market:
        yield market


async def _live(cfg: RunConfig, p: Project, text: str, options: Sequence[Option], seed: int,
                table: OutcomeTable, factory: HubFactory) -> tuple[dict[str, Any], RunResult]:
    """One live run of HASHIRU (or of the LLM split). Past --max-usd it raises WalletExceeded, which stops the
    batch: every call would be refused, and the results would record failures that are not the allocator's."""
    from purse.hashiru_run import bench_settings, propose_split, run_hashiru, variant_text
    factory.wallet.check()
    hub = factory(bench_settings(p, options, out_dir=cfg.out, max_usd=factory.wallet.max_usd,
                                 peer_urls=dict(cfg.peers), ceo_model=cfg.ceo, port=cfg.port))
    async with hub.serving():
        if cfg.allocator == "llm-split":
            alloc, ceo = await propose_split(hub, p, variant_text(text, "split"))
            result = await run_upfront(with_budget(p, max(0.0, p.budget_usd - ceo)), alloc,
                                       ReplayMarket(table, options, draw=seed), options)
            return {"allocation": alloc}, replace(result, spent_usd=result.spent_usd + ceo, allocator_usd=ceo)
        live = await run_hashiru(hub, p, variant_text(text, "names-models" if cfg.allocator.endswith("models")
                                                      else ""))
        extras = await live_extras(p, Path(live.trace), table, options, seed)
        return extras, RunResult(live.answers, live.spent_usd, (), live.ceo_usd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="purse", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    pool = sub.add_parser("pool", help="write the job pool")
    pool.add_argument("--seed", type=int, default=1)
    pool.add_argument("--per-cell", type=int, default=8)
    pool.add_argument("--out", type=Path, required=True)
    cal = sub.add_parser("calibrate", help="record outcomes of jobs on options (calls real models)")
    check = sub.add_parser("check", help="the validity checks V1 to V7")
    runp = sub.add_parser("run", help="an allocator on a condition's projects")
    for p in (cal, check, runp):
        p.add_argument("--pool", type=Path, required=True)
        p.add_argument("--table", type=Path, required=True)
    for p in (cal, runp):
        p.add_argument("--max-usd", type=float, default=2.0, help="the hub's cap on real spend")
        p.add_argument("--peer", action="append", default=[], help="NAME=URL of a running outside agent")
        p.add_argument("--spend", action="store_true", help="allow calls to real models")
        p.add_argument("--port", type=int, default=9250, help="the hub's port (one per concurrent command)")
    cal.add_argument("--options", required=True, help="comma-separated market options")
    cal.add_argument("--repeats", type=int, default=1)
    cal.add_argument("--levels", default=",".join(map(str, LEVELS)))
    cal.add_argument("--per-cell", type=int, default=8, help="jobs per cell (the pilot uses 5)")
    cal.add_argument("--memory-mb", type=int, default=24576)
    cal.add_argument("--out", type=Path, default=Path("bench/out/calibration"))
    check.add_argument("--seed", type=int, default=0)
    runp.add_argument("--allocator", required=True)
    runp.add_argument("--rho", type=float, default=0.5)
    runp.add_argument("--tau", type=float, default=0.5)
    runp.add_argument("--projects", type=int, default=10)
    runp.add_argument("--seeds", type=int, default=3)
    runp.add_argument("--price", type=float, help="outside agents' price (default: c_ref, the sweep's middle)")
    runp.add_argument("--memory-mb", type=int, default=12288)
    runp.add_argument("--ceo", default=CEO_MODEL)
    runp.add_argument("--uninformed", action="store_true")
    runp.add_argument("--live", action="store_true", help="run a replayed allocator through a live hub (V8)")
    runp.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    argv = list(sys.argv if argv is None else argv)
    if args.command == "pool":
        jobs, skipped = build_pool(args.seed, args.per_cell)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(asdict(j)) + "\n" for j in jobs))
        print(f"{len(jobs)} jobs in {args.out}" + (f"; skipped cells: {', '.join(skipped)}" if skipped else ""))
        return 0
    pool, table = load_pool(args.pool), OutcomeTable(args.table)
    if args.command == "check":
        checks = run_checks(pool, table, calibrated(table.options()), args.seed)
        for c in checks:
            print(f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.value:.3f} ({c.threshold})")
        return 0 if all(c.passed for c in checks) else 1
    peers = _peers(args.peer)
    if args.command == "calibrate":
        levels = {int(x) for x in args.levels.split(",")}
        jobs = [j for j in pool if j.level in levels and int(j.uid.rsplit("-", 1)[1]) < args.per_cell]
        missing = [o for o in args.options.split(",") if lookup(MARKET, o).kind == "agent" and o not in peers]
        if missing:  # without its agent running, every attempt would be recorded as a failure
            print("outside agents need a running peer: " + " ".join(f"--peer {o.removeprefix('a2a/')}=URL"
                                                                     for o in missing))
            return 2
        todo = planned(jobs, args.options.split(","), args.repeats, table, MARKET)
        if not args.spend:
            print(f"calibrate would make {len(todo)} calls to {args.options} (pass --spend to make them)")
            return 2
        calib = Project("calibration", (), 1.0, budget_usd=args.max_usd + 1, memory_mb=args.memory_mb,
                        deadline_s=600)

        factory = HubFactory(args.max_usd)

        async def go() -> int:
            async with _hub_market(calib, MARKET, args.out, peers, RESERVE_USD, factory, args.port) as market:
                return await calibrate(todo, table, market)
        try:
            print(f"recorded {asyncio.run(go())} attempts in {args.table}")
        except BaseException as exc:
            return _stopped(exc, factory, str(args.table))
        return 0
    market = calibrated(table.options())  # the options the table holds records for
    price = args.price if args.price is not None else sweep_prices(table, pool, market)[1]
    options = priced(market, price)
    cfg = RunConfig(args.allocator, args.rho, args.tau, args.projects, args.seeds, args.memory_mb, args.ceo,
                    not args.uninformed, args.live, args.max_usd, args.out, peers, args.port)
    if is_live(cfg):
        options = tuple(o for o in options if o.kind != "agent" or o.name in peers)
        if not args.spend:
            print(f"{args.allocator} runs live ({cfg.projects} projects x {cfg.seeds} seeds; pass --spend)")
            return 2
    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "run.json").write_text(json.dumps(environment(argv) | {"price": price}, indent=2))

    factory = HubFactory(cfg.max_usd)  # one cap for the whole batch

    def live_market(p: Project) -> Any:
        return _hub_market(p, options, cfg.out, peers, None, factory, cfg.port)

    async def live(c: RunConfig, p: Project, text: str, opts: Sequence[Option], seed: int) -> Any:
        return await _live(c, p, text, opts, seed, table, factory)
    try:
        lines = asyncio.run(run(cfg, pool, table, options, live if cfg.allocator in LIVE else None, live_market))
    except BaseException as exc:
        return _stopped(exc, factory, str(cfg.out / "results.jsonl"))
    mean = sum(x["score"] for x in lines) / len(lines) if lines else 0.0
    print(f"{args.allocator}: {len(lines)} runs, mean score {mean:.3f}, results in {cfg.out / 'results.jsonl'}")
    return 0
