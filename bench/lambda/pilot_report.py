"""The open-weight pilot's report (no model calls): merges the pilot's tables with round 2's outside-agent records for
the same jobs, then prints each model's success by hidden bin and its attempts cut at the output cap, the validity
checks for every three-tier ladder, the value of knowing the hidden bin out of fold, and go or stop (purse/pilot.py).
Run from the repo root: PYTHONPATH=bench uv run python bench/lambda/pilot_report.py"""
import logging
import sys
from pathlib import Path

from purse import pilot
from purse.checks import run_checks
from purse.cli import load_pool
from purse.market import OPEN, calibrated
from purse.table import OutcomeTable

R2, P = Path("bench/out/r2"), Path("bench/out/pilot")
LEVELS = (1, 3, 5)
PER_CELL = 4

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("pilot_report")


def main() -> int:
    pool = [j for j in load_pool(R2 / "pool.jsonl")
            if j.level in LEVELS and int(j.uid.rsplit("-", 1)[1]) < PER_CELL]
    sources = [*sorted(P.glob("t-*.jsonl")), R2 / "t-agents.jsonl"]
    n = pilot.merge(sources, {j.uid for j in pool}, P / "table.jsonl")
    table = OutcomeTable(P / "table.jsonl")
    print(f"{n} records for {len(pool)} jobs (levels {LEVELS}, {PER_CELL} per cell) in {P / 'table.jsonl'}")
    served = [o for o in OPEN if o.name in table.options()]
    if len(served) < 3:
        log.error("fewer than three open models are calibrated (%s): run bench/lambda/pilot.sh first",
                  [o.name for o in served])
        return 2

    print("\n== success by family, level and hidden bin (low / mid / high) ==")
    for o in served:
        rates = pilot.bin_rates(table, pool, o.name)
        tried = sum(len(table.records(j.uid, o.name)) for j in pool)
        print(f"\n{o.name}: {tried} attempts, {pilot.cap_hits(table, pool, o)} cut at the output cap")
        for (fam, level), by_bin in sorted(rates.items()):
            cells = " / ".join(f"{by_bin.get(b, float('nan')):.0%}" for b in ("low", "mid", "high"))
            print(f"  {fam:12} level {level}: {cells}")

    any_go = False
    for tiers in pilot.ladders(table.options()):
        market = tuple(o for o in calibrated(table.options()) if o.kind == "agent" or o.name in vars(tiers).values())
        checks = run_checks(pool, table, market, tiers=tiers)
        oof = pilot.out_of_fold_voi(pool, table, market)
        verdict = pilot.go(checks, oof)
        any_go |= verdict
        print(f"\n== ladder: {tiers.small} < {tiers.mid} < {tiers.top}: {'GO' if verdict else 'stop'} ==")
        for c in checks:
            print(f"{'PASS' if c.passed else 'FAIL'} {c.name}: {c.value:.3f} ({c.threshold})")
        print(f"value of the hidden bin out of fold, rho 0: {oof:+.3f}")
    print("\nVerdict:", "GO: a full round with 8 to 16 jobs per cell is worth running" if any_go else
          "STOP: no ladder makes the hidden bin worth knowing; keep the credits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
