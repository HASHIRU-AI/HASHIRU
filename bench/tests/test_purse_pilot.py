"""The open-weight pilot's report: merging its tables, the three-tier ladders it can judge, the value of knowing a
job's hidden bin measured out of fold, and the go/stop verdict (synthetic tables, no model calls)."""
import json
from dataclasses import asdict, replace

from purse import pilot
from purse.checks import run_checks
from purse.market import OPEN, SMALL, Tiers
from purse.table import OutcomeTable, Record
from tests.synthetic import FULL, build

NAMES = {SMALL: OPEN[0].name, "gpt-6-luna": OPEN[1].name, "meta/muse-spark-1.3": OPEN[3].name}


def renamed(broken=None):
    """A synthetic table built to pass (or to fail `broken`), with its three tiers renamed to open models 0, 1 and 3."""
    pool, table = build(broken)
    out = OutcomeTable(None)
    for j in pool:
        for o in FULL:
            for r in table.records(j.uid, o.name):
                out.add(replace(r, option=NAMES.get(o.name, o.name)))
    market = tuple(replace(o, name=NAMES.get(o.name, o.name)) for o in FULL)
    return pool, out, market


def test_merge_keeps_the_pool_s_jobs_once_each(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rec = Record("u1", "m", 0, "1", True, 0.01, 1.0)
    a.write_text(json.dumps(asdict(rec)) + "\n" + json.dumps(asdict(replace(rec, uid="other"))) + "\n")
    b.write_text(json.dumps(asdict(rec)) + "\n" + json.dumps(asdict(replace(rec, repeat=1))) + "\n")
    out = tmp_path / "merged.jsonl"
    assert pilot.merge([a, b, tmp_path / "missing.jsonl"], {"u1"}, out) == 2
    assert OutcomeTable(out).records("u1", "m") == [rec, replace(rec, repeat=1)]
    assert pilot.merge([a], {"u1"}, out) == 1  # written afresh, not appended


def test_ladders_are_the_calibrated_triples_in_price_order():
    _, table, _ = renamed()
    assert pilot.ladders(table.options()) == [Tiers(OPEN[0].name, OPEN[1].name, OPEN[3].name)]
    assert len(pilot.ladders({o.name for o in OPEN})) == 4


def test_a_table_built_to_pass_is_a_go_and_one_without_hidden_effects_is_not():
    pool, table, market = renamed()
    tiers = pilot.ladders(table.options())[0]
    checks = run_checks(pool, table, market, tiers=tiers)
    oof = pilot.out_of_fold_voi(pool, table, market)
    assert oof > 0 and pilot.go(checks, oof)
    pool, table, market = renamed("V5")
    flat = pilot.out_of_fold_voi(pool, table, market)
    assert abs(flat) < 0.02 and not pilot.go(run_checks(pool, table, market, tiers=tiers), flat)


def test_bin_rates_and_cap_hits():
    pool, table, _ = renamed()
    rates = pilot.bin_rates(table, pool, OPEN[1].name)
    assert rates[("arithmetic", 3)]["low"] > rates[("arithmetic", 3)]["high"]  # the hidden bin moves difficulty
    o, short = OPEN[3], OutcomeTable(None)
    for repeat, tokens in enumerate((pilot.CAP_TOKENS, pilot.CAP_TOKENS // 2)):  # one cut off, one not
        short.add(Record(pool[0].uid, o.name, repeat, "", False, tokens * o.output_usd_per_m / 1e6, 1.0))
    assert pilot.cap_hits(short, pool, o) == 1 and pilot.cap_hits(short, pool, OPEN[1]) == 0


def test_go_needs_v3_v4_v5_and_a_positive_out_of_fold_value():
    from purse.checks import Check

    def checks(failing=()):
        return [Check(f"V{i} x", 0.0, "", f"V{i}" not in failing) for i in range(1, 8)]
    assert pilot.go(checks(), 0.01) and pilot.go(checks(("V1", "V2", "V6", "V7")), 0.01)
    assert not any(pilot.go(checks((v,)), 0.01) for v in ("V3", "V4", "V5"))
    assert not pilot.go(checks(), 0.0)
