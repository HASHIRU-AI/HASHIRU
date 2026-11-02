"""The purse command: pool, calibrate, check and run, with fake markets and synthetic tables (no model is called)."""
import asyncio
import json
from dataclasses import asdict

import pytest

from purse import cli
from purse.market import MARKET, Outcome
from purse.table import OutcomeTable
from tests.helpers import job
from tests.synthetic import FULL, build


def write(tmp_path, broken=None):
    pool, table = build(broken)
    pool_file, table_file = tmp_path / "pool.jsonl", tmp_path / "table.jsonl"
    pool_file.write_text("".join(json.dumps(asdict(j)) + "\n" for j in pool))
    saved = OutcomeTable(table_file)
    for uid, option in {(j.uid, o.name) for j in pool for o in FULL}:
        for r in table.records(uid, option):
            saved.add(r)
    return pool_file, table_file


def test_pool_round_trips(tmp_path):
    out = tmp_path / "pool.jsonl"
    assert cli.main(["pool", "--seed", "3", "--per-cell", "1", "--out", str(out)]) == 0
    jobs = cli.load_pool(out)
    assert len(jobs) == 60 and len({j.uid for j in jobs}) == 60 and isinstance(jobs[0].features, tuple)


def test_live_commands_refuse_without_spend(tmp_path, capsys):
    pool_file, table_file = write(tmp_path)
    code = cli.main(["calibrate", "--pool", str(pool_file), "--table", str(tmp_path / "new.jsonl"), "--options",
                     "gpt-6-luna", "--repeats", "2", "--levels", "1,3", "--per-cell", "1", "--max-usd", "1"])
    assert code == 2 and "48 calls" in capsys.readouterr().out  # 4 families x 2 levels x 3 bins x 1 job x 2
    assert not (tmp_path / "new.jsonl").exists()
    for allocator in ("hashiru", "llm-split"):
        assert cli.main(["run", "--allocator", allocator, "--pool", str(pool_file), "--table", str(table_file),
                         "--out", str(tmp_path / "o")]) == 2
    assert cli.main(["run", "--allocator", "knapsack", "--live", "--pool", str(pool_file), "--table", str(table_file),
                     "--out", str(tmp_path / "o")]) == 2


class FlakyMarket:
    def __init__(self, fail_at=None):
        self.calls, self.fail_at = 0, fail_at

    async def attempt(self, j, option):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("provider down")
        if j.uid == "b":
            return Outcome("", False, 0.0, 0.1, refused="ollama_chat/x needs 5000 MB")
        return Outcome(f"<answer>{j.answer}</answer>", True, 0.01, 0.1)


def test_calibrate_resumes_and_records_refusals(tmp_path):
    jobs = [job("a"), job("b"), job("c")]
    table = OutcomeTable(tmp_path / "t.jsonl")
    todo = cli.planned(jobs, ["gpt-6-luna"], 1, table, MARKET)
    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(cli.calibrate(todo, table, FlakyMarket(fail_at=3)))
    again = OutcomeTable(tmp_path / "t.jsonl")
    assert again.has("a", "gpt-6-luna", 0) and again.has("b", "gpt-6-luna", 0) and not again.has("c", "gpt-6-luna", 0)
    assert again.records("b", "gpt-6-luna")[0].refused == "ollama_chat/x needs 5000 MB"
    rest = cli.planned(jobs, ["gpt-6-luna"], 1, again, MARKET)
    market = FlakyMarket()
    assert asyncio.run(cli.calibrate(rest, again, market)) == 1 and market.calls == 1
    assert OutcomeTable(tmp_path / "t.jsonl").rate("c", "gpt-6-luna") == 1.0


def test_check_exits_nonzero_on_a_failing_table(tmp_path, capsys):
    good = tmp_path / "good"
    good.mkdir()
    assert cli.main(["check", "--pool", *map(str, write(good)[:1]), "--table", str(good / "table.jsonl")]) == 0
    bad = tmp_path / "bad"
    bad.mkdir()
    pool_file, table_file = write(bad, "V4")
    assert cli.main(["check", "--pool", str(pool_file), "--table", str(table_file)]) == 1
    assert "FAIL V4" in capsys.readouterr().out


@pytest.mark.parametrize("allocator", ["knapsack", "cascade", "oracle", "hybrid", "uniform", "random",
                                       "fixed:gpt-6-luna"])
def test_run_replays_solvers_without_spend(tmp_path, allocator):
    pool_file, table_file = write(tmp_path)
    out = tmp_path / "out"
    assert cli.main(["run", "--allocator", allocator, "--pool", str(pool_file), "--table", str(table_file), "--rho",
                     "0.5", "--tau", "0.5", "--projects", "4", "--seeds", "2", "--out", str(out)]) == 0
    lines = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert len(lines) == 8 and {x["allocator"] for x in lines} == {allocator}
    assert all(0 <= x["score"] <= 1 and x["spent_usd"] >= 0 and x["budget_usd"] > 0 for x in lines)
    assert all(x["spent_usd"] <= x["budget_usd"] + 0.05 for x in lines)  # one call may overrun, as in the hub


@pytest.mark.parametrize("allocator", ["cascade", "hybrid", "random", "fixed:meta/muse-spark-1.3"])
def test_run_bounded_never_spends_past_the_budget(tmp_path, allocator):
    pool_file, table_file = write(tmp_path)
    runs = {}
    for rule in ("first", "v2"):
        out = tmp_path / rule
        assert cli.main(["run", "--allocator", allocator, "--pool", str(pool_file), "--table", str(table_file),
                         "--tau", "0.5", "--projects", "4", "--seeds", "2", "--out", str(out)]
                        + (["--bounded"] if rule == "v2" else [])) == 0
        runs[rule] = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert {x["pool_rule"] for x in runs["v2"]} == {"v2"} and {x["pool_rule"] for x in runs["first"]} == {"first"}
    assert all(x["spent_usd"] <= x["budget_usd"] + 1e-12 for x in runs["v2"])


def test_run_writes_the_environment(tmp_path):
    pool_file, table_file = write(tmp_path)
    out = tmp_path / "out"
    cli.main(["run", "--allocator", "knapsack", "--pool", str(pool_file), "--table", str(table_file), "--projects",
              "2", "--seeds", "1", "--out", str(out)])
    env = json.loads((out / "run.json").read_text())
    assert {"git", "uv_lock_sha256", "python", "platform", "ollama", "price_map", "argv"} <= set(env)
    assert "sk-" not in json.dumps(env) and "LLM_" not in json.dumps(env)


def test_live_variants_are_switched_by_flags(tmp_path, monkeypatch):
    pool_file, table_file = write(tmp_path)
    seen = []

    async def fake_live(cfg, p, text, options, seed, table, factory, stats=None):
        seen.append((cfg.by_reference, cfg.route_on_rates, stats is not None))
        from purse.solvers import RunResult
        return {}, RunResult({}, 0.0, ())
    monkeypatch.setattr(cli, "_live", fake_live)
    args = ["run", "--allocator", "hashiru", "--pool", str(pool_file), "--table", str(table_file), "--projects", "1",
            "--seeds", "1", "--spend", "--out", str(tmp_path / "out")]
    assert cli.main([*args, "--by-reference", "--route-on-rates"]) == 0 and seen == [(True, True, True)]
    line = json.loads((tmp_path / "out" / "results.jsonl").read_text().splitlines()[-1])
    assert line["by_reference"] is True and line["route_on_rates"] is True


def test_calibrate_plans_the_small_tier_and_the_unreliable_seller(tmp_path, capsys):
    from purse.market import SMALL
    pool_file, _ = write(tmp_path)
    base = ["calibrate", "--pool", str(pool_file), "--table", str(tmp_path / "new.jsonl"), "--levels", "1",
            "--per-cell", "1", "--max-usd", "1"]
    assert cli.main([*base, "--options", SMALL]) == 2 and "12 calls" in capsys.readouterr().out  # 4 families x 3 bins
    assert cli.main([*base, "--options", "a2a/gridfast"]) == 2
    assert "--peer gridfast=URL" in capsys.readouterr().out  # an outside agent must be running
    assert cli.main([*base, "--options", "a2a/gridfast", "--peer", "gridfast=http://127.0.0.1:9303/"]) == 2
    assert "3 calls" in capsys.readouterr().out  # grids only


def test_calibrate_leaves_provider_errors_out_of_the_table(tmp_path):
    from purse.market import PROVIDER_ERROR

    class Glitchy(FlakyMarket):
        async def attempt(self, j, option):
            if j.uid == "a":
                return Outcome("", False, 0.0, 0.1, refused=PROVIDER_ERROR + "429")
            return await super().attempt(j, option)
    table = OutcomeTable(tmp_path / "t.jsonl")
    todo = cli.planned([job("a"), job("c")], ["gpt-6-luna"], 1, table, MARKET)
    assert asyncio.run(cli.calibrate(todo, table, Glitchy())) == 1  # recorded: c; a is left for a rerun
    assert not table.has("a", "gpt-6-luna", 0) and table.has("c", "gpt-6-luna", 0)


def test_calibrate_can_take_some_families(tmp_path, capsys):
    pool_file, _ = write(tmp_path)
    base = ["calibrate", "--pool", str(pool_file), "--table", str(tmp_path / "new.jsonl"), "--options", "gpt-6-luna",
            "--levels", "1", "--per-cell", "1", "--max-usd", "1"]
    assert cli.main([*base, "--families", "grids,strings"]) == 2 and "6 calls" in capsys.readouterr().out
    assert cli.main([*base, "--families", "nope"]) == 2 and "unknown families" in capsys.readouterr().out


def test_run_can_start_at_a_later_seed(tmp_path):
    pool_file, table_file = write(tmp_path)
    out = tmp_path / "out"
    base = ["run", "--allocator", "random", "--pool", str(pool_file), "--table", str(table_file), "--projects", "2"]
    assert cli.main([*base, "--seeds", "1", "--first-seed", "2", "--out", str(out)]) == 0
    later = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    assert {x["seed"] for x in later} == {2}
    assert cli.main([*base, "--seeds", "3", "--out", str(tmp_path / "all")]) == 0
    every = [json.loads(x) for x in (tmp_path / "all" / "results.jsonl").read_text().splitlines()]
    assert [x["score"] for x in every if x["seed"] == 2] == [x["score"] for x in later]  # the same seed, the same run
