"""The purse command: pool, calibrate, check and run, with fake markets and synthetic tables (no model is called)."""
import asyncio
import json
from dataclasses import asdict

import pytest

from purse import cli
from purse.market import MARKET, Outcome
from purse.table import OutcomeTable
from tests.helpers import job
from tests.synthetic import build


def write(tmp_path, broken=None):
    pool, table = build(broken)
    pool_file, table_file = tmp_path / "pool.jsonl", tmp_path / "table.jsonl"
    pool_file.write_text("".join(json.dumps(asdict(j)) + "\n" for j in pool))
    saved = OutcomeTable(table_file)
    for uid, option in {(j.uid, o.name) for j in pool for o in MARKET}:
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


def test_run_writes_the_environment(tmp_path):
    pool_file, table_file = write(tmp_path)
    out = tmp_path / "out"
    cli.main(["run", "--allocator", "knapsack", "--pool", str(pool_file), "--table", str(table_file), "--projects",
              "2", "--seeds", "1", "--out", str(out)])
    env = json.loads((out / "run.json").read_text())
    assert {"git", "uv_lock_sha256", "python", "platform", "ollama", "price_map", "argv"} <= set(env)
    assert "sk-" not in json.dumps(env) and "LLM_" not in json.dumps(env)
