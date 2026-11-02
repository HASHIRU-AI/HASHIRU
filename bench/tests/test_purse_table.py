"""The market, the outcome table and replay over it."""
import asyncio

from purse.market import MARKET, Option, priced, serves
from purse.replay import ReplayMarket
from purse.table import OutcomeTable, Record, cell_stats
from tests.helpers import job

LUNA = "gpt-6-luna"


def rec(uid, option=LUNA, repeat=0, correct=True, cost=0.01, refused=""):
    return Record(uid, option, repeat, "1" if correct else "2", correct, cost, 0.5, refused)


def test_table_survives_a_reload_and_skips_done_records(tmp_path):
    path = tmp_path / "table.jsonl"
    table = OutcomeTable(path)
    for r in (rec("a"), rec("a", repeat=1, correct=False), rec("b", refused="does not fit")):
        table.add(r)
    with path.open("a") as f:
        f.write('{"uid": "c", "opt')  # a write cut short by Ctrl-C
    again = OutcomeTable(path)
    assert again.has("a", LUNA, 0) and again.has("a", LUNA, 1) and again.has("b", LUNA, 0)
    assert not again.has("c", LUNA, 0) and not again.has("a", LUNA, 2)
    assert again.rate("a", LUNA) == 0.5 and again.rate("b", LUNA) == 0.0  # a refusal is a failure
    assert again.cost("a", LUNA) == 0.01 and again.rate("zzz", LUNA) is None
    again.add(rec("d"))
    assert OutcomeTable(path).has("d", LUNA, 0)  # appending after a cut line still reads back


def test_cell_stats_with_and_without_hidden_bins():
    jobs = [job("x", level=2, hidden="low"), job("y", level=2, hidden="high")]
    table = OutcomeTable(None)
    table.add(rec("x", correct=True, cost=0.01))
    table.add(rec("y", correct=False, cost=0.03))
    visible = cell_stats(table, jobs, hidden=False)
    assert visible[("arithmetic", 2, None, LUNA)].rate == 0.5
    assert abs(visible[("arithmetic", 2, None, LUNA)].cost_usd - 0.02) < 1e-12
    assert visible[("arithmetic", 2, None, LUNA)].n == 2
    split = cell_stats(table, jobs, hidden=True)
    assert split[("arithmetic", 2, "low", LUNA)].rate == 1.0 and split[("arithmetic", 2, "high", LUNA)].rate == 0.0


def test_an_outside_agents_cost_is_its_current_price():
    jobs = [job("x")]
    table = OutcomeTable(None)
    table.add(rec("x", option="a2a/calc", cost=0.001))
    market = priced(MARKET, 0.05)
    assert cell_stats(table, jobs, hidden=False, market=market)[("arithmetic", 1, None, "a2a/calc")].cost_usd == 0.05
    outcome = asyncio.run(ReplayMarket(table, market).attempt(jobs[0], "a2a/calc"))
    assert outcome.cost_usd == 0.05 and outcome.correct


def test_replay_draws_are_deterministic_and_rotate():
    table = OutcomeTable(None)
    table.add(rec("x", repeat=0, correct=True, cost=0.01))
    table.add(rec("x", repeat=1, correct=False, cost=0.02))
    j = job("x")
    first = [asyncio.run(ReplayMarket(table, draw=d).attempt(j, LUNA)) for d in range(3)]
    assert [o.correct for o in first] == [True, False, True] and first[1].cost_usd == 0.02
    assert asyncio.run(ReplayMarket(table).attempt(j, "nope")).refused == "no record"


def test_agents_serve_only_their_families():
    calc = next(o for o in MARKET if o.name == "a2a/calc")
    assert serves(calc, job("x", family="arithmetic")) and not serves(calc, job("y", family="grids"))
    assert serves(Option("gpt-6-luna", "cloud"), job("y", family="grids"))
    assert {o.kind for o in MARKET} == {"cloud", "agent"} and len(MARKET) == 5
