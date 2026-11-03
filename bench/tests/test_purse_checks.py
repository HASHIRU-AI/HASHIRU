"""Oracles and the validity checks, on synthetic tables."""
import pytest

from purse.checks import run_checks
from purse.market import SMALL as QWEN
from purse.market import priced
from purse.oracles import c_full, expected_cost, expected_score, oracle_buys, oracle_full, sweep_prices, voi
from purse.projects import make_project, with_budget
from purse.solvers import SKIP, fixed, hybrid, plan_knapsack
from purse.table import cell_stats, visible_stats
from tests.synthetic import FULL as MARKET
from tests.synthetic import build

LUNA, SPARK = "gpt-6-luna", "meta/muse-spark-1.3"


def projects(pool, table, rho, tau=0.5, n=20):
    used, out = set(), []
    for s in range(n):
        p = make_project(pool, s % 2, rho, s, used)
        out.append(with_budget(p, tau * c_full(p, table, MARKET)))
    return out


def test_visible_stats_mix_the_bins_by_rho():
    pool, table = build()
    hidden = cell_stats(table, pool, hidden=True)
    sure, blind = visible_stats(hidden, 1.0), visible_stats(hidden, 0.0)
    assert sure[("strings", 1, None, QWEN)].rate == hidden[("strings", 1, "low", QWEN)].rate
    assert abs(blind[("strings", 3, None, QWEN)].rate - sum(hidden[("strings", 3, b, QWEN)].rate
                                                           for b in ("low", "mid", "high")) / 3) < 1e-9


def test_oracle_full_bounds_every_allocator():
    pool, table = build()
    stats = visible_stats(cell_stats(table, pool, hidden=True), 0.5)
    for p in projects(pool, table, 0.5):
        best = expected_score(p, oracle_full(p, table, MARKET, p.budget_usd), table, MARKET)
        assert expected_cost(p, oracle_full(p, table, MARKET, p.budget_usd), table, MARKET) <= p.budget_usd + 1e-9
        others = [plan_knapsack(p, stats, MARKET), hybrid(p), *(fixed(p, o.name, MARKET) for o in MARKET)]
        for alloc in others:
            if expected_cost(p, alloc, table, MARKET) <= p.budget_usd + 1e-9:
                assert expected_score(p, alloc, table, MARKET) <= best + 1e-9
        assert expected_score(p, {j.id: SKIP for j in p.jobs}, table, MARKET) == 0.0


def test_voi_is_zero_when_hidden_bins_carry_no_information():
    pool, table = build("V5")
    hidden = cell_stats(table, pool, hidden=True)
    for p in projects(pool, table, 0.0, n=6):
        assert abs(voi(p, table, hidden, MARKET, p.budget_usd)) < 1e-9


def test_voi_is_zero_at_rho_one_and_positive_at_rho_zero():
    pool, table = build()
    hidden = cell_stats(table, pool, hidden=True)
    assert all(abs(voi(p, table, hidden, MARKET, p.budget_usd)) < 1e-9 for p in projects(pool, table, 1.0, n=6))
    assert sum(voi(p, table, hidden, MARKET, p.budget_usd) for p in projects(pool, table, 0.0)) > 0


def test_oracle_buys_cheap_agents_and_not_dear_ones():
    pool, table = build()
    prices = sweep_prices(table, pool, MARKET)
    assert len(prices) == 4 and prices == tuple(sorted(prices)) and abs(prices[1] - 0.002) < 1e-12
    p = projects(pool, table, 0.5, n=1)[0]
    cheap, dear = priced(MARKET, prices[0]), priced(MARKET, prices[-1])
    assert len(oracle_buys(p, table, cheap, p.budget_usd)) > len(oracle_buys(p, table, dear, p.budget_usd))


def test_checks_pass_on_a_table_built_to_pass():
    pool, table = build()
    checks = run_checks(pool, table, MARKET)
    assert [c.name.split()[0] for c in checks] == [f"V{i}" for i in range(1, 8)]
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]


@pytest.mark.parametrize("broken", [f"V{i}" for i in range(1, 8)])
def test_each_check_fails_on_a_table_built_to_fail_it(broken):
    pool, table = build(broken)
    failed = {c.name.split()[0] for c in run_checks(pool, table, MARKET) if not c.passed}
    assert broken in failed


def test_the_checks_judge_any_market_by_its_tiers():
    """The open-weight pilot names its own small, middle and top tiers: renaming every model (and telling the checks
    which is which) leaves every check's value as it was."""
    from dataclasses import replace

    from purse.market import Tiers
    from purse.table import OutcomeTable, Record
    names = {QWEN: "hosted_vllm/s", LUNA: "hosted_vllm/m", SPARK: "hosted_vllm/t"}
    market = tuple(replace(o, name=names.get(o.name, o.name)) for o in MARKET)
    tiers = Tiers(*(names[n] for n in (QWEN, LUNA, SPARK)))
    for broken in (None, "V2", "V3", "V4"):  # the tier-specific checks, each passing and failing
        pool, table = build(broken)
        renamed = OutcomeTable(None)
        for j in pool:
            for o in MARKET:
                for r in table.records(j.uid, o.name):
                    renamed.add(Record(**(vars(r) | {"option": names.get(o.name, o.name)})))
        before = [(c.name.split()[0], c.value) for c in run_checks(pool, table, MARKET)]
        after = [(c.name.split()[0], c.value) for c in run_checks(pool, renamed, market, tiers=tiers)]
        assert after == before, broken
    assert sweep_prices(renamed, pool, market, ref=tiers.mid) == sweep_prices(table, pool, MARKET)
    p = make_project(pool, 0, 0.5, 0, set())
    assert hybrid(p, tiers) == {k: names[v] for k, v in hybrid(p).items()}
