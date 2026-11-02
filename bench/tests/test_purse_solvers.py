"""Knapsack, the up-front executor, the cascade and fitting its threshold."""
import asyncio
import itertools
import random

from purse.market import Option, Outcome
from purse.projects import Project
from purse.solvers import SKIP, UNIT, cascade, fit_theta, knapsack, plan_knapsack, run_upfront
from purse.table import CellStat
from tests.helpers import job

QWEN, LUNA, SPARK = "ollama_chat/qwen3.5:4b", "gpt-6-luna", "meta/muse-spark-1.3"
OPTIONS = (Option(QWEN, "local"), Option(LUNA, "cloud"), Option(SPARK, "cloud"))


class FakeMarket:
    def __init__(self, outcomes):
        self.outcomes, self.calls = outcomes, []

    async def attempt(self, j, option):
        self.calls.append((j.id, option))
        return self.outcomes[(j.id, option)]


def ok(answer="1", cost=0.0, correct=True):
    return Outcome(f"<answer>{answer}</answer>", correct, cost, 0.1)


def test_knapsack_matches_brute_force():
    rng = random.Random(0)
    for _ in range(100):
        items = [(f"j{i}", [(f"o{k}", rng.random() * 4, rng.randint(0, 40) * UNIT) for k in range(3)])
                 for i in range(4)]
        budget = rng.randint(0, 80) * UNIT
        best = 0.0
        for pick in itertools.product(range(4), repeat=4):  # 3: skip
            chosen = [items[i][1][k] for i, k in enumerate(pick) if k < 3]
            if sum(c for _, _, c in chosen) <= budget + 1e-12:
                best = max(best, sum(g for _, g, _ in chosen))
        alloc = knapsack(items, budget)
        gains = {(j, o): g for j, choices in items for o, g, _ in choices}
        costs = {(j, o): c for j, choices in items for o, _, c in choices}
        assert sum(costs[(j, o)] for j, o in alloc.items() if o != SKIP) <= budget + 1e-12
        assert abs(sum(gains[(j, o)] for j, o in alloc.items() if o != SKIP) - best) < 1e-9


def test_plan_knapsack_keeps_a_margin_and_reads_cells():
    p = Project("p", (job("a", id="j1", value=4), job("b", id="j2", value=1)), 1.0, budget_usd=0.011)
    stats = {("arithmetic", 1, None, LUNA): CellStat(0.9, 0.006, 8), ("arithmetic", 1, None, QWEN): CellStat(0.2, 0, 8)}
    alloc = plan_knapsack(p, stats, OPTIONS)
    assert alloc == {"j1": LUNA, "j2": QWEN}  # 0.012 for both on Luna is over 0.9 x 0.011


def test_executor_charges_an_overrunning_call_in_full_then_refuses_cloud():
    jobs = (job("a", id="j1", value=4), job("b", id="j2", value=2), job("c", id="j3", value=1))
    p = Project("p", jobs, 1.0, budget_usd=0.01)
    market = FakeMarket({("j1", LUNA): ok(cost=0.03), ("j2", LUNA): ok(cost=0.001), ("j3", QWEN): ok()})
    result = asyncio.run(run_upfront(p, {"j1": LUNA, "j2": LUNA, "j3": QWEN}, market, OPTIONS))
    assert market.calls == [("j1", LUNA), ("j3", QWEN)]
    assert abs(result.spent_usd - 0.03) < 1e-12 and set(result.answers) == {"j1", "j3"}
    assert [a.outcome.refused for a in result.attempts] == ["", "the pool cannot pay for it", ""]


def stats_for(j, rates):
    return {(j.family, j.level, None, o): CellStat(r, c, 8) for o, (r, c) in rates.items()}


def test_cascade_escalates_on_disagreement():
    j = job("a", id="j1", family="strings")
    p = Project("p", (j,), 1.0, budget_usd=1.0)
    stats = stats_for(j, {QWEN: (0.6, 0.0), LUNA: (0.8, 0.001), SPARK: (0.95, 0.01)})
    market = FakeMarket({("j1", QWEN): ok("x"), ("j1", LUNA): ok("y"), ("j1", SPARK): ok("y")})
    result = asyncio.run(cascade(p, stats, market, OPTIONS, theta=0.5))
    assert market.calls == [("j1", QWEN), ("j1", LUNA), ("j1", SPARK)] and result.answers["j1"].endswith("y</answer>")
    agree = FakeMarket({("j1", QWEN): ok("x"), ("j1", LUNA): ok("x")})
    asyncio.run(cascade(p, stats, agree, OPTIONS, theta=0.5))
    assert agree.calls == [("j1", QWEN), ("j1", LUNA)]


def test_cascade_stops_at_three_checks():
    j = job("g", id="j1", family="grids", answer='{"1": {"color": "red"}}')
    p = Project("p", (j,), 1.0, budget_usd=1.0)
    four = (*OPTIONS, Option("gpt-6.1-sol", "cloud"))
    stats = stats_for(j, {QWEN: (0.6, 0.0), LUNA: (0.7, 0.001), SPARK: (0.8, 0.01), "gpt-6.1-sol": (0.9, 0.02)})
    market = FakeMarket({("j1", o.name): ok('{"1": {"color": "blue"}}', correct=False) for o in four})
    asyncio.run(cascade(p, stats, market, four, theta=0.5))
    assert len(market.calls) == 3


def test_cascade_without_a_qualifying_option_uses_the_most_reliable_one():
    j = job("a", id="j1", family="strings")
    p = Project("p", (j,), 1.0, budget_usd=1.0)
    market = FakeMarket({("j1", SPARK): ok("z")})
    asyncio.run(cascade(p, stats_for(j, {QWEN: (0.1, 0.0), SPARK: (0.3, 0.01)}), market, OPTIONS, theta=0.9))
    assert market.calls == [("j1", SPARK)]


def test_fit_theta_picks_the_best_on_the_other_fold():
    j = job("a", id="j1", family="strings", answer="right")
    p = Project("p", (j,), 1.0, budget_usd=1.0)
    stats = stats_for(j, {QWEN: (0.5, 0.0), LUNA: (0.6, 0.001), SPARK: (0.95, 0.01)})
    market = FakeMarket({("j1", QWEN): ok("wrong"), ("j1", LUNA): ok("wrong"), ("j1", SPARK): ok("right")})
    assert asyncio.run(fit_theta([p], stats, market, OPTIONS)) == 0.7


def test_an_outside_agent_needs_its_price_left_in_the_pool():
    from purse.solvers import Pool
    pool = Pool(0.005)
    assert not pool.allows(Option("a2a/calc", "agent", 0.01)) and pool.allows(Option("a2a/calc", "agent", 0.004))
    assert pool.allows(Option(LUNA, "cloud")) and pool.allows(Option(QWEN, "local"))
    pool.charge(0.006)
    assert not pool.allows(Option(LUNA, "cloud")) and pool.allows(Option(QWEN, "local"))


# The v2 pool rule (bounded=True): a call is bounded by its pool before it is made, as the hub bounds it.
PRICED = Option(LUNA, "cloud", input_usd_per_m=100.0, output_usd_per_m=1000.0)  # a short answer: about $0.256


def test_bounded_pool_refuses_a_call_that_cannot_pay_for_a_short_answer():
    from purse.solvers import MIN_ANSWER_TOKENS, Pool, short_answer_usd
    j = job("a", id="j1")
    need = short_answer_usd(PRICED, j)
    assert need > MIN_ANSWER_TOKENS * PRICED.output_usd_per_m / 1e6  # reading the prompt costs something too
    assert Pool(need, bounded=True).allows(PRICED, j) and not Pool(need * 0.99, bounded=True).allows(PRICED, j)
    assert Pool(need * 0.99).allows(PRICED, j)  # the old rule asks only for money left
    assert Pool(0.0, bounded=True).allows(Option(QWEN, "local"), j)


def test_bounded_executor_cuts_an_overrunning_call_at_the_pool():
    jobs = (job("a", id="j1", value=4), job("b", id="j2", value=2))
    p = Project("p", jobs, 1.0, budget_usd=0.3)
    market = FakeMarket({("j1", LUNA): ok(cost=0.5), ("j2", LUNA): ok(cost=0.001)})
    old = asyncio.run(run_upfront(p, {"j1": LUNA, "j2": LUNA}, market, (PRICED,)))
    assert set(old.answers) == {"j1"} and abs(old.spent_usd - 0.5) < 1e-12  # charged in full, past the pool
    market.calls.clear()
    new = asyncio.run(run_upfront(p, {"j1": LUNA, "j2": LUNA}, market, (PRICED,), bounded=True))
    assert market.calls == [("j1", LUNA)]  # j2 is refused: nothing is left for a short answer
    assert new.answers == {} and abs(new.spent_usd - 0.3) < 1e-12  # the cut call spends the pool and answers nothing
    assert [a.outcome.refused for a in new.attempts] == ["cut off at the pool", "the pool cannot pay for it"]


def test_bounded_executor_leaves_a_call_that_fits_alone():
    p = Project("p", (job("a", id="j1"),), 1.0, budget_usd=1.0)
    market = FakeMarket({("j1", LUNA): ok(cost=0.4)})
    result = asyncio.run(run_upfront(p, {"j1": LUNA}, market, (PRICED,), bounded=True))
    assert set(result.answers) == {"j1"} and abs(result.spent_usd - 0.4) < 1e-12


def test_bounded_cascade_cannot_spend_past_its_pool():
    j = job("a", id="j1", family="strings")
    p = Project("p", (j,), 1.0, budget_usd=0.3)
    spark = Option(SPARK, "cloud", input_usd_per_m=100.0, output_usd_per_m=1000.0)
    stats = stats_for(j, {LUNA: (0.8, 0.1), SPARK: (0.9, 0.2)})
    market = FakeMarket({("j1", LUNA): ok("x", cost=0.2), ("j1", SPARK): ok("y", cost=0.2)})
    result = asyncio.run(cascade(p, stats, market, (PRICED, spark), theta=0.5, bounded=True))
    assert result.spent_usd <= p.budget_usd + 1e-12 and market.calls == [("j1", LUNA)]


def test_bounded_rule_uses_the_hubs_constants():
    from hashiru.llm import INPUT_MARGIN
    from hashiru.runtime import MIN_ANSWER_TOKENS
    from purse import solvers
    assert (solvers.MIN_ANSWER_TOKENS, solvers.INPUT_MARGIN) == (MIN_ANSWER_TOKENS, INPUT_MARGIN)
