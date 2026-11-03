"""The trained splitter's data and scoring (plan/2026-11-02-finetuning-scope.md, §10): the prompt a splitter reads,
the knapsack's split as its target, reading a reply, training projects that share no job with the evaluation
projects, and replay scoring that charges the splitter's own call (synthetic tables, no model calls)."""
import json

import pytest

from purse import splitter
from purse.projects import make_project, with_budget
from purse.solvers import SKIP, plan_knapsack
from purse.table import cell_stats, visible_stats
from tests.synthetic import FULL as MARKET
from tests.synthetic import build


def setup(budget=0.01):
    pool, table = build()
    p = with_budget(make_project(pool, 0, 0.5, 3, set()), budget)
    stats = visible_stats(cell_stats(table, [j for j in pool if j.uid.endswith(("1", "3", "5", "7"))], hidden=True,
                                     market=MARKET), 0.5)
    return pool, table, p, stats


def test_the_prompt_gives_budget_market_statistics_and_job_headings_only():
    _, _, p, stats = setup()
    text = splitter.split_text(p, MARKET, stats)
    assert f"${p.budget_usd:.4f}" in text and all(o.name in text for o in MARKET)
    assert "| family | level | option | success | mean cost |" in text and '"skip"' in text
    for j in p.jobs:
        assert f"### {j.id} ({j.family}, level {j.level}, value {j.value})" in text
        assert j.prompt not in text or len(j.prompt) <= splitter.PREVIEW_CHARS  # long job texts are cut


def test_the_target_is_the_knapsack_on_what_the_reply_leaves():
    _, _, p, stats = setup()
    text = splitter.split_text(p, MARKET, stats)
    own = splitter.own_cost(text, splitter.TARGET_TOKENS)
    assert 0 < own < p.budget_usd
    target = json.loads(splitter.target(p, MARKET, stats, own))
    assert target == plan_knapsack(with_budget(p, p.budget_usd - own), stats, MARKET)
    assert set(target) == {j.id for j in p.jobs}


def test_a_reply_becomes_an_allocation():
    _, _, p, _ = setup()
    ids = [j.id for j in p.jobs]
    reply = "Here it is:\n```json\n" + json.dumps({ids[0]: "gpt-6-luna", ids[1]: "made-up", ids[2]: "skip"}) + "\n```"
    alloc, bad = splitter.parse_split(reply, p, MARKET)
    assert alloc[ids[0]] == "gpt-6-luna" and alloc[ids[1]] == SKIP and alloc[ids[2]] == SKIP
    assert all(alloc[i] == SKIP for i in ids[3:]) and bad == 1  # a name not in the market; missing jobs are skipped
    assert splitter.parse_split("no json", p, MARKET)[0] == {i: SKIP for i in ids}


def test_training_projects_share_no_job_with_the_evaluation_projects():
    pool, table = build(per_cell=4)
    held = splitter.eval_projects(pool, table, MARKET, n=6)
    rows = splitter.training_rows(pool, table, MARKET, held, n=40, seed=1)
    held_uids = {j.uid for p in held for j in p.jobs}
    assert len(rows) == 40 and all(not (set(r["uids"]) & held_uids) for r in rows)
    names = {o.name for o in MARKET} | {SKIP}
    for r in rows:
        split = json.loads(r["completion"])
        assert len(split) == 8 and set(split.values()) <= names and all(f"### {i} (" in r["prompt"] for i in split)
    assert len({r["prompt"] for r in rows}) == 40  # every project differs


def test_scoring_charges_the_splitter_s_own_call():
    _, table, p, stats = setup(budget=0.02)
    alloc = plan_knapsack(p, stats, MARKET)
    free = splitter.score_split(p, alloc, 0.0, table, MARKET, seed=0)
    assert free > 0
    assert splitter.score_split(p, alloc, p.budget_usd, table, MARKET, seed=0) == 0.0  # the reply took it all
    with pytest.raises(ValueError):
        splitter.score_split(p, alloc, -1.0, table, MARKET, seed=0)


def test_a_reply_is_scored_with_its_own_tokens_paid():
    _, table, p, stats = setup(budget=0.02)
    reply = json.dumps(plan_knapsack(p, stats, MARKET))
    cheap = splitter.score_reply(p, reply, 1000, 100, table, MARKET)
    assert cheap["own_usd"] == pytest.approx(splitter.reply_cost(1000, 100)) and cheap["bad"] == 0
    each = [splitter.score_split(p, cheap["alloc"], cheap["own_usd"], table, MARKET, s) for s in range(3)]
    assert cheap["score"] == pytest.approx(sum(each) / 3)
    dear = splitter.score_reply(p, reply, 10**9, 10**9, table, MARKET)  # a reply that costs more than the budget
    assert dear["score"] == 0.0 and dear["own_usd"] > p.budget_usd
