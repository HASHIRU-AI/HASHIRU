"""The calibration pool, projects drawn from it, the text an allocator reads, and scoring."""
import random

import pytest

from hashiru.runtime import outline
from purse.jobs import BINS, LEVELS, InfeasibleCell, register_family
from purse.market import MARKET
from purse.projects import (
    ALL,
    build_pool,
    draw_hidden,
    fold_of,
    level_bin,
    make_project,
    parse_answers,
    project_text,
    score,
    with_budget,
)
from purse.table import CellStat
from tests.helpers import job


@pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
def test_draw_hidden_agrees_with_the_visible_level_at_rate_rho(rho):
    rng = random.Random(1)
    draws = [(lvl := rng.choice(LEVELS), draw_hidden(lvl, rho, rng)) for _ in range(10_000)]
    agree = sum(level_bin(lvl) == h for lvl, h in draws) / len(draws)
    assert abs(agree - (rho + (1 - rho) / 3)) < 0.02


def full_pool(per_cell: int = 4) -> list:
    return [job(f"{f}-l{lvl}-{h}-{i:03d}", family=f, level=lvl, hidden=h, answer=str(i))
            for f in ALL for lvl in LEVELS for h in BINS for i in range(per_cell)]


def test_projects_never_reuse_a_job_and_stay_in_their_fold():
    pool, used = full_pool(), set()
    projects = [make_project(pool, fold=s % 2, rho=0.5, seed=s, used=used) for s in range(6)]
    uids = [j.uid for p in projects for j in p.jobs]
    assert len(uids) == len(set(uids)) == 48
    for p in projects:
        assert [j.id for j in p.jobs] == [f"j{i}" for i in range(1, 9)]
        assert all(fold_of(j) == int(p.id.split("-f")[1][0]) for j in p.jobs)
        assert sorted(j.family for j in p.jobs) == sorted(ALL * 2)
        assert {j.value for j in p.jobs} <= {1, 2, 4}
    assert make_project(pool, 0, 0.5, seed=3, used=set()) == make_project(pool, 0, 0.5, seed=3, used=set())


class Flaky:
    name, checkable = "flaky", False

    def generate(self, level, hidden, rng):
        if (level, hidden) == (2, "high"):
            raise InfeasibleCell("no")
        return "t", "1", {}

    def correct(self, answer, response):
        return answer == response


def test_pool_skips_infeasible_cells_and_reports_them():
    register_family(Flaky())
    jobs, skipped = build_pool(seed=1, per_cell=2, families=("flaky",))
    assert skipped == ["flaky-l2-high"]
    assert len(jobs) == 2 * (len(LEVELS) * len(BINS) - 1)
    assert not any(j.uid.startswith("flaky-l2-high") for j in jobs)


def test_parse_answers_reads_fenced_and_messy_json():
    assert parse_answers('Done.\n```json\n{"j1": "42", "J3": 7}\n```\nThanks') == {"j1": "42", "j3": "7"}
    assert parse_answers('{"j1": "a", "j2": "b",}') == {"j1": "a", "j2": "b"}
    grid = parse_answers('{"j4": {"1": {"color": "red"}}}')
    assert grid["j4"].replace(" ", "") == '{"1":{"color":"red"}}'
    assert parse_answers("I could not do it.") == {}
    assert parse_answers('{"j1": "x"} and later {"j2": "y"}') == {"j2": "y"}


def test_score_weights_by_value():
    jobs = (job("a", id="j1", value=1, answer="1"), job("b", id="j2", value=4, answer="2"))
    p = make_project(full_pool(), 0, 1.0, seed=0, used=set())
    p = p.__class__(id="t", jobs=jobs, rho=1.0)
    assert score(p, {"j2": "2"}) == 0.8 and score(p, {"j1": "1", "j2": "9"}) == 0.2 and score(p, {}) == 0.0


def test_project_text_lists_jobs_market_and_stats():
    p = with_budget(make_project(full_pool(), 0, 0.5, seed=2, used=set()), 0.1234)
    j = p.jobs[0]
    stats = {(j.family, j.level, None, "gpt-6-luna"): CellStat(0.75, 0.0021, 8)}
    text = project_text(p, MARKET, stats)
    assert "$0.1234" in text and "12288 MB" in text and j.prompt in text
    assert f"### j1 ({j.family}, level {j.level}, value {j.value})" in text
    assert "a2a/calc: an outside agent, $0.001 per message, arithmetic jobs only" in text
    assert "gpt-6-luna: a cloud model, $0.10 / $0.50 per million input / output tokens" in text
    assert "| " + j.family + f" | {j.level} | gpt-6-luna | 75% | $0.0021 |" in text
    assert "measured" not in project_text(p, MARKET, None)
    assert 'e.g. {"j1": "..."}.' in text


def test_the_ceo_still_reads_the_market_and_answer_format_when_long_jobs_are_held_back():
    # under request_by_reference the CEO reads each long section cut to its first lines, and a section runs to the
    # next heading: whatever followed the last job would be cut with it (the first round-2 runs lost the hire list,
    # the statistics and the answer format this way)
    p = with_budget(make_project(full_pool(), 0, 0.5, seed=2, used=set()), 0.1234)
    j = p.jobs[-1]
    stats = {(j.family, j.level, None, "gpt-6-luna"): CellStat(0.75, 0.0021, 8)}
    seen = outline(project_text(p, MARKET, stats), 1)  # every section held back
    assert "Who you can hire:" in seen and "a2a/calc: an outside agent, $0.001 per message" in seen
    assert "| " + j.family + f" | {j.level} | gpt-6-luna | 75% | $0.0021 |" in seen
    assert 'Answer with one JSON object mapping each job id to its answer, e.g. {"j1": "..."}.' in seen
    assert "$0.1234" in seen and all(f"### {x.id} (" in seen for x in p.jobs)
