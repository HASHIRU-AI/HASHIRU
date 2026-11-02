"""What a live HASHIRU run did, read back from its trace; and the statistics the results use."""
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

from hashiru.hub import Hub
from purse import metrics, traces
from purse.hashiru_run import bench_settings, run_hashiru
from purse.jobs import WORKER_PROMPT, make_job
from purse.market import Option
from purse.projects import Project
from purse.replay import ReplayMarket
from purse.solvers import Attempt
from purse.table import OutcomeTable, Record
from tests.conftest import FakeOllama
from tests.test_purse_hashiru import CLOUD, LOCAL, TEST_MARKET, call, reply


class EscalatingLLM:
    """The local worker answers nothing (so it escalates); the cloud one answers the job's number."""

    def __init__(self, script):
        self.script = script

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == WORKER_PROMPT:
            if model == LOCAL:
                return reply(""), 0.0
            return reply(f"<answer>{messages[1]['content'].split()[1].rstrip('.')}</answer>"), 0.002
        return self.script[sum(1 for m in messages if m["role"] == "tool")], 0.001


@pytest.fixture
def run(tmp_path, fake_family):
    jobs = tuple(replace(make_job("fake", 2, "low", seed=i, uid=f"fake-l2-low-00{i}"), id=f"j{i}", value=1)
                 for i in (1, 2))
    p = Project("p", jobs, 0.5, budget_usd=0.5, memory_mb=1000)
    script = [reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "difficulty": "easy",
                                               "prompt": WORKER_PROMPT, "budget_usd": 0.1}, "c1")]),
              reply(calls=[call("ask_agent", {"name": "w1", "text": jobs[0].prompt}, "c2")]),
              reply(calls=[call("fire_agent", {"name": "w1"}, "c3")]),
              reply(calls=[call("hire_agent", {"name": "w2", "role": "worker", "model": CLOUD,
                                               "prompt": WORKER_PROMPT, "budget_usd": 0.05}, "c4")]),
              reply(calls=[call("ask_agent", {"name": "w2", "text": "Please do j2 for me."}, "c5")]),
              reply(f'{{"j1": "{jobs[0].answer}"}}')]
    settings = bench_settings(p, TEST_MARKET, out_dir=tmp_path, max_usd=1.0, port=9371, ceo_model=CLOUD)
    hub = Hub(settings, llm=EscalatingLLM(script), ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            return await run_hashiru(hub, p, "do it")
    live = asyncio.run(body())
    return p, traces.read(Path(live.trace))


def test_hires_spend_and_refunds(run):
    _, events = run
    found = {h.name: h for h in traces.hires(events)}
    assert set(found) == {"w1", "w2"} and found["w1"].boss == "ceo" and found["w1"].difficulty == "easy"
    assert found["w1"].model == LOCAL and found["w1"].fired and not found["w2"].fired
    assert found["w1"].spent_usd == pytest.approx(0.002) and found["w2"].spent_usd == pytest.approx(0.002)
    assert traces.slack(events) == pytest.approx(0.1 - 0.002)  # only fired hires count


def test_live_allocation_follows_escalations(run):
    p, events = run
    assert traces.escalations(events) == [("w1", LOCAL, CLOUD)]
    assert traces.jobs_of(events, p) == {"j1": ["w1"], "j2": ["w2"]}
    assert traces.live_allocation(events, p) == {"j1": [LOCAL, CLOUD], "j2": [CLOUD]}


def test_declared_difficulty_per_job(run):
    p, events = run
    assert traces.declared(events, p) == {"j1": "easy", "j2": ""}


def test_prompt_fidelity(run):
    p, events = run
    assert traces.prompt_fidelity(events, p) == 0.5  # j1 got its text verbatim; j2 only its id


def test_replay_live_scores_the_last_model_tried(run):
    p, events = run
    table = OutcomeTable(None)
    for j in p.jobs:
        table.add(Record(j.uid, CLOUD, 0, f"<answer>{j.answer}</answer>", True, 0.003, 1.0))
    result = asyncio.run(traces.replay_live(p, traces.live_allocation(events, p), ReplayMarket(table), TEST_MARKET))
    assert set(result.answers) == {"j1", "j2"} and result.spent_usd == pytest.approx(0.006)


def test_spearman_and_bootstrap_on_known_values():
    assert metrics.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert metrics.spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert metrics.spearman([1, 1, 2], [1, 2, 3]) == pytest.approx(0.8660254, abs=1e-6)  # tied ranks averaged
    lo, hi = metrics.bootstrap_ci([0.5] * 10)
    assert lo == hi == 0.5
    lo, hi = metrics.bootstrap_ci([0.0, 1.0] * 50, seed=1)
    assert lo < 0.5 < hi and hi - lo < 0.25
    lo, hi = metrics.paired_ci([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert lo == hi == 1.0
    assert metrics.efficiency(0.6, 0.2, 1.0) == pytest.approx(0.5) and metrics.efficiency(0.5, 0.5, 0.5) == 0.0
    assert metrics.confusion({"j1": "easy", "j2": "hard", "j3": ""}, {"j1": "low", "j2": "low", "j3": "high"}) == {
        ("easy", "low"): 1, ("hard", "low"): 1}
    attempts = [Attempt("j1", "a2a/calc", None), Attempt("j2", CLOUD, None)]  # type: ignore[arg-type]
    assert metrics.buy_share(attempts, (*TEST_MARKET, Option("a2a/calc", "agent", 0.01))) == 0.5


def test_a_hire_that_never_ran_a_model_tried_nothing():
    p = Project("p", (replace(make_job("arithmetic", 1, "low", seed=1, uid="a"), id="j1"),), 0.5)
    events = [{"kind": "tool", "agent": "ceo", "name": "hire_agent", "args": {"name": "w1", "difficulty": "easy",
                                                                             "prompt": "", "budget_usd": 0.1},
               "result": {"hired": "w1"}},
              {"kind": "msg", "agent": "ceo", "to": "w1", "text": "please do j1", "reply": False}]
    assert traces.live_allocation(events, p) == {}
