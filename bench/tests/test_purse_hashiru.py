"""Running allocators through a live HASHIRU hub, with scripted models: no keys, no Ollama, no money."""
import asyncio
import json
from types import SimpleNamespace

from hashiru.hub import Hub
from hashiru.planning import Hire, Plan
from purse.hashiru_run import HubMarket, bench_settings, plan_to_allocation, propose_split, run_hashiru, variant_text
from purse.jobs import WORKER_PROMPT, make_job
from purse.market import CEO_MODEL, MARKET, Option
from purse.projects import Project
from tests.conftest import FakeOllama

CLOUD, LOCAL = "gpt-4.1-mini", "ollama_chat/tiny"
TEST_MARKET = (Option(LOCAL, "local"), Option(CLOUD, "cloud", input_usd_per_m=0.4, output_usd_per_m=1.6))


def call(name, args, cid):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


class ScriptedLLM:
    """Workers answer the job's number; the CEO follows `script`, one entry per turn."""

    def __init__(self, script=()):
        self.script, self.models = list(script), []

    async def chat(self, model, messages, tools=None):
        self.models.append(model)
        if messages[0]["content"] == WORKER_PROMPT:
            answer = reply(f"<answer>{messages[-1]['content'].split()[1].rstrip('.')}</answer>")
            return answer, 0.0 if model.startswith("ollama_chat/") else 0.002  # as LLM.chat: local calls cost $0
        turn = sum(1 for m in messages if m["role"] == "tool")
        return self.script[turn], 0.001


def project(tmp_path, budget=0.5):
    job = make_job("fake", 2, "low", seed=4, uid="fake-l2-low-000")
    from dataclasses import replace
    return Project("p", (replace(job, id="j1", value=2),), 0.5, budget_usd=budget, memory_mb=1000), job


def hub_for(tmp_path, p, llm, port, sizes=None, **kw):
    s = bench_settings(p, TEST_MARKET, out_dir=tmp_path, max_usd=1.0, port=port, ceo_model=CLOUD, **kw)
    return Hub(s, llm=llm, ollama=FakeOllama(sizes))  # type: ignore[arg-type]


def test_bench_settings_turn_off_web_and_approvals(tmp_path, fake_family):
    p, _ = project(tmp_path, budget=0.25)
    s = bench_settings(p, MARKET, out_dir=tmp_path, max_usd=2.0, peer_urls={"a2a/calc": "http://127.0.0.1:9301/"})
    assert s.ceo_model == CEO_MODEL and s.budget_usd == 0.25 and s.memory_mb == 1000 and s.max_usd == 2.0
    assert not s.web_tools and (s.approve_hire_usd, s.approve_hire_mb, s.approve_turn_usd) == (0, 0, 0)
    assert s.models == ("ollama_chat/qwen3.5:4b", "ollama_chat/gemma4:12b", "gpt-6-luna", "meta/muse-spark-1.3")
    assert [(a.name, a.price_usd_per_message) for a in s.a2a_peers] == [("calc", 0.001)]
    assert s.mcp_servers[0].name == "purse" and s.mcp_servers[0].args == ("-m", "purse.peers.grid_check")
    assert str(tmp_path) in s.trace_dir and s.task_timeout == p.deadline_s


def test_hub_market_hires_asks_fires_and_reports_cost(tmp_path, fake_family):
    p, job = project(tmp_path)
    llm = ScriptedLLM()
    hub = hub_for(tmp_path, p, llm, 9361)

    async def body():
        async with hub.serving():
            session = hub.new_session("cal", trace=False)
            async with HubMarket(hub, session, reserve_usd=0.25) as market:
                cloud = await market.attempt(job, CLOUD)
                local = await market.attempt(job, LOCAL)
            return cloud, local, session
    cloud, local, session = asyncio.run(body())
    assert cloud.correct and cloud.cost_usd == 0.002 and not cloud.refused and cloud.seconds >= 0
    assert local.correct and local.cost_usd == 0.0
    assert not any(a.live for name, a in session.ledger.agents.items() if name != "ceo")  # each was fired
    assert llm.models == [CLOUD, LOCAL]


def test_hub_market_reports_a_refused_hire(tmp_path, fake_family):
    p, job = project(tmp_path)
    hub = hub_for(tmp_path, p, ScriptedLLM(), 9362, sizes={"tiny": 5000})

    async def body():
        async with hub.serving(), HubMarket(hub, hub.new_session("cal", trace=False), reserve_usd=0.25) as market:
            return await market.attempt(job, LOCAL)
    out = asyncio.run(body())
    assert out.refused and "5000 MB" in out.refused and not out.correct


def test_run_hashiru_reads_the_ceos_json_answer(tmp_path, fake_family):
    p, job = project(tmp_path)
    script = [reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": CLOUD, "prompt": WORKER_PROMPT,
                                               "budget_usd": 0.1}, "c1")]),
              reply(calls=[call("ask_agent", {"name": "w1", "text": job.prompt}, "c2")]),
              reply(calls=[call("fire_agent", {"name": "w1"}, "c3")]),
              reply(f'Done: ```json\n{{"J1": "{job.answer}"}}\n```')]
    hub = hub_for(tmp_path, p, ScriptedLLM(script), 9363)

    async def body():
        async with hub.serving():
            return await run_hashiru(hub, p, "do the project")
    run = asyncio.run(body())
    assert run.answers == {"j1": job.answer} and abs(run.spent_usd - (4 * 0.001 + 0.002)) < 1e-9
    assert abs(run.ceo_usd - 0.004) < 1e-9 and run.trace.endswith(".jsonl")


def test_propose_split_reads_the_plan(tmp_path, fake_family):
    p, _ = project(tmp_path)
    plan = {"summary": "one worker", "hires": [{"name": "J1", "role": "worker", "model": CLOUD, "budget_usd": 0.05,
                                                "task": "solve j1"}]}
    hub = hub_for(tmp_path, p, ScriptedLLM([reply(calls=[call("propose_plan", plan, "c1")]), reply("planned")]), 9364)

    async def body():
        async with hub.serving():
            return await propose_split(hub, p, "plan it")
    alloc, spent = asyncio.run(body())
    assert alloc == {"j1": CLOUD} and abs(spent - 0.002) < 1e-9


def test_plan_to_allocation_maps_hires_to_jobs_and_the_rest_to_the_ceo(fake_family):
    jobs = tuple(__import__("dataclasses").replace(make_job("fake", 1, "low", seed=i, uid=f"u{i}"), id=f"j{i}")
                 for i in (1, 2))
    p = Project("p", jobs, 0.5)
    plan = Plan(hires=(Hire("J1", "worker", "gpt-6-luna", "cloud", 0.1, 0, "t"),
                       Hire("helper", "worker", "ollama_chat/qwen3.5:4b", "local", 0, 100, "t")))
    assert plan_to_allocation(plan, p) == {"j1": "gpt-6-luna", "j2": CEO_MODEL}


def test_variant_text():
    assert variant_text("base", "") == "base"
    assert "do not give a difficulty" in variant_text("base", "names-models")
    assert "named after the job id" in variant_text("base", "split")


def test_hub_market_can_reserve_what_is_left(tmp_path, fake_family):
    p, job = project(tmp_path, budget=0.01)
    hub = hub_for(tmp_path, p, ScriptedLLM(), 9365, turn_reserve_usd=0.0)  # the CEO never thinks here

    async def body():
        async with hub.serving():
            session = hub.new_session("cal", trace=False)
            async with HubMarket(hub, session, reserve_usd=None) as market:
                first = await market.attempt(job, CLOUD)  # reserves all $0.01, spends 0.002, fire refunds the rest
                second = await market.attempt(job, CLOUD)
            return first, second, session.ledger.left()["ceo"]
    first, second, left = asyncio.run(body())
    assert first.correct and second.correct and abs(left - 0.006) < 1e-9
