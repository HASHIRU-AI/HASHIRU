"""The outside agents PURSE offers (the example calc, and csp) and the grid_check tool."""
import asyncio
import json
import random
import sys
from pathlib import Path

import httpx2
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from hashiru import runtime
from hashiru.config import A2APeer, McpServer
from hashiru.hub import Hub
from purse.families.arithmetic import ARITHMETIC
from purse.families.grids import GRIDS
from purse.jobs import make_job
from purse.peers import csp_agent, grid_check
from tests.conftest import FakeOllama
from tests.helpers import NoLLM, example, served

BENCH = Path(__file__).resolve().parents[1]


def settings(tmp_path, **overrides):
    from hashiru.config import Settings
    return Settings(**({"ceo_model": "gpt-4.1-mini", "models": ("gpt-4.1-mini",), "data_dir": str(tmp_path / "d"),
                        "trace_dir": str(tmp_path / "t"), "obfuscate_with": "patterns", "approve_hire_usd": 0,
                        "approve_hire_mb": 0, "approve_turn_usd": 0} | overrides))


def test_csp_agent_solves_a_generated_grid():
    job = make_job("grids", 4, "high", seed=3, uid="g")
    assert GRIDS.correct(job.answer, csp_agent.solve_text(f"Please solve this.\n\n{job.prompt}"))
    assert csp_agent.solve_text("hello").startswith("error:")


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_calc_agent_answers_the_canonical_arithmetic_prompt(level):
    job = make_job("arithmetic", level, "high", seed=level, uid="a")
    assert ARITHMETIC.correct(job.answer, example("calc_a2a_agent").calculate(job.prompt))


def test_calc_agent_refuses_level_5_expressions_as_too_long():
    job = make_job("arithmetic", 5, "mid", seed=1, uid="a")
    assert example("calc_a2a_agent").calculate(job.prompt).startswith("error:")


def test_csp_agent_is_hired_and_paid_like_any_peer(tmp_path):
    hub = Hub(settings(tmp_path, port=9354, a2a_peers=(A2APeer("csp", "http://127.0.0.1:9353/"),)),
              llm=NoLLM(), ollama=FakeOllama())  # type: ignore[arg-type]
    job = make_job("grids", 2, "low", seed=5, uid="g")

    async def body():
        async with served(csp_agent.build_app("http://127.0.0.1:9353/", 0.002), 9353), hub.serving():
            session = hub.new_session("t", trace=False)
            await hub.tools.hire_agent(runtime_ctx(session), "csp", "worker", "a2a/csp", "", 0.01)
            answer = await runtime.ask_agent(hub, session, "ceo", "csp", job.prompt)
            return answer, session.ledger.agents["csp"].spent
    answer, spent = asyncio.run(body())
    assert GRIDS.correct(job.answer, answer) and spent == pytest.approx(0.002)


def runtime_ctx(session):
    from types import SimpleNamespace
    return SimpleNamespace(headers={"authorization": f"Bearer {session.tokens['ceo']}"})


def test_grid_check_limits_calls_per_puzzle():
    job = make_job("grids", 2, "mid", seed=2, uid="g")
    counts: dict[str, int] = {}
    assert grid_check.check(job.prompt, job.answer, counts) == "valid"
    assert grid_check.check(job.prompt, '{"1": {"color": "nope"}}', counts) == "invalid"
    assert grid_check.check(job.prompt, job.answer, counts) == "valid"
    assert grid_check.check(job.prompt, job.answer, counts) == "limit reached"
    other = make_job("grids", 2, "mid", seed=3, uid="g2")
    assert grid_check.check(other.prompt, other.answer, counts) == "valid"
    assert grid_check.check("no puzzle here", "{}", counts) == "invalid: no grid puzzle in the text"


def test_grid_check_is_offered_to_agents_over_mcp(tmp_path):
    server = McpServer("purse", command=sys.executable, args=("-m", "purse.peers.grid_check"),
                       env=(("PYTHONPATH", str(BENCH)),), labels=("trusted",), egress=False)
    hub = Hub(settings(tmp_path, port=9355, mcp_servers=(server,)), llm=NoLLM(), ollama=FakeOllama())  # type: ignore[arg-type]
    job = make_job("grids", 1, "low", seed=random.Random(0).randint(0, 99), uid="g")

    async def body():
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            async with (httpx2.AsyncClient(headers=runtime.bearer(session, "ceo")) as http,
                        streamable_http_client(f"{hub.settings.internal_url}/mcp", http_client=http) as (r, w, *_),
                        ClientSession(r, w) as mcp):
                await mcp.initialize()  # the path an agent takes: its own token, the hub's /mcp
                names = {t.name for t in (await mcp.list_tools()).tools}
                result = await mcp.call_tool("purse__grid_check", {"puzzle": job.prompt, "answer": job.answer})
                return names, " ".join(getattr(c, "text", "") for c in result.content)
    names, result = asyncio.run(body())
    assert "purse__grid_check" in names and json.loads(result) == {"result": "valid"}  # the gateway wraps results
