"""Regressions from the Phase 4 code review: money, credentials, deadlines and cleanup around A2A peers."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest
import uvicorn
from a2a.types import TaskState
from a2a_peer import peer
from conftest import FakeOllama, ctx, make_settings
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from hashiru import hub as hub_module
from hashiru import runtime
from hashiru.config import A2APeer, load_settings
from hashiru.hub import Hub
from hashiru.peers import COST_KEY, DEADLINE_KEY, BoundedTaskStore, Price, send
from hashiru.session import USER


def reply(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


def peer_hub(tmp_path, port, **peer_kw):
    peers = (A2APeer("p1", f"http://127.0.0.1:{port}/", **peer_kw),)
    return Hub(make_settings(tmp_path, port=port + 1, a2a_peers=peers), ollama=FakeOllama())  # type: ignore[arg-type]


async def hire_peer(hub, session, budget=0.05, name="r", caller="ceo"):
    return await hub.tools.hire_agent(ctx(session, caller), name, "worker", "a2a/p1", "", budget)  # type: ignore[arg-type]


# H1: a peer's reported cost is bounded

@pytest.mark.parametrize("report", [1e300, "1e9", "inf", -5, "nonsense"])
def test_a_peers_reported_cost_never_exceeds_the_pool(tmp_path, report):
    hub = peer_hub(tmp_path, 9310)

    async def body():
        async with peer(9310, price=Price(per_message_usd=0.001), report=report), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session, budget=0.05)
            answer = await runtime.ask_agent(hub, session, "ceo", "r", "hi")
            return answer, session.ledger.agents["r"].spent
    answer, spent = asyncio.run(body())
    assert answer.startswith("echo:") and 0.001 <= spent <= 0.05  # capped at what the pool held
    assert hub.wallet.spent <= 0.05
    hub.wallet.check()  # the wallet, shared by every session, is not exhausted by one peer


# H2: credentials go only to the configured URL

def test_a_card_cannot_redirect_requests_or_credentials(tmp_path):
    hits: list[str] = []

    async def sink(request):
        hits.append(request.headers.get("authorization", ""))
        return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {}})

    async def body():
        srv = uvicorn.Server(uvicorn.Config(Starlette(routes=[Route("/{p:path}", sink, methods=["GET", "POST"])]),
                                            host="127.0.0.1", port=9312, log_level="warning"))
        task = asyncio.create_task(srv.serve())
        while not srv.started:
            await asyncio.sleep(0.05)
        hub = peer_hub(tmp_path, 9313, headers=(("Authorization", "Bearer PEER-SECRET"),))
        try:
            async with (peer(9313, price=Price(0.001), token="PEER-SECRET", declared_url="http://127.0.0.1:9312/admin"),
                        hub.serving()):
                session = hub.new_session("t", trace=False)
                await hire_peer(hub, session)
                return await runtime.ask_agent(hub, session, "ceo", "r", "hi")
        finally:
            srv.should_exit = True
            await task
    assert asyncio.run(body()) == "echo: hi" and hits == []


# H3: a cloud manager keeps money for its own turns

def test_a_cloud_manager_keeps_enough_to_carry_on(hub, session):
    args = {"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "prompt": "p"}
    assert "keep $0.01" in asyncio.run(hub.tools.hire_agent(ctx(session, "ceo"), budget_usd=0.5, **args))["error"]
    assert "hired" in asyncio.run(hub.tools.hire_agent(ctx(session, "ceo"), budget_usd=0.49, **args))
    assert runtime.spent_out(hub, session, session.ledger.agents["ceo"]) is None


# M4: concurrent asks to one peer cannot overdraw its pool

def test_concurrent_asks_to_a_peer_cannot_overdraw(tmp_path):
    hub = peer_hub(tmp_path, 9314)

    async def body():
        async with peer(9314, price=Price(per_message_usd=0.01), delay=0.1) as (_, echo), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session, budget=0.01)
            answers = await asyncio.gather(*(runtime.ask_agent(hub, session, "ceo", "r", f"q{i}") for i in range(5)))
            return answers, session.ledger.left()["r"], len(echo.seen)
    answers, left, sent = asyncio.run(body())
    assert sum(a.startswith("echo:") for a in answers) == 1 and sent == 1 and left >= -1e-9


# M5: a token-priced peer is billed by length

def test_a_token_priced_peer_is_not_free(tmp_path):
    hub = peer_hub(tmp_path, 9316)

    async def body():
        async with peer(9316, price=Price(0.0, 1e-4, 1e-4)), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session)
            await runtime.ask_agent(hub, session, "ceo", "r", "a question of some length")
            return session.ledger.agents["r"].spent
    assert asyncio.run(body()) > 0


# M6: a peer relay is bounded by the deadline and stopped by a fire

def test_a_slow_peer_is_cut_off_at_the_deadline_and_by_a_fire(tmp_path):
    peers = (A2APeer("p1", "http://127.0.0.1:9318/"),)
    hub = Hub(make_settings(tmp_path, port=9319, a2a_peers=peers, task_timeout=2), ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with peer(9318, price=Price(0.001), delay=30), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session, name="slow")
            await hire_peer(hub, session, name="fired")
            began = time.monotonic()
            timed_out = await runtime.ask_agent(hub, session, "ceo", "slow", "hi")
            took = time.monotonic() - began
            asking = asyncio.create_task(runtime.ask_agent(hub, session, "ceo", "fired", "hi"))
            while not hub._running.get((session.id, "fired")):
                await asyncio.sleep(0.05)
            await hub.tools.fire_agent(ctx(session, "ceo"), "fired")  # type: ignore[arg-type]
            return timed_out, took, await asyncio.wait_for(asking, 10), dict(hub._running)
    timed_out, took, fired, running = asyncio.run(body())
    assert timed_out == "failed: out of time" and took < 5
    assert "stopped" in fired and running == {}


# M7: peer replies are labeled and capped

def test_peer_replies_are_untrusted_and_capped(tmp_path):
    hub = peer_hub(tmp_path, 9320)

    async def body():
        async with peer(9320, price=Price(0.001), reply_size=50_000), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session)
            answer = await runtime.ask_agent(hub, session, "ceo", "r", "hi")
            return answer, session.trace.labels("ceo", session.id)
    answer, labels = asyncio.run(body())
    assert len(answer) == 20_000 and labels == {"untrusted"}


# M8: empty messages are refused before any work

def test_an_empty_message_is_refused(hub, session):
    session.ledger.add("w1", "worker", "cloud", "gpt-4.1-mini", "p", "ceo", 1, 0.1)
    assert asyncio.run(runtime.ask_agent(hub, session, "ceo", "w1", "  ")) == "error: the message is empty"


# M9: an LLM call in flight when its run is cancelled is still charged

def test_a_cancelled_call_is_charged_when_it_returns(hub, session):
    class SlowLLM:
        async def chat(self, model, messages, tools=None):
            await asyncio.sleep(0.2)
            return reply("late"), 0.003
    hub.llm = SlowLLM()
    session.ledger.add("w1", "worker", "cloud", "gpt-4.1-mini", "p", "ceo", 1, 0.1)

    async def body():
        run = asyncio.create_task(runtime.run_agent(hub, session, "w1", "go"))
        await asyncio.sleep(0.05)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        await asyncio.sleep(0.3)
    asyncio.run(body())
    late = [e for e in session.trace.events if e["kind"] == "llm"]
    assert session.ledger.agents["w1"].spent == 0.003 and late[0].get("after_stop") is True


# M10: local hires check memory and load one at a time

def test_two_local_hires_cannot_both_take_the_last_memory(tmp_path, monkeypatch):
    hub = Hub(make_settings(tmp_path, models=("gpt-4.1-mini", "ollama_chat/tiny", "ollama_chat/tiny2")),
              ollama=FakeOllama(delay=0.05))  # type: ignore[arg-type]

    async def sizes():
        return {"tiny": 600, "tiny2": 600}
    hub.ollama.sizes = sizes
    session = hub.new_session("t", trace=False)
    loaded = lambda: sum(a.gpu_mb for a in session.ledger.agents.values() if a.live)  # noqa: E731
    monkeypatch.setattr(hub_module, "free_mb", lambda: 1000 - loaded())
    hub.ollama.load = lambda model: asyncio.sleep(0.05, result=600)

    async def both():
        c = ctx(session, "ceo")
        base = {"role": "worker", "prompt": "p", "budget_usd": 0.0}
        return await asyncio.gather(hub.tools.hire_agent(c, name="a", model="ollama_chat/tiny", **base),  # type: ignore[arg-type]
                                    hub.tools.hire_agent(c, name="b", model="ollama_chat/tiny2", **base))  # type: ignore[arg-type]
    results = asyncio.run(both())
    assert sorted("hired" in r for r in results) == [False, True]


# M11: every peer hire is its own conversation

def test_each_peer_hire_has_its_own_context(tmp_path):
    hub = peer_hub(tmp_path, 9322)

    async def body():
        async with peer(9322, price=Price(0.001)) as (_, echo), hub.serving():
            session = hub.new_session("t", trace=False)
            for name in ("ra", "rb"):
                await hire_peer(hub, session, name=name)
                await runtime.ask_agent(hub, session, "ceo", name, "hi")
            return echo.contexts
    contexts = asyncio.run(body())
    assert len(set(contexts)) == 2


# L13: a peer task that did not complete is reported as a failure

def test_a_peer_task_that_failed_is_reported_as_failed(tmp_path):
    hub = peer_hub(tmp_path, 9324)

    async def body():
        async with peer(9324, price=Price(0.001), state=TaskState.TASK_STATE_FAILED), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session)
            return await runtime.ask_agent(hub, session, "ceo", "r", "hi")
    assert asyncio.run(body()).startswith("failed: p1 ended in state TASK_STATE_FAILED")


# L14 and L15: reported costs include hires; a caller cannot extend the deadline

def test_reported_cost_covers_hires_and_deadlines_are_clamped(tmp_path):
    class LLM:
        async def chat(self, model, messages, tools=None):
            if messages[0]["content"] == "slow":
                await asyncio.sleep(30)
            turn = sum(1 for m in messages if m["role"] == "tool")
            if messages[0]["content"] == "p":
                return reply("worker answer"), 0.001
            if messages[1]["content"] == "cost":
                script = [reply(calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
                              name="hire_agent", arguments=json.dumps({"name": "w1", "role": "worker",
                                                                       "model": "gpt-4.1-mini", "prompt": "p",
                                                                       "budget_usd": 0.1})))]),
                          reply(calls=[SimpleNamespace(id="c2", function=SimpleNamespace(
                              name="ask_agent", arguments=json.dumps({"name": "w1", "text": "go"})))]),
                          reply("done")]
                return script[turn], 0.002
            await asyncio.sleep(30)  # the "far future" request
            return reply("never"), 0.0
    hub = Hub(make_settings(tmp_path, port=9326, task_timeout=2), llm=LLM(), ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            headers = {"Authorization": f"Bearer {session.tokens[USER]}"}
            url = hub.agent_url(session, "ceo")
            _, metadata, _ = await send(url, headers, "cost", session.id, 30)
            began = time.monotonic()
            late, _, _ = await send(url, headers, "far", session.id, 30, {DEADLINE_KEY: 1e18})
            return metadata[COST_KEY], late, time.monotonic() - began
    cost, late, took = asyncio.run(body())
    assert cost == pytest.approx(3 * 0.002 + 0.001)  # the CEO's three turns and its worker's one
    assert late == "failed: out of time" and took < 5


# L16: config values that would break a hub are refused

@pytest.mark.parametrize("bad", ['+a2a_peers.x={url: "https://a", price_usd_per_message: .nan}',
                                 '+a2a_peers.x={url: "https://user:pw@a/"}', '+a2a_peers.x={url: "https://a/?key=1"}',
                                 '+a2a_peers.x={url: "https:///"}', "task_timeout=0", "task_timeout=-5",
                                 "budget_usd=-1"])
def test_unusable_settings_are_refused(bad):
    with pytest.raises(ValueError):
        load_settings([bad])


# L17: a pool that holds exactly N messages pays for N

def test_rounding_does_not_lose_a_message(tmp_path):
    hub = peer_hub(tmp_path, 9328)

    async def body():
        async with peer(9328, price=Price(per_message_usd=0.1)) as (_, echo), hub.serving():
            session = hub.new_session("t", trace=False)
            await hire_peer(hub, session, budget=0.3)
            for i in range(3):
                await runtime.ask_agent(hub, session, "ceo", "r", f"q{i}")
            return len(echo.seen)
    assert asyncio.run(body()) == 3


# L18: a local hire cancelled mid-load leaves no model loaded

def test_a_cancelled_local_hire_unloads_its_model(hub, session, ollama):
    hub.ollama = loading = FakeOllama(delay=0.2)

    async def body():
        hiring = asyncio.create_task(hub.tools.hire_agent(ctx(session, "ceo"), "w1", "worker",  # type: ignore[arg-type]
                                                          "ollama_chat/tiny", "p", 0.1))
        await asyncio.sleep(0.05)
        hiring.cancel()
        with pytest.raises(asyncio.CancelledError):
            await hiring
        await asyncio.sleep(0.1)
    asyncio.run(body())
    assert session.ledger.get("w1") is None and loading.unloaded == ["ollama_chat/tiny"]


# L19: an agent's task store does not grow without bound

def test_finished_tasks_are_not_kept_forever():
    from a2a.server.context import ServerCallContext
    from a2a.types import Task, TaskStatus
    store, context = BoundedTaskStore(keep=3), ServerCallContext()

    async def body():
        for i in range(6):
            await store.save(Task(id=f"t{i}", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED)), context)
        return [await store.get(f"t{i}", context) is not None for i in range(6)]
    assert asyncio.run(body()) == [False, False, False, True, True, True]
