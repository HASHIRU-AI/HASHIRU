"""Phase 4: budget enforcement, modes, deadlines and cancellation, remote A2A peers, and the cost extension."""
import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from a2a_peer import peer
from conftest import FakeOllama, ctx, make_settings

from hashiru import hub as hub_module
from hashiru import runtime
from hashiru.config import A2APeer, load_settings
from hashiru.hub import Hub
from hashiru.peers import COST_EXTENSION, COST_KEY, DEADLINE_KEY, Price, fetch_card, price_of, send
from hashiru.session import USER


def hire(hub, session, caller="ceo", **kw):
    args = {"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "prompt": "p", "budget_usd": 0.1} | kw
    return asyncio.run(hub.tools.hire_agent(ctx(session, caller), **args))  # type: ignore[arg-type]


def call(name, args, cid):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


# ---- budget enforcement ----

def test_a_hire_cannot_reserve_more_than_is_left(hub, session):
    assert "not enough to reserve" in hire(hub, session, budget_usd=0.6)["error"]  # the CEO has 0.50
    assert "keep $0.01" in hire(hub, session, budget_usd=0.5)["error"]  # a cloud CEO keeps some for its own turns
    assert "hired" in hire(hub, session, budget_usd=0.49)
    assert "not enough to reserve" in hire(hub, session, name="w2", budget_usd=0.01)["error"]


def test_without_enforcement_hires_only_record_overspending(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, enforce_budget=False), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    assert "hired" in hire(hub, session, budget_usd=5.0)
    assert session.ledger.left()["ceo"] < 0


def test_a_local_model_must_fit_in_free_memory(hub, session, monkeypatch):
    monkeypatch.setattr(hub_module, "free_mb", lambda: 10)  # "tiny" needs 50 MB
    assert "needs 50 MB" in hire(hub, session, model="ollama_chat/tiny")["error"]
    session.ledger.add("other", "worker", "local", "ollama_chat/tiny", "p", "ceo", 1, 0.0)  # already loaded
    assert "hired" in hire(hub, session, model="ollama_chat/tiny")


def test_an_agent_with_a_spent_pool_makes_no_more_cloud_calls(hub, session):
    calls = []

    class LLM:
        async def chat(self, model, messages, tools=None):
            calls.append(model)
            return reply("hi"), 0.0
    hub.llm = LLM()
    hire(hub, session, budget_usd=0.1)
    session.ledger.agents["w1"].spent = 0.1
    assert asyncio.run(runtime.run_agent(hub, session, "w1", "x")) == "failed: w1's pool is spent" and calls == []


# ---- modes ----

def listed(hub):
    return {t.name for t in asyncio.run(hub.mcp.list_tools())}


def test_hiring_can_be_switched_off(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, allow_hiring=False), ollama=ollama)  # type: ignore[arg-type]
    assert not {"hire_agent", "fire_agent"} & listed(hub)


def test_local_or_cloud_agents_can_be_switched_off(tmp_path, ollama):
    peers = (A2APeer("p1", "http://127.0.0.1:9/", price_usd_per_message=0.01),)
    no_local = Hub(make_settings(tmp_path, allow_local_agents=False, a2a_peers=peers), ollama=ollama)  # type: ignore[arg-type]
    no_cloud = Hub(make_settings(tmp_path, allow_cloud_agents=False, a2a_peers=peers, port=9101), ollama=ollama)  # type: ignore[arg-type]
    assert no_local.hireable() == ["gpt-4.1-mini", "a2a/p1"]
    assert no_cloud.hireable() == ["ollama_chat/tiny"]
    session = no_cloud.new_session("t", trace=False)
    assert "disabled" in hire(no_cloud, session)["error"]


# ---- remote A2A peers ----

def test_a2a_peers_are_parsed_and_validated():
    s = load_settings(['+a2a_peers.helper={url: "https://example.com/a/", price_usd_per_message: 0.02, '
                       'headers: {Authorization: Bearer t}}'])
    assert s.a2a_peers == (A2APeer("helper", "https://example.com/a/", (("Authorization", "Bearer t"),), 0.02),)
    for bad in ['+a2a_peers.x={url: "ftp://a"}', '+a2a_peers.x={url: "https://a", price_usd_per_message: -1}',
                '+a2a_peers.x={url: "https://a", shell: 1}', '+a2a_peers.Bad={url: "https://a"}']:
        with pytest.raises(ValueError):
            load_settings([bad])


def peer_hub(tmp_path, port, **peer_kw):
    peers = (A2APeer("p1", f"http://127.0.0.1:{port}/", **peer_kw),)
    return Hub(make_settings(tmp_path, port=port + 1, a2a_peers=peers), ollama=FakeOllama())  # type: ignore[arg-type]


def test_hiring_and_asking_a_priced_peer(tmp_path):
    hub = peer_hub(tmp_path, 9180, headers=(("Authorization", "Bearer t0k"),))

    async def body():
        async with (peer(9180, price=Price(per_message_usd=0.003), report=0.0021, token="t0k") as (_, echo),
                    hub.serving()):
            session = hub.new_session("t", trace=False)
            hired = await hub.tools.hire_agent(ctx(session, "ceo"), "remote", "worker", "a2a/p1",  # type: ignore[arg-type]
                                               "Answer briefly.", 0.01)
            answer = await runtime.ask_agent(hub, session, "ceo", "remote", "hello")
            return session, hired, answer, echo.seen
    session, hired, answer, seen = asyncio.run(body())
    rec = session.ledger.agents["remote"]
    assert hired["tier"] == "remote" and hired["price_usd_per_message"] == 0.003
    assert answer == "echo: Answer briefly.\n\nhello"  # the hire prompt goes with each message
    assert rec.spent == 0.003 and hub.wallet.spent == 0.003  # the larger of declared (0.003) and reported (0.0021)
    assert DEADLINE_KEY in seen[0][1]  # the request's deadline travels to the peer
    llm = [e for e in session.trace.events if e["kind"] == "llm" and e["agent"] == "remote"]
    assert llm[0]["cost_usd"] == 0.003 and llm[0]["model"] == "a2a/p1"


def test_a_peer_reporting_more_than_it_declared_is_charged_what_it_reported(tmp_path):
    hub = peer_hub(tmp_path, 9182)

    async def body():
        async with peer(9182, price=Price(per_message_usd=0.001), report=0.004), hub.serving():
            session = hub.new_session("t", trace=False)
            await hub.tools.hire_agent(ctx(session, "ceo"), "remote", "worker", "a2a/p1", "", 0.01)  # type: ignore[arg-type]
            await runtime.ask_agent(hub, session, "ceo", "remote", "hi")
            return session.ledger.agents["remote"].spent
    assert asyncio.run(body()) == 0.004


def test_unpriced_unreachable_or_misused_peers_are_refused(tmp_path):
    hub = peer_hub(tmp_path, 9184)
    priced = peer_hub(tmp_path / "b", 9184, price_usd_per_message=0.02)
    wrong_token = peer_hub(tmp_path / "c", 9184, price_usd_per_message=0.02, headers=(("Authorization", "Bearer x"),))

    async def body():
        out = {}
        async with peer(9184):  # declares no price
            for label, h in (("unpriced", hub), ("priced", priced)):
                session = h.new_session("t", trace=False)
                out[label] = await h.tools.hire_agent(ctx(session, "ceo"), "r", "worker", "a2a/p1", "", 0.05)  # type: ignore[arg-type]
            s = priced.new_session("t", trace=False)
            out["manager"] = await priced.tools.hire_agent(ctx(s, "ceo"), "r", "manager", "a2a/p1", "", 0.05)  # type: ignore[arg-type]
            out["too_small"] = await priced.tools.hire_agent(ctx(s, "ceo"), "r2", "worker", "a2a/p1", "", 0.01)  # type: ignore[arg-type]
        async with peer(9184, price=Price(0.001), token="right"):
            s = wrong_token.new_session("t", trace=False)
            out["token"] = await wrong_token.tools.hire_agent(ctx(s, "ceo"), "r", "worker", "a2a/p1", "", 0.05)  # type: ignore[arg-type]
        return out
    out = asyncio.run(body())
    assert "declares no price" in out["unpriced"]["error"]
    assert out["priced"]["price_usd_per_message"] == 0.02
    assert "only be hired as a worker" in out["manager"]["error"]
    assert "cover one message" in out["too_small"]["error"]
    assert "could not reach" in out["token"]["error"]


def test_a_peer_stops_when_its_pool_cannot_pay(tmp_path):
    hub = peer_hub(tmp_path, 9186)

    async def body():
        async with peer(9186, price=Price(per_message_usd=0.004)) as (_, echo), hub.serving():
            session = hub.new_session("t", trace=False)
            await hub.tools.hire_agent(ctx(session, "ceo"), "r", "worker", "a2a/p1", "", 0.006)  # type: ignore[arg-type]
            first = await runtime.ask_agent(hub, session, "ceo", "r", "one")
            second = await runtime.ask_agent(hub, session, "ceo", "r", "two")
            return first, second, len(echo.seen)
    first, second, sent = asyncio.run(body())
    assert first == "echo: one" and "cannot pay" in second and sent == 1


# ---- the cost extension on our own agents ----

def test_our_agents_publish_prices_and_report_costs(tmp_path):
    class LLM:
        async def chat(self, model, messages, tools=None):
            return reply("done"), 0.002
    hub = Hub(make_settings(tmp_path, port=9188), llm=LLM(), ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            headers = {"Authorization": f"Bearer {session.tokens[USER]}"}
            url = hub.agent_url(session, "ceo")
            card = await fetch_card(url, headers, 10)
            answer, metadata, _ = await send(url, headers, "hi", session.id, 30)
            return card, answer, metadata
    card, answer, metadata = asyncio.run(body())
    assert any(e.uri == COST_EXTENSION for e in card.capabilities.extensions)
    assert price_of(card).per_input_token_usd > 0  # gpt-4.1-mini's LiteLLM price
    assert answer == "done" and metadata[COST_KEY] == 0.002


# ---- deadlines and cancellation ----

class SlowWorkerLLM:
    """The CEO hires w1 and asks it; w1 takes `delay` seconds. Records whether w1's call was cancelled."""

    def __init__(self, delay, fire=False):
        self.delay, self.fire, self.cancelled = delay, fire, False

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == "slow":
            try:
                await asyncio.sleep(self.delay)
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            return reply("late"), 0.0
        turn = sum(1 for m in messages if m["role"] == "tool")
        if turn == 0:
            return reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini",
                                                    "prompt": "slow", "budget_usd": 0.1}, "c1")]), 0.0
        if turn == 1:
            asks = [call("ask_agent", {"name": "w1", "text": "go"}, "c2")]
            fires = [call("fire_agent", {"name": "w1"}, "c3")] if self.fire else []
            return reply(calls=asks + fires), 0.0
        return reply(f"w1 said: {json.loads(messages[-1 - int(self.fire)]['content'])['reply']}"), 0.0


async def ceo_run(hub, **kw):
    async with hub.serving():
        session = hub.new_session("t", trace=False)
        began = time.monotonic()
        answer = await runtime.ask_agent(hub, session, USER, "ceo", "start")
        took = time.monotonic() - began
        await asyncio.sleep(0.2)
        return answer, took, dict(hub._running)


def test_every_agent_stops_at_the_requests_deadline(tmp_path):
    llm = SlowWorkerLLM(delay=60)
    hub = Hub(make_settings(tmp_path, port=9189, task_timeout=3), llm=llm, ollama=FakeOllama())  # type: ignore[arg-type]
    answer, took, running = asyncio.run(ceo_run(hub))
    assert answer == "failed: out of time" and took < 6
    assert llm.cancelled and running == {}  # w1's run ended with the deadline; nothing is left running


def test_firing_an_agent_cancels_its_running_work(tmp_path):
    llm = SlowWorkerLLM(delay=60)
    hub = Hub(make_settings(tmp_path, port=9190), llm=llm, ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            began = time.monotonic()
            asking = asyncio.create_task(runtime.ask_agent(hub, session, USER, "ceo", "start"))
            while not hub._running.get((session.id, "w1")):  # w1 is busy with its 60 s call
                await asyncio.sleep(0.05)
            fired = await hub.tools.fire_agent(ctx(session, "ceo"), "w1")  # type: ignore[arg-type]
            answer = await asyncio.wait_for(asking, 10)
            await asyncio.sleep(0.2)
            return fired, answer, time.monotonic() - began, dict(hub._running)
    fired, answer, took, running = asyncio.run(body())
    assert fired == {"fired": ["w1"]} and took < 10 and llm.cancelled and running == {}
    assert answer == "w1 said: failed: stopped (fired, or the session closed)"


def test_closing_a_session_cancels_its_runs(tmp_path):
    llm = SlowWorkerLLM(delay=60)
    hub = Hub(make_settings(tmp_path, port=9191), llm=llm, ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            asking = asyncio.create_task(runtime.ask_agent(hub, session, USER, "ceo", "start"))
            while not hub._running.get((session.id, "w1")):
                await asyncio.sleep(0.05)
            await hub.close_session(session)
            answer = await asyncio.wait_for(asking, 10)
            await asyncio.sleep(0.2)
            return answer, dict(hub._running)
    answer, running = asyncio.run(body())
    assert running == {} and llm.cancelled and "stopped" in answer


# ---- A2A task mode: replies finish their task, so nothing is left running ----

def test_requests_leave_no_a2a_worker_tasks_behind(tmp_path):
    class LLM:
        async def chat(self, model, messages, tools=None):
            return reply("ok"), 0.0
    hub = Hub(make_settings(tmp_path, port=9176), llm=LLM(), ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            answers = [await runtime.ask_agent(hub, session, USER, "ceo", f"q{i}") for i in range(3)]
            await asyncio.sleep(0.3)
            leftover = [t for t in asyncio.all_tasks() if "ActiveTask" in t.get_coro().__qualname__]
            return answers, leftover
    answers, leftover = asyncio.run(body())
    assert answers == ["ok", "ok", "ok"] and leftover == []


def test_replies_are_read_from_messages_tasks_and_artifacts():
    from a2a.helpers import new_text_artifact, new_text_message
    from a2a.types import Role, StreamResponse, Task, TaskState, TaskStatus

    from hashiru.peers import reply_of
    message = new_text_message("as message", role=Role.ROLE_AGENT)
    message.metadata.update({COST_KEY: 0.1})
    status = TaskStatus(state=TaskState.TASK_STATE_COMPLETED, message=new_text_message("as task", role=Role.ROLE_AGENT))
    artifacts_only = Task(id="t", status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                          artifacts=[new_text_artifact("answer", "in an artifact")])
    done = TaskState.TASK_STATE_COMPLETED
    assert reply_of(StreamResponse(message=message)) == ("as message", {COST_KEY: 0.1}, done)
    assert reply_of(StreamResponse(task=Task(id="t", status=status))) == ("as task", {}, done)
    assert reply_of(StreamResponse(task=artifacts_only)) == ("in an artifact", {}, done)


def test_concurrent_hires_cannot_spend_the_same_money(hub, session, ollama):
    async def both():
        c = ctx(session, "ceo")
        args = {"role": "worker", "model": "ollama_chat/tiny", "prompt": "p", "budget_usd": 0.3}
        return await asyncio.gather(hub.tools.hire_agent(c, name="a", **args),  # type: ignore[arg-type]
                                    hub.tools.hire_agent(c, name="b", **args))  # type: ignore[arg-type]
    hub.ollama = FakeOllama(delay=0.05)
    results = asyncio.run(both())
    assert sorted("hired" in r for r in results) == [False, True]  # 0.30 + 0.30 > the CEO's 0.50
    assert session.ledger.left()["ceo"] >= 0
