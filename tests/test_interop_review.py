"""Regression tests for the Phase 5 review findings, and for agents that end a turn with an empty reply."""
import asyncio
import json
import time
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from conftest import FakeOllama, make_settings
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI
from test_interop import PORT, TeamLLM, call, reply, serve, user

from hashiru.config import load_settings
from hashiru.hub import Hub
from hashiru.interop import public_card
from hashiru.interop.conversations import Conversation, ShuttingDown, normalize
from hashiru.peers import price_of
from hashiru.runtime import EMPTY_REPLY


def test_normalizing_hostile_replies_takes_linear_time():
    for hostile in ("<think>" * 300_000, "<details " * 230_000, "<think>a</think>" * 130_000 + "<details"):
        started = time.perf_counter()
        normalize(hostile)
        assert time.perf_counter() - started < 1.0  # the old regex took minutes on 2 MB of "<think>"
    assert normalize("a <think>x</think> b <details open>y</details> c <think>open") == "a b c <think>open"
    assert normalize("<detailsx>kept</details>") == "<detailsx>kept</details>"


def test_a_non_streaming_client_that_leaves_stops_its_turn(tmp_path):
    llm = TeamLLM()

    async def body(hub, base):
        llm.gate = asyncio.Event()  # the CEO never answers
        async with httpx.AsyncClient(base_url=base, timeout=0.5) as http:
            with pytest.raises(httpx.ReadTimeout):
                await http.post("/v1/chat/completions", json={"messages": [user("hi")]})
        for _ in range(100):
            if not hub.conversations.open_ and len(hub.sessions) == 0:
                break
            await asyncio.sleep(0.05)
        return dict(hub.conversations.open_), dict(hub.sessions)

    assert serve(tmp_path, body, llm) == ({}, {})


def conversations(hub, n, age=10_000.0):
    out = []
    for _ in range(n):
        conv = Conversation(hub.new_session("t", trace=False), used=time.monotonic() - age)
        hub.conversations.open_[conv.id] = conv
        out.append(conv)
    return out


def slow_close(hub, monkeypatch):
    real = hub.close_session

    async def close_session(session):
        await asyncio.sleep(0.05)
        await real(session)
    monkeypatch.setattr(hub, "close_session", close_session)


def test_a_sweep_does_not_close_a_conversation_that_got_busy_meanwhile(tmp_path, monkeypatch):
    hub = Hub(make_settings(tmp_path, session_ttl=1), ollama=FakeOllama())  # type: ignore[arg-type]
    a, b = conversations(hub, 2)
    slow_close(hub, monkeypatch)

    async def go():
        sweep = asyncio.create_task(hub.conversations.sweep())
        await asyncio.sleep(0.01)  # a is being closed
        turn = hub.conversations.ask(b, "hi")  # a client claims b
        await sweep
        await turn.aclose()
    asyncio.run(go())
    assert a.id not in hub.conversations.open_ and b.id in hub.conversations.open_


def test_concurrent_opens_never_exceed_max_sessions(tmp_path, monkeypatch):
    hub = Hub(make_settings(tmp_path, max_sessions=2), ollama=FakeOllama())  # type: ignore[arg-type]
    conversations(hub, 2, age=0)
    slow_close(hub, monkeypatch)

    async def go():
        await asyncio.gather(*(hub.conversations.open("t") for _ in range(3)))
    asyncio.run(go())
    assert len(hub.conversations.open_) == 2


def test_after_shutdown_begins_no_conversation_opens(hub):
    async def go():
        await hub.conversations.close_all()
        with pytest.raises(ShuttingDown):
            await hub.conversations.open("t")
    asyncio.run(go())


def test_idle_conversations_expire_without_new_ones_opening(tmp_path):
    hub = Hub(make_settings(tmp_path, session_ttl=0.2), ollama=FakeOllama())  # type: ignore[arg-type]
    (old,) = conversations(hub, 1, age=5)

    async def go():
        sweeper = asyncio.create_task(hub.conversations.sweeper())
        await asyncio.sleep(0.4)
        sweeper.cancel()
    asyncio.run(go())
    assert old.id not in hub.conversations.open_


def test_a_conversation_closed_during_its_turn_leaves_no_head(hub):
    (conv,) = conversations(hub, 1, age=0)
    hub.conversations.open_.pop(conv.id)  # closed (DELETE, end_session, sweep) while the turn ran
    hub.conversations._remember(conv, "hi", "hello")
    assert hub.conversations._heads == {} and conv.history == [("hi", "hello")]


def test_without_a_key_only_this_hubs_own_page_may_call(hub):
    host = [(b"host", b"127.0.0.1:9200")]
    assert hub.asgi.check_public([*host, (b"origin", b"http://127.0.0.1:9200")]) is None
    assert hub.asgi.check_public([*host, (b"origin", b"http://localhost:3000")])[0] == 403  # another local app
    assert hub.asgi.check_public([*host, (b"origin", b"http://127.0.0.1:9100")])[0] == 403


def test_a_pasted_key_with_whitespace_still_works(monkeypatch):
    monkeypatch.setenv("HASHIRU_API_KEY", " k3y\n")
    assert load_settings().api_key == "k3y"
    monkeypatch.setenv("HASHIRU_API_KEY", "  ")
    with pytest.raises(ValueError, match="blank"):
        load_settings()


def test_the_public_card_declares_what_one_message_can_cost(hub):
    assert price_of(public_card(hub)).per_message_usd == hub.settings.budget_usd


def test_with_a_key_the_mcp_server_answers_any_host_like_the_rest_of_the_api(tmp_path):
    async def body(hub, base):
        headers = {"Host": "hashiru.example", "Authorization": "Bearer k3y"}
        async with (httpx2.AsyncClient(headers=headers, timeout=10) as http,
                    streamable_http_client(f"{base}/public/mcp", http_client=http) as (read, write, *_),
                    ClientSession(read, write) as mcp):
            return (await mcp.initialize()).server_info.name

    assert serve(tmp_path, body, api_key="k3y") == "hashiru"


def test_a_consumer_that_fails_mid_turn_releases_the_conversation_at_once(tmp_path):
    async def body(hub, base):
        async def report_progress(*_):
            raise RuntimeError("client gone")
        with pytest.raises(RuntimeError):
            await hub.public_mcp.ask(SimpleNamespace(report_progress=report_progress), "team")
        (conv,) = hub.conversations.open_.values()
        return conv.busy, conv.session.trace._listeners

    assert serve(tmp_path, body) == (False, [])


class QuietLLM:
    """The CEO hires w1 and asks it; w1, then the CEO, first reply with nothing at all."""

    def __init__(self) -> None:
        self.seen: dict[str, list[list[dict]]] = {"ceo": [], "w1": []}

    async def chat(self, model, messages, tools=None):
        who = "w1" if messages[0]["content"] == "Greet people." else "ceo"
        self.seen[who].append(messages)
        nudged = messages[-1]["content"] == EMPTY_REPLY
        if who == "w1":
            return reply("hello" if nudged else ""), 0.001
        tools_used = sum(1 for m in messages if m["role"] == "tool")
        if tools_used == 0:
            return reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini",
                                                    "prompt": "Greet people.", "budget_usd": 0.1}, "c1")]), 0.002
        if tools_used == 1:
            return reply(calls=[call("ask_agent", {"name": "w1", "text": "say hello"}, "c2")]), 0.002
        said = json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])["reply"]
        return reply(f"w1 said {said}" if nudged else ""), 0.002


def test_agents_that_reply_with_nothing_are_asked_once_more(tmp_path):
    llm = QuietLLM()
    hub = Hub(make_settings(tmp_path, port=PORT), llm=llm, ollama=FakeOllama())  # type: ignore[arg-type]

    async def go():
        async with hub.serving():
            client = AsyncOpenAI(base_url=f"{hub.settings.internal_url}/v1", api_key="unused")
            return await client.chat.completions.create(model="hashiru", messages=[user("go")])
    answer = asyncio.run(go()).choices[0].message.content
    assert answer == "w1 said hello"
    assert len(llm.seen["w1"]) == 2 and len(llm.seen["ceo"]) == 4  # one extra call each, no more


class SilentLLM:
    """Every agent replies with nothing, every time."""

    async def chat(self, model, messages, tools=None):
        return reply(""), 0.0


def test_an_agent_that_answers_nothing_twice_fails_clearly(tmp_path):
    hub = Hub(make_settings(tmp_path, port=PORT, ceo_model="ollama_chat/tiny"), llm=SilentLLM(),
              ollama=FakeOllama())  # type: ignore[arg-type]

    async def go():
        async with hub.serving():
            client = AsyncOpenAI(base_url=f"{hub.settings.internal_url}/v1", api_key="unused")
            return await client.chat.completions.create(model="hashiru", messages=[user("go")])
    answer = asyncio.run(go()).choices[0].message.content
    assert answer == "failed: ceo gave an empty answer (a local model may have filled its context window: raise " \
                     "ollama_num_ctx)"
