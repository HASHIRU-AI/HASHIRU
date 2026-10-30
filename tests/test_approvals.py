"""Phase 8c: approvals before a big spend. A cloud hire reserving approve_hire_usd or more, a local hire or escalation
loading a model of approve_hire_mb or more, and a turn whose cloud calls cost approve_turn_usd wait for the user;
the chat asks in-band and the user's next message decides (fake LLMs, no network)."""
import asyncio
import time

import httpx
import httpx2
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest
from conftest import ctx, make_settings
from google.protobuf.json_format import MessageToDict
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI
from test_interop import call, reply, serve, user
from test_router import SizedOllama, worker_replies

from hashiru import approvals, cli, runtime
from hashiru import hub as hub_module
from hashiru.approvals import Decision
from hashiru.config import load_settings
from hashiru.hub import Hub
from hashiru.interop.a2a_public import APPROVAL_KEY
from hashiru.interop.activity import describe
from hashiru.interop.mcp_public import MCP_PATH
from hashiru.peers import fetch_card
from hashiru.session import USER
from hashiru.trace import PRIVATE

MODELS = ("gpt-4.1-mini", "gpt-4.1", "ollama_chat/tiny", "ollama_chat/big")


def make_hub(tmp_path, llm=None, **settings):
    return Hub(make_settings(tmp_path, models=MODELS, **settings), llm=llm, ollama=SizedOllama())  # type: ignore[arg-type]


def attended(hub):
    """A session someone answers (as a conversation's is)."""
    session = hub.new_session("t", trace=False)
    session.approvals.attended = True
    session.approvals.new_turn(spent=0.0, deadline=time.time() + 10)
    return session


def hire(hub, session, **kw):
    args = {"name": "w1", "role": "worker", "prompt": "p", "budget_usd": 0.1} | kw
    return hub.tools.hire_agent(ctx(session, "ceo"), **args)  # type: ignore[arg-type]


async def asked(session):
    """The questions waiting for the user, once there are some."""
    for _ in range(300):
        if questions := session.approvals.show():
            return questions
        await asyncio.sleep(0.01)
    raise AssertionError("nobody asked")


def answer(session, questions, text):
    session.approvals.decide([q.id for q in questions], Decision.of(text, []))


# ---- hires ----

def test_a_big_cloud_hire_waits_for_approval_and_a_small_one_does_not(tmp_path):
    hub = make_hub(tmp_path, approve_hire_usd=0.2)
    session = attended(hub)

    async def go():
        assert "hired" in await hire(hub, session, name="small", model="gpt-4.1", budget_usd=0.1)
        assert session.approvals.show() == []
        big = asyncio.create_task(hire(hub, session, name="big", model="gpt-4.1", budget_usd=0.3))
        questions = await asked(session)
        assert session.ledger.get("big") is None                   # nothing reserved while it waits
        answer(session, questions, "approve")
        return questions, await big
    questions, out = asyncio.run(go())
    assert [(q.kind, q.agent) for q in questions] == [("api", "ceo")]
    assert questions[0].what == "ceo wants to hire big on gpt-4.1 with $0.30 reserved"
    assert out["hired"] == "big" and session.ledger.agents["big"].reserved == 0.3
    kinds = [(e["action"], e["what"]) for e in session.trace.events if e["kind"] == "approval"]
    assert kinds == [("asked", questions[0].what), ("approved", questions[0].what)]


def test_a_denial_ends_the_hire_and_the_hirer_hears_the_users_words(tmp_path):
    hub = make_hub(tmp_path, approve_hire_usd=0.2)
    session = attended(hub)

    async def go():
        task = asyncio.create_task(hire(hub, session, difficulty="hard", budget_usd=0.3))
        answer(session, await asked(session), "no, use a local model")
        return await task
    out = asyncio.run(go())
    assert out["denied"] is True and "no, use a local model" in out["error"]
    assert session.ledger.get("w1") is None                         # no other model was tried
    assert len([e for e in session.trace.events if e["kind"] == "approval" and e["action"] == "asked"]) == 1
    assert out["_labels"] == []                                     # the user's words carry the user's labels
    private = attended(hub)
    private.trace.receive(USER, private.id, [PRIVATE])

    async def again():
        task = asyncio.create_task(hire(hub, private, model="gpt-4.1", budget_usd=0.3))
        questions = await asked(private)
        private.approvals.decide([q.id for q in questions],
                                 Decision.of("not for my tax return", private.trace.outgoing(USER, private.id, "x")))
        return await task
    assert asyncio.run(again())["_labels"] == [PRIVATE]            # call_tool masks it for a cloud hirer


def test_a_local_hire_asks_only_to_load_a_big_model_that_is_not_loaded(tmp_path, monkeypatch):
    hub = make_hub(tmp_path, approve_hire_mb=400)
    session = attended(hub)

    async def go():
        task = asyncio.create_task(hire(hub, session, name="b1", model="ollama_chat/big"))  # 500 MB
        questions = await asked(session)
        assert "hired" in await hire(hub, session, name="t1", model="ollama_chat/tiny")  # 50 MB, while b1 waits
        assert hub.ollama.loaded == ["ollama_chat/tiny"]           # big is not loaded while it waits
        answer(session, questions, "yes")
        assert "hired" in await task
        assert "hired" in await hire(hub, session, name="b2", model="ollama_chat/big")  # loaded: no new memory
        return questions
    questions = asyncio.run(go())
    assert [(q.kind, q.what) for q in questions] == [("gpu", "ceo wants to hire b1 on ollama_chat/big, loading "
                                                             "500 MB of memory")]
    assert session.approvals.show() == []
    for module in (approvals, hub_module):
        monkeypatch.setattr(module, "free_mb", lambda: 100)
    small = make_hub(tmp_path / "small", approve_hire_mb=400)
    tight = attended(small)
    assert "needs 500 MB" in asyncio.run(hire(small, tight, model="ollama_chat/big"))["error"]
    assert tight.approvals.show() == []                             # it could not load: nobody was asked


def test_nobody_to_ask_denies_and_an_unanswered_question_expires_at_the_deadline(tmp_path):
    hub = make_hub(tmp_path, approve_hire_usd=0.2)
    session = hub.new_session("t", trace=False)                     # e.g. `hashiru ask` without a terminal
    out = asyncio.run(hire(hub, session, model="gpt-4.1", budget_usd=0.3))
    assert out["denied"] is True and "nobody can approve" in out["error"]
    late = attended(hub)
    late.approvals.new_turn(spent=0.0, deadline=time.time() + 0.2)
    out = asyncio.run(hire(hub, late, model="gpt-4.1", budget_usd=0.3))
    assert out["denied"] is True and "in time" in out["error"]
    assert [e["action"] for e in late.trace.events if e["kind"] == "approval"] == ["asked", "expired"]
    closing = attended(hub)

    async def close_while_waiting():
        task = asyncio.create_task(hire(hub, closing, model="gpt-4.1", budget_usd=0.3))
        await asked(closing)
        started = time.monotonic()
        await hub.close_session(closing)
        out = await task
        return out, time.monotonic() - started
    out, waited = asyncio.run(close_while_waiting())
    assert out["denied"] is True and waited < 2                    # denied at once, not at the deadline


def test_an_escalation_to_a_big_local_model_asks_and_a_denial_skips_it(tmp_path, monkeypatch):
    hub = make_hub(tmp_path, approve_hire_mb=400)
    session = attended(hub)
    asyncio.run(hire(hub, session, difficulty="easy"))              # tiny, then big, then the cloud
    replies = worker_replies(monkeypatch, session, {"ollama_chat/tiny": "failed: no", "ollama_chat/big": "big",
                                                    "gpt-4.1-mini": "mini"})

    async def go():
        task = asyncio.create_task(runtime.ask_agent(hub, session, "ceo", "w1", "solve it"))
        questions = await asked(session)
        answer(session, questions, "deny")
        return questions, await task
    questions, out = asyncio.run(go())
    assert questions[0].what == "w1 failed on ollama_chat/tiny; move it to ollama_chat/big, loading 500 MB of memory"
    assert out == "mini" and replies == ["ollama_chat/tiny", "gpt-4.1-mini"]
    assert "ollama_chat/big" not in hub.ollama.loaded


# ---- turns ----

class Priced:
    """Every call answers "ok"; a cloud call costs `cost` dollars (and counts in `calls`), a local one nothing."""

    def __init__(self, cost):
        self.cost, self.calls = cost, 0

    async def chat(self, model, messages, tools=None):
        if model.startswith("ollama_chat/"):
            return reply("ok"), 0.0
        self.calls += 1
        return reply("ok"), self.cost


def test_a_turn_asks_before_spending_past_each_threshold_and_a_denial_stops_its_cloud_calls(tmp_path, monkeypatch):
    llm = Priced(0.006)
    hub = make_hub(tmp_path, llm=llm, approve_turn_usd=0.005)
    session = attended(hub)
    asyncio.run(hire(hub, session, model="gpt-4.1-mini"))
    asyncio.run(hire(hub, session, name="l1", model="ollama_chat/tiny"))

    async def go():
        assert await runtime.run_agent(hub, session, "w1", "one") == "ok"          # $0.006: under at the start
        second = asyncio.create_task(runtime.run_agent(hub, session, "w1", "two"))
        first_question = await asked(session)
        assert llm.calls == 1
        answer(session, first_question, "approve")                                    # up to $0.010
        assert await second == "ok"
        third = asyncio.create_task(runtime.run_agent(hub, session, "w1", "three"))  # $0.012 spent
        answer(session, await asked(session), "deny")
        return first_question, await third, await runtime.run_agent(hub, session, "w1", "four")
    first_question, third, fourth = asyncio.run(go())
    assert first_question[0].what == "this turn has spent $0.0060 on cloud models; continue, up to $0.0100"
    assert third.startswith("failed:") and fourth.startswith("failed:") and llm.calls == 2
    assert len([e for e in session.trace.events if e["kind"] == "approval" and e["action"] == "asked"]) == 2
    assert asyncio.run(runtime.run_agent(hub, session, "l1", "local")) == "ok"      # local calls cost nothing
    monkeypatch.setattr(runtime, "a2a_send", lambda *args: asyncio.sleep(0, result="answered"))
    assert asyncio.run(runtime.ask_agent(hub, session, USER, "ceo", "next")) == "answered"
    assert not session.approvals.turn_denied and session.approvals.turn_base == pytest.approx(0.012)
    assert asyncio.run(runtime.run_agent(hub, session, "w1", "five")) == "ok"         # the next turn starts afresh


# ---- in the chat ----

class HireLLM:
    """The CEO hires w1 with $0.30, asks it, and answers with what w1 said (or with the hire's error)."""

    def __init__(self):
        self.ceo_prompts, self.tool_results = [], []

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == "Greet people.":
            return reply("hello from w1"), 0.001
        self.ceo_prompts.append(messages[1]["content"])
        results = [m["content"] for m in messages if m["role"] == "tool"]
        self.tool_results = results
        if not results:
            return reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini",
                                                    "prompt": "Greet people.", "budget_usd": 0.3}, "c1")]), 0.002
        if "error" in results[0]:
            return reply("could not hire"), 0.002
        if len(results) == 1:
            return reply(calls=[call("ask_agent", {"name": "w1", "text": "say hello"}, "c2")]), 0.002
        return reply("w1 says hello"), 0.002


def test_the_chat_asks_in_band_and_the_next_message_approves(tmp_path):
    llm = HireLLM()

    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused")
        first = await client.chat.completions.create(model="hashiru", messages=[user("build a team")])
        question = first.choices[0].message.content
        conv = hub.conversations.get(first.model_extra["hashiru"]["session"])
        waiting = hub.conversations.state(conv)
        chat = [user("build a team"), {"role": "assistant", "content": question}, user("approve")]
        stream = await client.chat.completions.create(model="hashiru", messages=chat, stream=True)
        chunks = [c async for c in stream]
        return first, waiting, chunks, conv.history

    first, waiting, chunks, history = serve(tmp_path, body, llm, approve_hire_usd=0.2)
    question = first.choices[0].message.content
    assert question.startswith("Waiting for your approval:\n- ceo wants to hire w1 on gpt-4.1-mini with $0.30 "
                               "reserved")
    assert [a["kind"] for a in first.model_extra["hashiru"]["approval"]] == ["api"]
    assert "waiting for your approval: ceo wants to hire w1" in first.choices[0].message.model_extra[
        "reasoning_content"]
    assert waiting["waiting"] is True and waiting["busy"] is False
    assert [a["name"] for a in waiting["agents"]] == ["ceo"]           # w1 waits for the approval
    streamed = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert streamed == "w1 says hello"
    costs = [c.model_extra["hashiru"]["cost_usd"] for c in chunks if "cost_usd" in c.model_extra.get("hashiru", {})]
    assert first.model_extra["hashiru"]["cost_usd"] == 0.002 and costs == [0.005]  # each answer bills its own part
    assert len(set(llm.ceo_prompts)) == 1                               # the CEO's run went on; it was not asked again
    assert [said for said, _ in history] == ["build a team", "approve"]


def test_a_denial_in_the_chat_reaches_the_hirer_and_closing_a_waiting_chat_stops_its_work(tmp_path):
    llm = HireLLM()

    async def body(hub, base):
        convs = hub.conversations
        conv = await convs.open("chat")
        first = [e async for e in convs.ask(conv, "build a team")]
        second = [e async for e in convs.ask(conv, "too dear, use a local model", private=True)]
        told = list(llm.tool_results), sorted(conv.session.trace.labels("ceo", conv.id))
        other = await convs.open("chat")
        [e async for e in convs.ask(other, "build a team")]
        run = other.waiting.run
        await convs.close(other)
        await asyncio.sleep(0.05)
        return first[-1], second[-1], told, run, other

    first, second, (told, ceo_labels), run, other = serve(tmp_path, body, llm, approve_hire_usd=0.2)
    assert first["approval"] and second["reply"] == "could not hire"
    assert "too dear, use a local model" in told[0]
    assert ceo_labels == ["obfuscated"]          # sent as private, so the cloud CEO got the words obfuscated (R3)
    assert run.cancelled() and other.waiting is None


def test_mcp_and_a2a_clients_get_the_question_and_answer_in_their_next_message(tmp_path):
    async def body(hub, base):
        async with (httpx2.AsyncClient(timeout=30) as http,
                    streamable_http_client(f"{base}{MCP_PATH}", http_client=http) as (read, write, *_),
                    ClientSession(read, write) as mcp):
            await mcp.initialize()
            first = (await mcp.call_tool("ask", {"text": "build a team"})).structured_content
            second = (await mcp.call_tool("ask", {"text": "approve", "session": first["session"]})).structured_content
        async with httpx.AsyncClient(timeout=10) as http:
            card = await fetch_card(base, {}, 10)
            client = ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create(card)
            asks = [e async for e in client.send_message(SendMessageRequest(
                message=new_text_message("build a team", role=Role.ROLE_USER)))]
            context = next(e.status_update.context_id for e in asks if e.HasField("status_update"))
            then = [e async for e in client.send_message(SendMessageRequest(
                message=new_text_message("approve", context_id=context, role=Role.ROLE_USER)))]
            await client.close()
        return first, second, [last_status(asks), last_status(then)]

    first, second, a2a = serve(tmp_path, body, HireLLM(), approve_hire_usd=0.2)
    assert first["answer"].startswith("Waiting for your approval:") and first["approval"][0]["kind"] == "api"
    assert second["answer"] == "w1 says hello" and "approval" not in second
    asks, then = a2a
    assert get_message_text(asks.message).startswith("Waiting for your approval:")
    assert MessageToDict(asks.message.metadata)[APPROVAL_KEY][0]["kind"] == "api"
    assert get_message_text(then.message) == "w1 says hello" and APPROVAL_KEY not in then.message.metadata


def last_status(events):
    return [e.status_update.status for e in events if e.HasField("status_update")][-1]


# ---- the terminal, the activity and the settings ----

def test_the_terminal_asks_on_stdin(tmp_path, monkeypatch, capsys):
    hub = make_hub(tmp_path, approve_hire_usd=0.2)
    session = hub.new_session("t", trace=False)
    session.approvals.answer = cli.terminal(session)
    monkeypatch.setattr("builtins.input", lambda prompt="": "approve")
    assert "hired" in asyncio.run(hire(hub, session, model="gpt-4.1", budget_usd=0.3))
    assert "ceo wants to hire w1 on gpt-4.1 with $0.30 reserved" in capsys.readouterr().out

    def eof(prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    again = hub.new_session("t", trace=False)
    again.approvals.answer = cli.terminal(again)
    assert asyncio.run(hire(hub, again, model="gpt-4.1", budget_usd=0.3))["denied"] is True


def test_approval_events_have_activity_lines():
    what = "ceo wants to hire w1 on gpt-4.1 with $0.30 reserved"
    lines = {action: describe({"kind": "approval", "action": action, "what": what})
             for action in ("asked", "approved", "denied", "expired", "unattended")}
    assert lines == {"asked": f"waiting for your approval: {what}", "approved": f"you approved: {what}",
                     "denied": f"you did not approve: {what}", "expired": f"not approved in time: {what}",
                     "unattended": f"nobody can approve here, so not: {what}"}


def test_approval_settings():
    s = load_settings()
    assert (s.approve_hire_usd, s.approve_hire_mb, s.approve_turn_usd) == (0.25, 16384, 0.25)
    off = load_settings(["approve_hire_usd=0", "approve_hire_mb=0", "approve_turn_usd=0"])
    assert (off.approve_hire_usd, off.approve_hire_mb, off.approve_turn_usd) == (0, 0, 0)
    for key in ("approve_hire_usd", "approve_hire_mb", "approve_turn_usd"):
        with pytest.raises(ValueError, match=key):
            load_settings([f"{key}=-1"])


def test_yes_and_no():
    assert Decision.of("Approve.", []).approved and Decision.of(" yes ", []).approved
    assert Decision.of("ok", []).said == "" and Decision.of("deny", []).said == ""
    no = Decision.of("no, too dear", ["private"])
    assert not no.approved and no.said == "no, too dear" and no.labels == ("private",)
