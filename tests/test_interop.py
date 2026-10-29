"""Phase 5 end to end on a test port with a scripted LLM (no tokens spent): the official OpenAI client against /v1,
an A2A client against the public CEO, an MCP client against /public/mcp, and the chat page."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import httpx2
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState
from conftest import FakeOllama, make_settings
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI

from hashiru.hub import Hub
from hashiru.peers import COST_EXTENSION, COST_KEY, fetch_card

PORT = 9196


def call(name, args, cid):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


class TeamLLM:
    """For a message with "team" in it, the CEO hires w1, asks it, delivers a note and answers; anything else it echoes.
    w1 says hello. While `gate` is set and closed, the CEO waits."""

    def __init__(self) -> None:
        self.gate: asyncio.Event | None = None
        self.ceo_prompts: list[str] = []

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == "Greet people.":
            return reply("hello from w1"), 0.001
        if self.gate is not None:
            await self.gate.wait()
        prompt = messages[1]["content"]
        self.ceo_prompts.append(prompt)
        if "team" not in prompt.split("The user's new message:")[-1]:
            return reply(f"echo: {prompt.splitlines()[-1]}"), 0.002
        script = [reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini",
                                                   "prompt": "Greet people.", "budget_usd": 0.1}, "c1")]),
                  reply(calls=[call("ask_agent", {"name": "w1", "text": "say hello"}, "c2")]),
                  reply(calls=[call("deliver_to_user", {"text": "a note for you"}, "c3")]),
                  reply("w1 says hello")]
        return script[sum(1 for m in messages if m["role"] == "tool")], 0.002


def serve(tmp_path, body, llm=None, **overrides):
    """Run `body(hub, base_url)` against a served hub; returns what it returns."""
    hub = Hub(make_settings(tmp_path, port=PORT, **overrides), llm=llm or TeamLLM(), ollama=FakeOllama())  # type: ignore[arg-type]

    async def go():
        async with hub.serving():
            return await body(hub, hub.settings.internal_url)
    return asyncio.run(go())


def user(text):
    return {"role": "user", "content": text}


def test_the_openai_client_chats_streams_and_continues_a_conversation(tmp_path):
    llm = TeamLLM()

    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused")
        models = [m.id async for m in client.models.list()]
        first = await client.chat.completions.create(model="hashiru", messages=[user("build a team")])
        content = first.choices[0].message.content
        chat = [user("build a team"), {"role": "assistant", "content": content}, user("thanks")]
        stream = await client.chat.completions.create(model="hashiru", messages=chat, stream=True,
                                                      stream_options={"include_usage": True})
        chunks = [c async for c in stream]
        return models, first, chunks, len(hub.conversations.open_)

    models, first, chunks, open_count = serve(tmp_path, body, llm)
    assert models == ["hashiru"]
    message = first.choices[0].message
    assert message.content == "**ceo:** a note for you\n\nw1 says hello"  # delivered text first, then the reply
    reasoning = message.model_extra["reasoning_content"]
    assert "ceo hired w1 as a worker on gpt-4.1-mini with $0.10" in reasoning
    assert "w1 → ceo: hello from w1" in reasoning
    session = first.model_extra["hashiru"]["session"]
    assert {c.model_extra["hashiru"]["session"] for c in chunks if c.choices} == {session}  # continued, not new
    assert open_count == 1
    assert "w1 (worker, cloud gpt-4.1-mini)" in llm.ceo_prompts[-1]  # its hires stayed on between turns
    streamed = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert streamed == "echo: The user's new message: thanks"
    assert chunks[-1].choices == [] and chunks[-1].usage is not None
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "stop"


def test_a_stream_carries_activity_as_reasoning_and_delivered_text_as_content(tmp_path):
    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused")
        stream = await client.chat.completions.create(model="hashiru", messages=[user("team")], stream=True)
        return [c async for c in stream]

    chunks = serve(tmp_path, body)
    deltas = [c.choices[0].delta for c in chunks]
    content = "".join(d.content or "" for d in deltas)
    reasoning = [d.model_extra.get("reasoning_content") for d in deltas if d.model_extra.get("reasoning_content")]
    assert content == "**ceo:** a note for you\n\nw1 says hello"
    assert reasoning[0].startswith("ceo hired w1")
    assert reasoning[-1] == "w1 → ceo: hello from w1\n"  # the delivered note arrives as content instead
    events = [c.model_extra["hashiru"]["event"] for c in chunks if "event" in c.model_extra["hashiru"]]
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
    assert all("args" not in e and "result" not in e for e in events)
    assert chunks[-1].model_extra["hashiru"]["cost_usd"] > 0


def test_an_edited_chat_gets_a_new_conversation_that_sees_the_chat_so_far(tmp_path):
    llm = TeamLLM()

    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused")
        first = await client.chat.completions.create(model="hashiru", messages=[user("hi")])
        said = first.choices[0].message.content
        folded = await client.chat.completions.create(model="hashiru", messages=[
            user("hi"), {"role": "assistant", "content": f"<details type=\"reasoning\">x</details>\n{said}"},
            user("again")])
        edited = await client.chat.completions.create(model="hashiru", messages=[
            user("hi"), {"role": "assistant", "content": "something else"}, user("again")])
        return [r.model_extra["hashiru"]["session"] for r in (first, folded, edited)]

    first, folded, edited = serve(tmp_path, body, llm)
    assert folded == first          # a reasoning block folded into the reply by a chat app does not matter
    assert edited != first
    assert "Our conversation so far:\nUser: hi\nYou: something else" in llm.ceo_prompts[-1]


def test_a_busy_conversation_refuses_another_turn_and_a_dropped_stream_stops_the_work(tmp_path):
    llm = TeamLLM()

    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused")
        first = await client.chat.completions.create(model="hashiru", messages=[user("hi")])
        chat = [user("hi"), {"role": "assistant", "content": first.choices[0].message.content}, user("wait")]
        llm.gate = asyncio.Event()  # the CEO now waits forever
        async with httpx.AsyncClient(base_url=base, timeout=10) as http:
            async with http.stream("POST", "/v1/chat/completions", json={"stream": True, "messages": chat}) as resp:
                session = resp.headers["x-hashiru-session"]
                await anext(resp.aiter_lines())
                again = await http.post("/v1/chat/completions", json={"messages": chat})
                busy = hub.conversations.get(session).busy
            for _ in range(100):  # the client hung up: the hub stops the turn and closes the conversation
                if session not in hub.sessions:
                    break
                await asyncio.sleep(0.05)
        return again.status_code, again.json()["error"]["code"], busy, session in hub.sessions, hub.conversations.open_

    status, code, busy, still_open, open_ = serve(tmp_path, body, llm)
    assert (status, code, busy) == (409, "conversation_busy", True)
    assert not still_open and open_ == {}


def test_full_hubs_close_the_oldest_idle_conversation_and_refuse_when_all_are_busy(tmp_path):
    llm = TeamLLM()

    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused", max_retries=0)
        a = await client.chat.completions.create(model="hashiru", messages=[user("one")])
        b = await client.chat.completions.create(model="hashiru", messages=[user("two")])
        llm.gate = asyncio.Event()
        async with (httpx.AsyncClient(base_url=base, timeout=10) as http,
                    http.stream("POST", "/v1/chat/completions", json={"stream": True, "messages": [user("x")]})):
            full = await http.post("/v1/chat/completions", json={"messages": [user("three")]})
        ids = [r.model_extra["hashiru"]["session"] for r in (a, b)]
        return ids, ids[0] in hub.sessions, full.status_code, full.json()["error"]["type"]

    (a, b), a_open, status, kind = serve(tmp_path, body, llm, max_sessions=1)
    assert a != b and not a_open                       # b's arrival closed a, the idle one
    assert (status, kind) == (429, "rate_limit_error")  # the only slot is busy


def test_session_state_lists_agents_with_pools_and_delete_closes_it(tmp_path):
    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused")
        first = await client.chat.completions.create(model="hashiru", messages=[user("team")])
        session = first.model_extra["hashiru"]["session"]
        async with httpx.AsyncClient(base_url=f"{base}/v1/hashiru", timeout=10) as http:
            state = (await http.get(f"/sessions/{session}")).json()
            info = (await http.get("/info")).json()
            closed = (await http.delete(f"/sessions/{session}")).status_code
            gone = (await http.get(f"/sessions/{session}")).status_code
        return state, info, closed, gone, session in hub.sessions

    state, info, closed, gone, still = serve(tmp_path, body)
    agents = {a["name"]: a for a in state["agents"]}
    assert agents["w1"] | {"spent_usd": 0} == {
        "name": "w1", "role": "worker", "tier": "cloud", "model": "gpt-4.1-mini", "parent": "ceo", "depth": 1,
        "live": True, "reserved_usd": 0.1, "spent_usd": 0, "left_usd": 0.099, "gpu_mb": 0}
    assert state["budget_usd"] == 0.5 and state["turns"] == 1
    assert info["ceo_model"] == "gpt-4.1-mini" and info["open_sessions"] == 1
    assert (closed, gone, still) == (200, 404, False)


def test_the_ceo_is_a_public_a2a_agent_that_streams_its_progress(tmp_path):
    async def body(hub, base):
        async with httpx.AsyncClient(timeout=10) as http:
            card = await fetch_card(base, {}, 10)
            client = ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create(card)
            first = [e async for e in client.send_message(SendMessageRequest(
                message=new_text_message("team", role=Role.ROLE_USER)))]
            context = next(e.status_update.context_id for e in first if e.HasField("status_update"))
            second = [e async for e in client.send_message(SendMessageRequest(
                message=new_text_message("thanks", context_id=context, role=Role.ROLE_USER)))]
            await client.close()
        return card, first, second, len(hub.conversations.open_)

    card, first, second, open_count = serve(tmp_path, body)
    assert card.name == "HASHIRU" and card.capabilities.streaming
    assert card.supported_interfaces[0].url == f"http://127.0.0.1:{PORT}/a2a"
    assert [e.uri for e in card.capabilities.extensions] == [COST_EXTENSION]
    assert not card.security_schemes                    # no api_key: nothing to declare
    updates = [e.status_update.status for e in first if e.HasField("status_update")]
    working = [get_message_text(s.message) for s in updates if s.state == TaskState.TASK_STATE_WORKING]
    assert working[0].startswith("ceo hired w1") and "w1 → ceo: hello from w1" in working
    done = updates[-1]
    assert done.state == TaskState.TASK_STATE_COMPLETED
    assert get_message_text(done.message) == "**ceo:** a note for you\n\nw1 says hello"
    assert done.message.metadata[COST_KEY] > 0
    last = [e.status_update.status for e in second if e.HasField("status_update")][-1]
    assert get_message_text(last.message).startswith("echo:") and open_count == 1  # one context, one conversation


def test_hashiru_is_an_mcp_server_with_progress_and_a_session_resource(tmp_path):
    async def body(hub, base):
        progress: list[str] = []

        async def on_progress(value, total, message):
            progress.append(message or "")

        async with (httpx2.AsyncClient(timeout=30) as http,
                    streamable_http_client(f"{base}/public/mcp", http_client=http) as (read, write, *_),
                    ClientSession(read, write) as mcp):
            await mcp.initialize()
            tools = sorted(t.name for t in (await mcp.list_tools()).tools)
            first = (await mcp.call_tool("ask", {"text": "team"}, progress_callback=on_progress)).structured_content
            again = (await mcp.call_tool("ask", {"text": "thanks", "session": first["session"]})).structured_content
            usage = json.loads((await mcp.read_resource(f"hashiru://sessions/{first['session']}")).contents[0].text)
            ended = (await mcp.call_tool("end_session", {"session": first["session"]})).structured_content
            stale = (await mcp.call_tool("ask", {"text": "hi", "session": first["session"]})).structured_content
        return tools, progress, first, again, usage, ended, stale

    tools, progress, first, again, usage, ended, stale = serve(tmp_path, body)
    assert tools == ["ask", "end_session"]
    assert progress[0].startswith("ceo hired w1") and "ceo sent you a message" in progress
    assert first["answer"] == "**ceo:** a note for you\n\nw1 says hello" and first["cost_usd"] > 0
    assert again["session"] == first["session"] and again["answer"].startswith("echo:")
    assert {a["name"] for a in usage["agents"]} == {"ceo", "w1"}
    assert ended == {"closed": first["session"]} and "no open session" in stale["error"]


def test_with_an_api_key_every_public_api_needs_it_but_the_page_and_card_do_not(tmp_path):
    async def body(hub, base):
        async with httpx.AsyncClient(base_url=base, timeout=10) as http:
            out = {path: (await http.request(method, path, json={})).status_code for method, path in [
                ("GET", "/v1/models"), ("POST", "/v1/chat/completions"), ("POST", "/a2a"), ("POST", "/public/mcp"),
                ("GET", "/"), ("GET", "/ui/app.js"), ("GET", "/.well-known/agent-card.json")]}
            ok = (await http.get("/v1/models", headers={"Authorization": "Bearer k3y"})).status_code
            card = await fetch_card(base, {}, 10)
        return out, ok, card

    out, ok, card = serve(tmp_path, body, api_key="k3y", public_url="https://me.example")
    assert out == {"/v1/models": 401, "/v1/chat/completions": 401, "/a2a": 401, "/public/mcp": 401,
                   "/": 200, "/ui/app.js": 200, "/.well-known/agent-card.json": 200}
    assert ok == 200
    assert card.supported_interfaces[0].url == "https://me.example/a2a"
    assert card.security_schemes["bearer"].http_auth_security_scheme.scheme == "Bearer"


def test_the_chat_page_is_self_contained(tmp_path):
    async def body(hub, base):
        async with httpx.AsyncClient(base_url=base, timeout=10) as http:
            page = await http.get("/")
            files = {p: (await http.get(p)) for p in ("/ui/app.js", "/ui/markdown.js", "/ui/app.css", "/ui/icon.svg")}
            escape = await http.get("/ui/../interop/page.py")
        return page, files, escape

    page, files, escape = serve(tmp_path, body, frame_ancestors="https://huggingface.co")
    csp = page.headers["content-security-policy"]
    assert "default-src 'self'" in csp and "frame-ancestors https://huggingface.co" in csp
    assert all(r.status_code == 200 and "content-security-policy" in r.headers for r in files.values())
    assert "http" not in page.text.replace("http://www.w3.org", "")  # nothing loaded from other sites
    assert escape.status_code == 404
