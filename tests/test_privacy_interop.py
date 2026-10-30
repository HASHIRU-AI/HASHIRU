"""Phase 8a at the front door: a chat marked private (the OpenAI-style header, A2A metadata, the MCP argument, or
private_by_default) runs on a local CEO from that turn on (scripted LLM, no tokens spent)."""
import asyncio

import httpx
import httpx2
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import new_text_message
from a2a.types import Role, SendMessageRequest
from conftest import FakeOllama, make_settings
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from test_interop import serve, user

from hashiru import cli, runtime
from hashiru.hub import Hub
from hashiru.interop import PRIVATE_HEADER, PRIVATE_KEY, SESSION_HEADER
from hashiru.peers import fetch_card


class ModelLLM:
    """Answers "ok" and records the model each call ran on."""

    def __init__(self) -> None:
        self.models: list[str] = []

    async def chat(self, model, messages, tools=None):
        self.models.append(model)
        return type("Reply", (), {"content": "ok", "tool_calls": None})(), 0.0


def chat(*said):
    """An OpenAI-style message list: the user's turns, each answered "ok" except the last."""
    out = []
    for text in said[:-1]:
        out += [user(text), {"role": "assistant", "content": "ok"}]
    return [*out, user(said[-1])]


def test_the_private_header_moves_the_chat_to_a_local_ceo_for_good(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            async def send(messages, **headers):
                r = await http.post("/chat/completions", json={"model": "hashiru", "messages": messages},
                                    headers=headers)
                assert r.status_code == 200, r.text
                return r.headers[SESSION_HEADER]
            cid = await send(chat("hi"))
            await send(chat("hi", "now a secret"), **{SESSION_HEADER: cid, PRIVATE_HEADER: "1"})
            await send(chat("hi", "now a secret", "and more"), **{SESSION_HEADER: cid})  # no header: still private
            state = (await http.get(f"/hashiru/sessions/{cid}")).json()
        return state

    state = serve(tmp_path, body, llm, private_mode="block")
    assert llm.models == ["gpt-4.1-mini", "ollama_chat/tiny", "ollama_chat/tiny"]
    assert state["private"] is True
    assert next(a for a in state["agents"] if a["name"] == "ceo")["tier"] == "local"


def test_private_by_default_makes_every_chat_private(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            r = await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("hi")})
            return r.status_code

    assert serve(tmp_path, body, llm, private_mode="block", private_by_default=True) == 200
    assert llm.models == ["ollama_chat/tiny"]


def test_a_private_chat_without_a_local_model_is_refused_before_any_model_sees_it(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            r = await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("a secret")},
                                headers={PRIVATE_HEADER: "true"})
            return r.status_code, r.json()

    status, reply = serve(tmp_path, body, llm, private_mode="block", models=("gpt-4.1-mini",))
    assert status == 200 and "no local model" in reply["choices"][0]["message"]["content"]
    assert llm.models == []


def test_a2a_and_mcp_clients_can_mark_a_message_private(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(timeout=10) as http:
            client = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(
                await fetch_card(base, {}, 10))
            message = new_text_message("a secret", role=Role.ROLE_USER)
            message.metadata.update({PRIVATE_KEY: True})
            [_ async for _ in client.send_message(SendMessageRequest(message=message))]
            await client.close()
        async with (httpx2.AsyncClient(timeout=30) as http,
                    streamable_http_client(f"{base}/public/mcp", http_client=http) as (read, write, *_),
                    ClientSession(read, write) as mcp):
            await mcp.initialize()
            await mcp.call_tool("ask", {"text": "another secret", "private": True})
            await mcp.call_tool("ask", {"text": "not private"})

    serve(tmp_path, body, llm, private_mode="block")
    assert llm.models == ["ollama_chat/tiny", "ollama_chat/tiny", "gpt-4.1-mini"]


# ---- from the review: a delivery, and conversations that close and reopen ----

def test_a_delivery_of_private_data_keeps_the_next_turn_off_the_cloud(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            r = await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("hi")})
            cid = r.headers[SESSION_HEADER]
            session = hub.conversations.get(cid).session  # a local agent showed the user a private file
            session.ledger.add("m1", "manager", "local", "ollama_chat/tiny", "p", "ceo", 1, 0.0)
            session.trace.receive("m1", session.id, ["private"])
            await runtime.call_tool(session, Delivered(), "m1", "deliver_to_user", {"text": "salary 100k"})
            await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("hi", "and?")},
                            headers={SESSION_HEADER: cid})

    serve(tmp_path, body, llm, private_mode="block")
    assert llm.models == ["gpt-4.1-mini", "ollama_chat/tiny"]


def test_a_private_a2a_context_stays_private_after_its_conversation_closes(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(timeout=10) as http:
            client = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(
                await fetch_card(base, {}, 10))

            async def send(text, private=False):
                message = new_text_message(text, context_id="ctx-1", role=Role.ROLE_USER)
                if private:
                    message.metadata.update({PRIVATE_KEY: True})
                [_ async for _ in client.send_message(SendMessageRequest(message=message))]
            await send("a secret", private=True)
            await hub.conversations.close(hub.conversations.for_context("ctx-1"))  # e.g. it expired
            await send("my bank PIN is 1234")
            await client.close()

    serve(tmp_path, body, llm, private_mode="block")
    assert llm.models == ["ollama_chat/tiny", "ollama_chat/tiny"]


def test_a_private_chat_resent_after_its_conversation_closed_stays_private(tmp_path):
    llm = ModelLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            r = await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("a secret")},
                                headers={PRIVATE_HEADER: "1"})
            await http.delete(f"/hashiru/sessions/{r.headers[SESSION_HEADER]}")  # e.g. evicted
            await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("a secret", "more")})
            await http.post("/chat/completions", json={"model": "hashiru", "messages": chat("unrelated")})

    serve(tmp_path, body, llm, private_mode="block")
    assert llm.models == ["ollama_chat/tiny", "ollama_chat/tiny", "gpt-4.1-mini"]


def test_the_cli_honors_private_by_default(tmp_path, monkeypatch, capsys):
    llm = ModelLLM()
    monkeypatch.setattr(cli, "Hub", lambda settings: Hub(settings, llm=llm, ollama=FakeOllama()))  # type: ignore[arg-type]
    asyncio.run(cli.ask(make_settings(tmp_path, port=9356, private_by_default=True, private_mode="block"), "a secret"))
    assert llm.models == ["ollama_chat/tiny"] and "[ceo -> you] ok" in capsys.readouterr().out


class Delivered:
    async def call_tool(self, name, args):
        return type("Result", (), {"content": [type("Text", (), {"text": '{"delivered": true}'})()],
                                   "is_error": False, "structured_content": None})()
