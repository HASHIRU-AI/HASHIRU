"""End to end with a scripted LLM: real MCP, A2A and auth on a test port, no tokens spent."""
import asyncio
import json
from types import SimpleNamespace

import httpx
from conftest import FakeOllama, make_settings

from hashiru.hub import Hub
from hashiru.runtime import ask_agent
from hashiru.session import USER

PORT = 9199


def call(name, args, cid):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def reply(content="", calls=None):
    return SimpleNamespace(content=content, tool_calls=calls)


class ScriptedLLM:
    """The CEO checks its pool, hires a worker, asks it, and answers. The worker says hello."""

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == "Greet people.":
            return reply("hello from w1"), 0.001
        turn = sum(1 for m in messages if m["role"] == "tool")
        script = [reply(calls=[call("check_usage", {"pool": "ceo"}, "c1")]),
                  reply(calls=[call("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini",
                                                   "prompt": "Greet people.", "budget_usd": 0.1}, "c2")]),
                  reply(calls=[call("ask_agent", {"name": "w1", "text": "say hello"}, "c3")]),
                  reply("w1 says: hello from w1")]
        return script[turn], 0.002


def test_ceo_hires_asks_and_traces(tmp_path):
    hub = Hub(make_settings(tmp_path, port=PORT), llm=ScriptedLLM(), ollama=FakeOllama())  # type: ignore[arg-type]

    async def go():
        async with hub.serving():
            session = hub.new_session("smoke")
            answer = await ask_agent(hub, session, USER, "ceo", "Get a greeting.")
            async with httpx.AsyncClient(base_url=hub.settings.internal_url) as http:
                card = f"/s/{session.id}/agents/w1/.well-known/agent-card.json"
                statuses = {
                    "mcp_without_token": (await http.post("/mcp", json={})).status_code,
                    "card_as_ceo": (await http.get(card, headers=bearer(session, "ceo"))).status_code,
                    "card_as_user": (await http.get(card, headers=bearer(session, USER))).status_code,
                }
            return session, answer, statuses

    session, answer, statuses = asyncio.run(go())
    assert answer == "w1 says: hello from w1"
    assert statuses == {"mcp_without_token": 401, "card_as_ceo": 200, "card_as_user": 403}
    events = [json.loads(line) for line in session.trace.path.read_text().splitlines()]
    assert events[0]["kind"] == "start"
    assert [e["name"] for e in events if e["kind"] == "tool"] == ["check_usage", "hire_agent"]
    assert [(e["agent"], e["to"]) for e in events if e["kind"] == "msg"] == [
        (USER, "ceo"), ("ceo", "w1"), ("w1", "ceo"), ("ceo", USER)]
    check = next(e for e in events if e["kind"] == "tool" and e["name"] == "check_usage")
    assert check["result"]["api_usd"] == 0.498          # 0.5 minus the CEO's first LLM call
    assert session.ledger.agents["w1"].spent == 0.001
    assert round(session.ledger.left()["ceo"], 6) == round(0.5 - 4 * 0.002 - 0.1, 6)


def bearer(session, name):
    return {"Authorization": f"Bearer {session.tokens[name]}"}
