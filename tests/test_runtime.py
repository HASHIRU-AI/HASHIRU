"""The agent runtime: tool and message chokepoints, prompts, and error handling (no network)."""
import asyncio
from types import SimpleNamespace

import pytest
from a2a.helpers import new_text_message

from hashiru import runtime
from hashiru.llm import WalletExceeded
from hashiru.session import USER


def team(session):
    """ceo -> m1 -> w1, all cloud."""
    session.ledger.add("m1", "manager", "cloud", "gpt-4.1-mini", "mgr", "ceo", 1, 0.3)
    session.ledger.add("w1", "worker", "cloud", "gpt-4.1-mini", "wkr", "m1", 2, 0.1)


def result(text, is_error=False):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], is_error=is_error, structured_content=None)


def msgs(session):
    return [e for e in session.trace.events if e["kind"] == "msg"]


def test_tool_event_records_the_llm_turn(session):
    class FakeMCP:
        async def call_tool(self, name, args):
            return result('{"pool": "ceo", "api_usd": 1.0, "free_mb": 1, "model_mb": {}}')
    asyncio.run(runtime.call_tool(session, FakeMCP(), "ceo", "check_usage", {"pool": "ceo"}, turn=7))
    assert [e["turn"] for e in session.trace.events if e["kind"] == "tool"] == [7]


def test_agents_can_only_ask_their_own_hires(hub, session):
    team(session)
    for me, name in [("m1", "ceo"), ("ceo", "w1"), ("w1", "m1"), ("m1", "m1"), (USER, "m1"), ("ceo", "nobody")]:
        assert asyncio.run(runtime.ask_agent(hub, session, me, name, "hi")).startswith("error")
    session.ledger.agents["w1"].live = False
    assert asyncio.run(runtime.ask_agent(hub, session, "m1", "w1", "hi")).startswith("error")
    assert not msgs(session)


def test_ask_agent_traces_and_moves_labels(hub, session, monkeypatch):
    ceo = session.ledger.agents["ceo"]
    ceo.tier, ceo.model = "local", "ollama_chat/tiny"  # a local CEO may hold private data (a cloud one: test_privacy)
    session.ledger.add("m1", "manager", "local", "ollama_chat/tiny", "notes", "ceo", 1, 0.0)

    async def fake_send(hub_, session_, me, name, text):
        session.trace.receive("m1", session.id, ["private"])  # m1 read memory while working
        return "your notes say X"
    monkeypatch.setattr(runtime, "a2a_send", fake_send)
    assert asyncio.run(runtime.ask_agent(hub, session, "ceo", "m1", "summarize my notes")) == "your notes say X"
    assert [(m["agent"], m["to"], m["to_tier"], m["labels"]) for m in msgs(session)] == [
        ("ceo", "m1", "local", []), ("m1", "ceo", "local", ["private"])]
    assert session.trace.labels("ceo", session.id) == {"private"}


def test_executor_reports_the_real_cause_inside_an_exception_group(hub, session, monkeypatch):
    async def boom(hub_, session_, name, text):
        raise ExceptionGroup("unhandled errors in a TaskGroup", [WalletExceeded("spent $3.0000; the cap is $2.0")])
    monkeypatch.setattr(runtime, "run_agent", boom)
    context = SimpleNamespace(context_id=session.id, message=new_text_message("hi"))
    with pytest.raises(WalletExceeded):
        asyncio.run(runtime.Executor(hub, session, "ceo").execute(context, None))  # type: ignore[arg-type]
    errors = [e["error"] for e in session.trace.events if e["kind"] == "error"]
    assert errors[0].startswith("WalletExceeded: spent $3")


def test_system_prompt_worker_vs_manager(session):
    team(session)
    assert runtime.system_prompt(session.ledger.agents["w1"]) == "wkr"
    text = runtime.system_prompt(session.ledger.agents["m1"])
    assert "your pool is 'm1'" in text and "Rules you must follow" in text and text.endswith("mgr")


def test_to_openai_keeps_the_schema():
    tool = SimpleNamespace(name="check_usage", description="d", input_schema={
        "type": "object", "properties": {"pool": {"type": "string"}}, "required": ["pool"]})
    fn = runtime.to_openai(tool)["function"]
    assert fn["name"] == "check_usage" and list(fn["parameters"]["properties"]) == ["pool"]
    assert fn["parameters"]["required"] == ["pool"]


def test_unpack_json_and_errors():
    assert runtime.unpack(result('{"api_usd": 0.5}')) == {"api_usd": 0.5}
    assert runtime.unpack(result("Error executing tool: bad args", is_error=True)) == {
        "error": "Error executing tool: bad args"}


def test_dispatch_rejects_malformed_arguments_without_calling_tools(hub, session):
    for bad in ["{not json", "[1, 2]"]:
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="check_usage", arguments=bad))
        assert "error" in asyncio.run(runtime.dispatch(hub, session, None, "ceo", call))  # mcp=None: never touched


# ---- regressions from the Phase 1 code review ----

def test_transport_errors_become_observations(session):
    from mcp.shared.exceptions import MCPError

    class BrokenMCP:
        async def call_tool(self, name, args):
            raise MCPError(code=-32000, message="Server returned an error response")
    out = asyncio.run(runtime.call_tool(session, BrokenMCP(), "ceo", "check_usage", {"pool": "ceo"}))
    assert out["error"].startswith("check_usage failed")
    assert [e["result"] for e in session.trace.events if e["kind"] == "tool"] == [out]


def test_a_fired_agent_makes_no_more_llm_calls(hub, session):
    team(session)
    calls = []

    class CountingLLM:
        async def chat(self, model, messages, tools=None):
            calls.append(model)
            return SimpleNamespace(content="hi", tool_calls=None), 0.0
    hub.llm = CountingLLM()
    session.ledger.agents["w1"].live = False
    assert asyncio.run(runtime.run_agent(hub, session, "w1", "hello")) == "failed" and calls == []
    hub.sessions.pop(session.id)  # a closed session: even live agents stop
    assert asyncio.run(runtime.run_agent(hub, session, "m1", "hello")) == "failed" and calls == []
