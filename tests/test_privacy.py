"""Phase 8a: R3, private stays local, enforced while agents run. Data labeled private never reaches a cloud model
or a remote agent, and an agent holding it does not send it off the machine (fake LLMs, no network)."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import FakeOllama, ctx, make_settings

from hashiru import hub as hub_module
from hashiru import runtime
from hashiru.config import A2APeer, McpServer, load_settings
from hashiru.hub import Hub
from hashiru.interop.activity import describe
from hashiru.interop.openai_api import wants_private
from hashiru.llm import is_local
from hashiru.session import USER
from hashiru.trace import PRIVATE

SALARY = '{"path": "notes.md", "text": "my salary is 100k"}'


def result(text, is_error=False):
    return SimpleNamespace(content=[SimpleNamespace(text=text)], is_error=is_error, structured_content=None)


class RecordingMCP:
    """Answers every tool call with `text` and remembers which tools ran."""

    def __init__(self, text=SALARY):
        self.text, self.ran = text, []

    async def call_tool(self, name, args):
        self.ran.append(name)
        return result(self.text)


def policy(session):
    return [e for e in session.trace.events if e["kind"] == "policy"]


def labels(session, agent):
    return session.trace.labels(agent, session.id)


def add(session, name, tier, model, role="worker", parent="ceo"):
    return session.ledger.add(name, role, tier, model, "p", parent, 1, 0.1)


def make_private(session, agent):
    session.trace.receive(agent, session.id, [PRIVATE])


# ---- tools ----

def test_a_cloud_agent_cannot_use_a_tool_that_returns_private_data(session):
    mcp = RecordingMCP()
    out = asyncio.run(runtime.call_tool(session, mcp, "ceo", "read_file", {"path": "notes.md"}))  # the CEO is cloud
    assert out["error"].startswith("R3") and mcp.ran == []        # refused before the tool ran
    assert PRIVATE not in labels(session, "ceo")
    assert [(e["rule"], e["agent"], e["action"]) for e in policy(session)] == [("R3", "ceo", "tool")]
    add(session, "m1", "local", "ollama_chat/tiny", role="manager")
    assert asyncio.run(runtime.call_tool(session, mcp, "m1", "read_file", {"path": "notes.md"}))["text"] \
        == "my salary is 100k" and PRIVATE in labels(session, "m1")  # a local agent may read it


def test_private_data_a_tool_attaches_to_its_result_is_withheld_from_a_cloud_agent(session):
    mcp = RecordingMCP('{"answer": "100k", "_labels": ["private"]}')
    out = asyncio.run(runtime.call_tool(session, mcp, "ceo", "srv__lookup", {}, sources={}))
    assert out["error"].startswith("R3") and mcp.ran == ["srv__lookup"]  # it ran, but the CEO never sees the result
    assert PRIVATE not in labels(session, "ceo")
    tool = next(e for e in session.trace.events if e["kind"] == "tool")
    assert "100k" not in json.dumps(tool)                          # the trace shows what the agent got
    assert [e["action"] for e in policy(session)] == ["withheld"]


def test_an_agent_holding_private_data_cannot_send_data_off_the_machine(session):
    add(session, "m1", "local", "ollama_chat/tiny", role="manager")
    add(session, "m2", "local", "ollama_chat/tiny", role="manager")
    make_private(session, "m1")
    mcp = RecordingMCP('{"text": "page"}')
    web = {"url": "https://example.com/?q=salary"}
    refused = asyncio.run(runtime.call_tool(session, mcp, "m1", "get_website", web, egress={"get_website"}))
    assert refused["error"].startswith("R3") and mcp.ran == []
    assert asyncio.run(runtime.call_tool(session, mcp, "m2", "get_website", web, egress={"get_website"})) \
        == {"text": "page"}                                        # an agent without private data may
    assert asyncio.run(runtime.call_tool(session, mcp, "m1", "get_website", web, egress=())) == {"text": "page"}
    assert [e["action"] for e in policy(session)] == ["egress"]


def test_tools_that_leave_the_machine(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, tool_creation=True), ollama=ollama)  # type: ignore[arg-type]
    fetch, files = McpServer("fetch", command="uvx"), McpServer("files", command="npx", egress=False)
    hub.gateway.routes = {"fetch__fetch": (fetch, "fetch"), "files__read": (files, "read")}
    egress = hub.egress()
    assert {"get_website", "arxiv_search", "create_tool", "fetch__fetch"} <= egress
    assert not {"files__read", "read_file", "memory", "deliver_to_user", "check_usage"} & egress


def test_the_hub_applies_private_egress_to_every_call(tmp_path, ollama):
    def attempt(egress_setting):
        hub = Hub(make_settings(tmp_path, private_egress=egress_setting), ollama=ollama)  # type: ignore[arg-type]
        session = hub.new_session("t", trace=False)
        add(session, "m1", "local", "ollama_chat/tiny", role="manager")
        make_private(session, "m1")
        call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_website",
                                                                 arguments=json.dumps({"url": "https://a.example"})))
        return asyncio.run(runtime.dispatch(hub, session, RecordingMCP('{"text": "page"}'), "m1", call))
    assert attempt("block")["error"].startswith("R3")
    assert attempt("allow") == {"text": "page"}


# ---- messages and hires ----

def test_a_message_carrying_private_data_is_not_sent_to_a_cloud_or_remote_agent(hub, session):
    add(session, "m1", "local", "ollama_chat/tiny", role="manager")
    add(session, "w1", "cloud", "gpt-4.1-mini", parent="m1")
    add(session, "p1", "remote", "a2a/p1", parent="m1")
    make_private(session, "m1")
    for target in ("w1", "p1"):
        reply = asyncio.run(runtime.ask_agent(hub, session, "m1", target, "the salary is 100k"))
        assert reply.startswith("error: R3") and labels(session, target) == set()
    assert not [e for e in session.trace.events if e["kind"] == "msg"]  # nothing was sent
    assert [e["action"] for e in policy(session)] == ["message", "message"]


def test_a_reply_carrying_private_data_is_withheld_from_a_cloud_hirer(hub, session, monkeypatch):
    add(session, "m1", "cloud", "gpt-4.1-mini", role="manager")
    add(session, "w1", "local", "ollama_chat/tiny", parent="m1")
    make_private(session, "w1")                                    # w1 read a private file

    async def fake_send(hub_, session_, me, name, text):
        return "the salary is 100k"
    monkeypatch.setattr(runtime, "a2a_send", fake_send)
    reply = asyncio.run(runtime.ask_agent(hub, session, "m1", "w1", "what does the file say?"))
    assert reply.startswith("error: R3") and "100k" not in reply
    assert PRIVATE not in labels(session, "m1")
    assert [e["agent"] for e in session.trace.events if e["kind"] == "msg"] == ["m1"]  # the question, not the answer
    assert [e["action"] for e in policy(session)] == ["reply"]
    assert asyncio.run(runtime.ask_agent(hub, session, "m1", "w1", "done?")) == "error: R3 (private stays local): " \
        "w1's reply carries private data and m1 runs on a cloud model"


def test_an_agent_holding_private_data_may_hire_only_local_agents(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, a2a_peers=(A2APeer("p1", "http://127.0.0.1:9/"),)), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    make_private(session, "ceo")

    def hire(name, model):
        return asyncio.run(hub.tools.hire_agent(ctx(session, "ceo"), name, "worker", model, "p", 0.05))  # type: ignore[arg-type]
    assert hire("c1", "gpt-4.1-mini")["error"].startswith("R3")
    assert hire("r1", "a2a/p1")["error"].startswith("R3")         # refused before contacting the peer
    assert hire("l1", "ollama_chat/tiny")["hired"] == "l1"
    assert [e["action"] for e in policy(session)] == ["hire", "hire"]


# ---- memory recall and the front door ----

def test_recalled_memories_are_not_injected_into_a_cloud_ceo(tmp_path, ollama):
    def recall(ceo_model):
        hub = Hub(make_settings(tmp_path, ceo_model=ceo_model, memory_autoinject=True), ollama=ollama)  # type: ignore[arg-type]
        session = hub.new_session("t", trace=False)
        session.memory.add("pet", "my cat is named Miso", [])
        return hub.recall(session, "what is my cat named?"), session
    text, session = recall("gpt-4.1-mini")
    assert "Miso" not in text and PRIVATE not in labels(session, "ceo")
    assert [e["action"] for e in policy(session)] == ["recall"]
    text, session = recall("ollama_chat/tiny")
    assert "Miso" in text and PRIVATE in labels(session, "ceo")


def test_a_private_chat_moves_a_cloud_ceo_to_a_local_model(hub, session):
    assert asyncio.run(hub.make_private(session)) is None
    ceo = session.ledger.agents["ceo"]
    assert (ceo.tier, ceo.model) == ("local", "ollama_chat/tiny")  # the first local model in models
    assert labels(session, USER) == {PRIVATE}                     # everything the user says from now on is private
    path = f"/s/{session.id}/agents/ceo"
    assert [getattr(r, "path", None) for r in hub.app.router.routes].count(path) == 1  # remounted once
    assert [e["action"] for e in policy(session)] == ["private"]
    assert asyncio.run(hub.make_private(session)) is None and len(policy(session)) == 1  # already private


def test_a_private_chat_needs_a_local_model(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, models=("gpt-4.1-mini",)), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    assert "no local model" in asyncio.run(hub.make_private(session))
    assert labels(session, USER) == set() and session.ledger.agents["ceo"].tier == "cloud"


def test_privacy_settings():
    assert load_settings(["private_ceo_model=ollama_chat/gemma4:12b"]).private_ceo_model == "ollama_chat/gemma4:12b"
    with pytest.raises(ValueError, match="private_ceo_model"):
        load_settings(["private_ceo_model=gpt-4.1-mini"])
    with pytest.raises(ValueError, match="private_egress"):
        load_settings(["private_egress=sometimes"])
    server = load_settings(['+mcp_servers.clock={command: python, args: [c.py], egress: false}']).mcp_servers[0]
    assert server.egress is False
    assert load_settings(['+mcp_servers.f={command: uvx}']).mcp_servers[0].egress is True


# ---- from the review: deliveries, settings that fail closed, what "local" means ----

def test_a_delivery_carrying_private_data_makes_the_chat_private(session):
    add(session, "m1", "local", "ollama_chat/tiny", role="manager")
    make_private(session, "m1")
    asyncio.run(runtime.call_tool(session, RecordingMCP('{"delivered": true}'), "m1", "deliver_to_user",
                                  {"text": "your salary is 100k"}))
    assert labels(session, USER) == {PRIVATE}  # the chat history now holds it, so the next turn must stay local


def test_a_private_chat_on_a_cloud_ceo_moves_it_even_after_the_user_turned_private(hub, session):
    session.trace.receive(USER, session.id, [PRIVATE])  # e.g. a delivery in the last turn
    assert asyncio.run(hub.make_private(session)) is None
    assert session.ledger.agents["ceo"].tier == "local"


def test_private_ceo_model_is_used_even_when_the_ceo_is_already_local(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, ceo_model="ollama_chat/tiny", private_ceo_model="ollama_chat/big",
                            models=("gpt-4.1-mini", "ollama_chat/tiny", "ollama_chat/big")), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    assert asyncio.run(hub.make_private(session)) is None
    assert session.ledger.agents["ceo"].model == "ollama_chat/big"


def test_the_switch_checks_free_memory(hub, session, monkeypatch):
    monkeypatch.setattr(hub_module, "free_mb", lambda: 10)  # "tiny" needs 50 MB
    why = asyncio.run(hub.make_private(session))
    assert "needs 50 MB" in why and session.ledger.agents["ceo"].tier == "cloud"
    assert [e["action"] for e in policy(session)] == ["private_failed"]


def test_the_ceos_card_names_its_new_model(hub, session):
    asyncio.run(hub.make_private(session))

    async def card():
        transport = httpx.ASGITransport(app=hub.asgi)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:9200") as http:
            r = await http.get(f"/s/{session.id}/agents/ceo/.well-known/agent-card.json",
                               headers={"Authorization": f"Bearer {session.tokens[USER]}"})
            return r.json()
    assert asyncio.run(card())["description"] == "manager on ollama_chat/tiny"


def test_the_private_event_records_the_model(hub, session):
    asyncio.run(hub.make_private(session))
    event = policy(session)[0]
    assert (event["model"], event["tier"]) == ("ollama_chat/tiny", "local")
    assert describe(event).startswith("R3 (private stays local): ")


def test_created_tools_leave_the_machine(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, tool_creation=True), ollama=ollama)  # type: ignore[arg-type]
    hub.toolsmith.procs["made"] = None  # type: ignore[assignment]
    assert "made" in hub.egress()


def test_privacy_settings_built_directly_fail_closed(tmp_path, ollama):
    typo = Hub(make_settings(tmp_path, private_egress="blcok"), ollama=ollama)  # type: ignore[arg-type]
    session = typo.new_session("t", trace=False)
    add(session, "m1", "local", "ollama_chat/tiny", role="manager")
    make_private(session, "m1")
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_website", arguments='{"url": "https://a.b"}'))
    assert asyncio.run(runtime.dispatch(typo, session, RecordingMCP(), "m1", call))["error"].startswith("R3")
    cloud = Hub(make_settings(tmp_path, private_ceo_model="gpt-4.1-mini"), ollama=ollama)  # type: ignore[arg-type]
    s2 = cloud.new_session("t", trace=False)
    assert "not a local model" in asyncio.run(cloud.make_private(s2)) and s2.ledger.agents["ceo"].tier == "cloud"


def test_ollama_cloud_tags_are_not_local():
    assert not is_local("ollama_chat/gpt-oss:120b-cloud") and not is_local("ollama/deepseek-v3.1:671b-cloud")
    assert is_local("ollama_chat/qwen3.5:4b")


def test_more_privacy_settings():
    with pytest.raises(ValueError, match="egress"):
        load_settings(['+mcp_servers.r={url: "https://example.com/mcp", egress: false}'])
    with pytest.raises(ValueError, match="private_by_default"):
        load_settings(["private_by_default=true", "ceo_model=gpt-4.1-mini", "models=[gpt-4.1-mini]"])


def test_a_chat_that_cannot_go_private_answers_why_and_stays_usable(tmp_path):
    class DownOllama(FakeOllama):
        async def sizes(self):
            raise httpx.ConnectError("connection refused")
    hub = Hub(make_settings(tmp_path), ollama=DownOllama())  # type: ignore[arg-type]

    async def go():
        conv = await hub.conversations.open("t")
        events = [e async for e in hub.conversations.ask(conv, "a secret", private=True)]
        return conv, events
    conv, events = asyncio.run(go())
    assert events[-1]["kind"] == "answer" and "Ollama" in events[-1]["text"]
    assert not conv.busy and conv.id in hub.conversations.open_  # not stuck, not closed: nothing ran


@pytest.mark.parametrize(("value", "private"), [("1", True), ("TRUE", True), ("yes", True), ("on", True),
                                                ("0", False), ("false", False), ("", False)])
def test_the_private_header_values(value, private):
    assert wants_private(value) is private
