"""Plan and act modes. In plan mode the CEO may only read (tools that declare readOnlyHint, memory search) and
propose a plan; hiring, messaging agents and every other action are refused in code. The plan shows each hire's
model, reservation and memory, and the user's "approve" carries it out in act mode, approving its hires
(fake LLMs, no network)."""
import asyncio
import sys
from pathlib import Path

import httpx
import httpx2
import pytest
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers import get_message_text, new_text_message
from a2a.types import Role, SendMessageRequest
from conftest import FakeOllama, make_settings
from google.protobuf.json_format import MessageToDict
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from openai import AsyncOpenAI
from test_interop import PORT, call, reply, serve, user

from hashiru import cli, planning, runtime
from hashiru.approvals import Approvals
from hashiru.config import McpServer, load_settings
from hashiru.hub import Hub
from hashiru.interop import MODE_HEADER, MODE_KEY, PLAN_KEY
from hashiru.interop.activity import describe
from hashiru.interop.mcp_public import MCP_PATH
from hashiru.obfuscation.vault import Span
from hashiru.peers import fetch_card

ECHO = str(Path(__file__).parent / "servers" / "echo_server.py")
READ_ONLY = {"check_usage", "get_website", "arxiv_search", "read_file", "memory"}
PLAN = {"summary": "w1 greets; I pass its greeting on.",
        "hires": [{"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "budget_usd": 0.3,
                   "task": "Say hello."}]}
STEPPED = {"summary": "w1 finds facts; w2 writes them up.",
           "hires": [{"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "budget_usd": 0.1,
                      "task": "Find facts.", "prompt": "Find facts about what you are asked."},
                     {"name": "w2", "role": "worker", "model": "gpt-4.1-mini", "budget_usd": 0.1,
                      "task": "Write it up."}],
           "steps": [{"id": "find", "agent": "w1", "task": "Find three facts about otters."},
                     {"id": "write", "agent": "w2", "task": "Write a paragraph from these facts.",
                      "after": ["find"]}]}


class PlanLLM:
    """Plan mode (propose_plan is offered): the CEO checks its pool, searches memory and proposes hiring w1 with
    $0.30, then replies. Act mode: it hires w1 (and w2 with $0.30 too, when `extra`), asks w1 and answers with its
    reply."""

    def __init__(self, extra: bool = False) -> None:
        self.extra = extra
        self.offered: list[list[str]] = []   # tool names, per CEO call
        self.systems: list[str] = []
        self.prompts: list[str] = []
        self.results: list[list[str]] = []   # tool results the CEO saw, per CEO call

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == "Greet people.":
            return reply("hello from w1"), 0.001
        names = [t["function"]["name"] for t in tools or []]
        results = [m["content"] for m in messages if m["role"] == "tool"]
        self.offered.append(names)
        self.systems.append(messages[0]["content"])
        self.prompts.append(messages[1]["content"])
        self.results.append(results)
        if "propose_plan" in names:
            if not results:
                return reply(calls=[call("check_usage", {"pool": "ceo"}, "p1"),
                                    call("memory", {"action": "search", "query": "hello"}, "p2"),
                                    call("propose_plan", PLAN, "p3")]), 0.002
            return reply("I will hire w1 to say hello."), 0.002
        if not results:
            hires = [call("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini",
                                         "prompt": "Greet people.", "budget_usd": 0.3}, "c1")]
            if self.extra:
                hires.append(call("hire_agent", {"name": "w2", "role": "worker", "model": "gpt-4.1-mini",
                                                 "prompt": "Greet people.", "budget_usd": 0.3}, "c2"))
            return reply(calls=hires), 0.002
        if not any("hello from w1" in r for r in results):
            return reply(calls=[call("ask_agent", {"name": "w1", "text": "say hello"}, "c3")]), 0.002
        return reply("w1 says hello"), 0.002


async def turn(convs, conv, text, **kw):
    return [e async for e in convs.ask(conv, text, **kw)][-1]


# ---- the mode ----

def test_plan_and_act_prefixes():
    assert planning.split("/plan compare two papers") == ("plan", "compare two papers")
    assert planning.split("  /act go\nnow") == ("act", "go\nnow")
    assert planning.split("/Plan x") == ("plan", "x")
    assert planning.split("/planet x") == (None, "/planet x")
    assert planning.split("/plan") == (None, "/plan")  # nothing to plan: sent as it is
    assert planning.split("hello") == (None, "hello")


def test_default_mode_setting():
    assert load_settings().default_mode == "act"
    assert load_settings(["default_mode=plan"]).default_mode == "plan"
    with pytest.raises(ValueError, match="default_mode"):
        load_settings(["default_mode=auto"])


# ---- what the CEO may do while planning ----

def test_plan_mode_refuses_every_action_in_code(hub, session):
    session.planning = True
    for name, args in [("hire_agent", {"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "budget_usd": 0.1}),
                       ("ask_agent", {"name": "w1", "text": "hi"}), ("fire_agent", {"name": "w1"}),
                       ("deliver_to_user", {"text": "hi"}), ("memory", {"action": "add", "key": "k", "text": "t"}),
                       ("memory", {"action": "delete", "key": "k"}), ("no_such_tool", {})]:
        out = asyncio.run(runtime.dispatch(hub, session, None, "ceo", call(name, args, "c")))  # mcp=None: never used
        assert "while planning" in out["error"], name
    assert list(session.ledger.agents) == ["ceo"]
    session.planning = False
    out = asyncio.run(runtime.dispatch(hub, session, None, "ceo", call("propose_plan", PLAN, "c")))
    assert "plan mode" in out["error"] and session.plan is None


def test_propose_plan_previews_each_hire_and_the_totals(hub, session):
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", {"summary": "s", "hires": [
        {"name": "w1", "role": "worker", "difficulty": "easy", "task": "a"},
        {"name": "m1", "role": "manager", "difficulty": "hard", "budget_usd": 0.2, "task": "b"}]}))
    assert [(h["name"], h["model"], h["tier"], h["budget_usd"], h["gpu_mb"]) for h in out["hires"]] == [
        ("w1", "ollama_chat/tiny", "local", 0.0, 50), ("m1", "gpt-4.1-mini", "cloud", 0.2, 0)]
    assert (out["total_usd"], out["total_mb"], out["left_usd"], out["problems"]) == (0.2, 50, 0.5, [])
    assert session.plan is not None and list(session.ledger.agents) == ["ceo"]  # nobody hired, nothing reserved
    assert session.ledger.left()["ceo"] == 0.5


def test_propose_plan_lists_the_problems_a_hire_would_hit(hub, session):
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", {"summary": "s", "hires": [
        {"name": "ceo", "role": "worker", "model": "gpt-4.1-mini"},
        {"name": "w2", "role": "boss", "model": "gpt-4.1-mini"},
        {"name": "w3", "role": "worker", "model": "gpt-5"},
        {"name": "w4", "role": "worker"},
        {"name": "w5", "role": "worker", "model": "gpt-4.1-mini", "budget_usd": 0.6},
        {"name": "w5", "role": "worker", "model": "gpt-4.1-mini"}]}))
    problems = "\n".join(out["problems"])
    for expected in ("'ceo'", "role must be", "gpt-5", "difficulty", "w5 twice", "$0.6"):
        assert expected in problems
    assert len(out["problems"]) == 6 and [h["name"] for h in out["hires"]] == ["w5"]


def test_a_plan_with_hires_is_flagged_where_hiring_is_off(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, allow_hiring=False), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    session.planning = True
    hires = {"summary": "s", "hires": [{"name": "w1", "role": "worker", "model": "gpt-4.1-mini"}]}
    assert "allow_hiring" in "\n".join(asyncio.run(planning.propose(hub, session, "ceo", hires))["problems"])
    assert asyncio.run(planning.propose(hub, session, "ceo", {"summary": "s", "hires": []}))["problems"] == []


def test_a_planned_hire_is_approved_once_and_only_as_planned():
    approvals = Approvals()
    approvals.planned = {"w1": ("gpt-4.1", 0.3), "w2": ("gpt-4.1", 0.3)}
    assert not approvals.in_plan("m1", "w1", "gpt-4.1", 0.3)        # only the CEO's own hires
    assert not approvals.in_plan("ceo", "w1", "gpt-4.1-mini", 0.3)  # on another model
    assert not approvals.in_plan("ceo", "w1", "gpt-4.1", 0.31)      # reserving more
    assert approvals.in_plan("ceo", "w1", "gpt-4.1", 0.2)           # less is fine
    assert not approvals.in_plan("ceo", "w1", "gpt-4.1", 0.2)       # once
    assert approvals.in_plan("ceo", "w2", "gpt-4.1", 0.3)


# ---- steps ----

def test_plan_execution_setting():
    assert load_settings().plan_execution == "graph"
    assert load_settings(["plan_execution=ceo"]).plan_execution == "ceo"
    with pytest.raises(ValueError, match="plan_execution"):
        load_settings(["plan_execution=fast"])


def test_propose_plan_compiles_steps_and_shows_them(hub, session):
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", STEPPED))
    assert out["problems"] == [] and [s["id"] for s in out["steps"]] == ["find", "write"]
    assert out["steps"][1]["after"] == ["find"] and "graph" not in out  # the CEO is not sent the drawing
    assert out["hires"][0]["prompt"] == "Find facts about what you are asked." and "prompt" not in out["hires"][1]
    assert session.plan is not None and session.plan.program is not None
    shown, public = planning.end(session, "my plan")
    assert "| write | w2 | find | Write a paragraph from these facts. |" in shown
    assert "runs this plan as a graph" in shown
    assert "step-find --> step-write" in public["graph"] and "fire-w2 --> answer" in public["graph"]
    assert [s["id"] for s in public["steps"]] == ["find", "write"]


def test_steps_that_cannot_run_are_plan_problems(hub, session):
    session.planning = True
    bad = STEPPED | {"steps": [{"id": "find", "agent": "w9", "task": "x", "after": ["write"]},
                               {"id": "write", "agent": "w2", "task": "y", "after": ["find"]}, "junk"]}
    out = asyncio.run(planning.propose(hub, session, "ceo", bad))
    problems = "\n".join(out["problems"])
    for expected in ("w9", "find, write", "w1 is hired but no step", "each step must be an object"):
        assert expected in problems, expected
    assert session.plan is not None and session.plan.program is None
    shown, public = planning.end(session, "my plan")
    assert "graph" not in public and "runs this plan as a graph" not in shown


def test_a_plan_without_steps_gets_a_step_per_hire(hub, session):
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", PLAN))
    assert out["steps"] == [{"id": "w1", "agent": "w1", "task": "Say hello.", "after": []}] and out["problems"] == []
    shown, public = planning.end(session, "my plan")
    assert "| Step |" not in shown and "runs this plan as a graph" in shown and "graph" in public


def test_steps_can_use_agents_hired_before(hub, session):
    session.ledger.add("old", "worker", "cloud", "gpt-4.1-mini", "p", "ceo", 1, 0.0)
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", {"summary": "s", "hires": [], "steps": [
        {"id": "again", "agent": "old", "task": "do it again"}]}))
    shown, _ = planning.end(session, "my plan")
    assert out["problems"] == [] and "No new hires." in shown and "| again | old |" in shown


def test_approve_hands_a_compiled_plan_to_the_ceos_run(hub, session):
    session.planning = True
    asyncio.run(planning.propose(hub, session, "ceo", STEPPED))
    planning.end(session, "my plan")
    prompt = planning.begin(hub, session, "approve")
    assert session.approved is not None and session.approved.program is not None and session.plan is None
    assert "the hub carried it out" in prompt and "w1 finds facts; w2 writes them up." in prompt
    assert session.approvals.planned == {"w1": ("gpt-4.1-mini", 0.1), "w2": ("gpt-4.1-mini", 0.1)}


def test_ceo_execution_tells_the_ceo_the_steps(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, plan_execution="ceo"), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", STEPPED))
    assert len(out["steps"]) == 2 and session.plan is not None and session.plan.program is None
    planning.end(session, "my plan")
    prompt = planning.begin(hub, session, "approve")
    assert "The user approved your plan. Carry it out now" in prompt and session.approved is None
    assert "- write: send w2: Write a paragraph from these facts. (after find, with their results)" in prompt


def test_begin_clears_a_stale_approved_plan(hub, session):
    session.approved = planning.Plan()
    planning.begin(hub, session, "hello")
    assert session.approved is None


# ---- in a conversation ----

def test_a_plan_turn_hires_nobody_and_approve_carries_it_out(tmp_path):
    llm = PlanLLM()

    async def body(hub, base):
        convs = hub.conversations
        conv = await convs.open("chat")
        first = await turn(convs, conv, "/plan say hello")
        after_plan = list(conv.session.ledger.agents)
        second = await turn(convs, conv, "approve")
        return first, after_plan, second, list(conv.session.trace.events), conv.history

    first, after_plan, second, events, history = serve(tmp_path, body, llm, approve_hire_usd=0.2,
                                                       plan_execution="ceo")
    assert set(llm.offered[0]) == READ_ONLY | {"propose_plan"}      # no hiring, messaging or delivering
    assert "plan mode" in llm.systems[0] and "plan mode" not in llm.systems[-1]
    assert "api_usd" in llm.results[1][0] and "memories" in llm.results[1][1]  # reading works
    assert after_plan == ["ceo"]
    assert [(h["name"], h["model"], h["budget_usd"]) for h in first["plan"]["hires"]] == [("w1", "gpt-4.1-mini", 0.3)]
    assert first["text"].startswith("I will hire w1 to say hello.")
    assert "| w1 | worker | gpt-4.1-mini | $0.30 |" in first["text"] and "Reply **approve**" in first["text"]
    assert "approval" not in first
    assert second["reply"] == "w1 says hello" and "plan" not in second and "approval" not in second
    assert "The user approved your plan" in llm.prompts[-1] and "w1" in llm.prompts[-1]
    actions = [e["action"] for e in events if e["kind"] == "approval"]
    assert actions == ["planned"]                                   # $0.30 > approve_hire_usd, but it was planned
    assert not [e for e in events if e["kind"] == "step"]           # plan_execution=ceo: the CEO, not a graph
    assert [said for said, _ in history] == ["/plan say hello", "approve"]


def test_a_hire_outside_the_plan_still_asks(tmp_path):
    llm = PlanLLM(extra=True)

    async def body(hub, base):
        convs = hub.conversations
        conv = await convs.open("chat")
        await turn(convs, conv, "say hello", mode="plan")
        return await turn(convs, conv, "approve")

    second = serve(tmp_path, body, llm, approve_hire_usd=0.2, budget_usd=1.0,  # the pool covers both hires
                   plan_execution="ceo")
    assert [a["what"] for a in second["approval"]] == ["ceo wants to hire w2 on gpt-4.1-mini with $0.30 reserved"]


def test_feedback_revises_the_plan_and_act_mode_drops_it(tmp_path):
    llm = PlanLLM()

    async def body(hub, base):
        convs = hub.conversations
        conv = await convs.open("chat")
        first = await turn(convs, conv, "say hello", mode="plan")
        second = await turn(convs, conv, "make it cheaper")        # a plan waits: this is feedback on it
        hired = list(conv.session.ledger.agents)
        third = await turn(convs, conv, "/act just say hello")      # leave plan mode without approving
        return first, second, hired, third, conv.session.plan

    first, second, hired, third, left_over = serve(tmp_path, body, llm)
    assert "plan" in first and "plan" in second and hired == ["ceo"]
    assert "make it cheaper" in llm.prompts[2] and "propose_plan" in llm.offered[2]
    assert third["reply"] == "w1 says hello" and "plan" not in third and left_over is None
    assert "The user approved" not in llm.prompts[-1]


def test_default_mode_plan_plans_each_new_request(tmp_path):
    llm = PlanLLM()

    async def body(hub, base):
        convs = hub.conversations
        conv = await convs.open("chat")
        return [await turn(convs, conv, text) for text in ("say hello", "approve", "and again")]

    first, second, third = serve(tmp_path, body, llm, default_mode="plan", plan_execution="ceo")
    assert "plan" in first and second["reply"] == "w1 says hello" and "plan" in third
    assert "w1 is taken" in "\n".join(third["plan"]["problems"])  # w1 is still hired


def test_the_user_sees_a_cloud_ceos_plan_with_private_values_restored(hub, session):
    masked = session.vault.mask("email ann@example.com", [Span(6, 21, "EMAIL_ADDRESS")])
    session.planning = True
    asyncio.run(planning.propose(hub, session, "ceo", {"summary": "s", "hires": [
        {"name": "w1", "role": "worker", "difficulty": "easy", "task": masked}]}))
    shown, public = planning.end(session, "my plan")
    assert "ann@example.com" in shown and public["hires"][0]["task"] == "email ann@example.com"
    assert session.plan is not None and session.plan.hires[0].task == masked  # the CEO keeps the placeholders


# ---- interfaces ----

def test_every_interface_can_ask_for_a_plan_and_gets_it_back(tmp_path):
    async def body(hub, base):
        client = AsyncOpenAI(base_url=f"{base}/v1", api_key="unused", default_headers={MODE_HEADER: "plan"})
        openai = await client.chat.completions.create(model="hashiru", messages=[user("say hello")])
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            bad = await http.post("/chat/completions", headers={MODE_HEADER: "maybe"},
                                  json={"model": "hashiru", "messages": [user("hi")]})
        async with (httpx2.AsyncClient(timeout=30) as http,
                    streamable_http_client(f"{base}{MCP_PATH}", http_client=http) as (read, write, *_),
                    ClientSession(read, write) as mcp):
            await mcp.initialize()
            mcp_out = (await mcp.call_tool("ask", {"text": "say hello", "mode": "plan"})).structured_content
        async with httpx.AsyncClient(timeout=10) as http:
            a2a = ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create(await fetch_card(base, {}, 10))
            message = new_text_message("say hello", role=Role.ROLE_USER)
            message.metadata.update({MODE_KEY: "plan"})
            events = [e async for e in a2a.send_message(SendMessageRequest(message=message))]
            await a2a.close()
        return openai, bad, mcp_out, events

    openai, bad, mcp_out, events = serve(tmp_path, body, PlanLLM())
    assert openai.model_extra["hashiru"]["plan"]["hires"][0]["name"] == "w1"
    assert bad.status_code == 400 and "plan or act" in bad.json()["error"]["message"]
    assert mcp_out["plan"]["hires"][0]["name"] == "w1" and "Reply **approve**" in mcp_out["answer"]
    last = [e.status_update.status for e in events if e.HasField("status_update")][-1]
    assert MessageToDict(last.message.metadata)[PLAN_KEY]["hires"][0]["name"] == "w1"
    assert "Reply **approve**" in get_message_text(last.message)


def test_the_terminal_chat_plans_and_approves(tmp_path, monkeypatch, capsys):
    llm = PlanLLM()
    monkeypatch.setattr(cli, "Hub", lambda settings: Hub(settings, llm=llm, ollama=FakeOllama()))  # type: ignore[arg-type]
    said = iter(["/plan say hello", "approve", "/quit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(said))
    asyncio.run(cli.chat(make_settings(tmp_path, port=PORT, plan_execution="ceo")))
    out = capsys.readouterr().out
    assert "| w1 | worker | gpt-4.1-mini | $0.30 |" in out and "[ceo -> you] w1 says hello" in out
    assert "propose_plan" in llm.offered[0] and "propose_plan" not in llm.offered[-1]

    asyncio.run(cli.ask(make_settings(tmp_path, port=PORT), "/plan say hello"))  # one shot: the plan, nothing hired
    out = capsys.readouterr().out
    assert "| w1 | worker | gpt-4.1-mini | $0.30 |" in out and "w1 says hello" not in out


def test_third_party_tools_that_declare_read_only_can_be_used_while_planning(tmp_path):
    async def body():
        server = McpServer("echo", command=sys.executable, args=(ECHO,))
        hub = Hub(make_settings(tmp_path, port=PORT, mcp_servers=(server,), tool_start_timeout=20),
                  ollama=FakeOllama())  # type: ignore[arg-type]
        async with hub.serving():
            return {t.name for t in await hub.mcp.list_tools() if planning.readable(t)}
    readable = asyncio.run(body())
    assert "echo__shout_loud" in readable and "echo__env" not in readable and "echo__die" not in readable
    assert readable >= READ_ONLY and not {"hire_agent", "fire_agent", "deliver_to_user"} & readable


def test_plan_events_have_activity_lines():
    proposed = {"kind": "tool", "agent": "ceo", "name": "propose_plan", "args": {},
                "result": {"hires": [{"name": "w1"}, {"name": "w2"}], "total_usd": 0.3, "problems": []}}
    assert describe(proposed) == "ceo proposed a plan: 2 hires, $0.30 reserved"
    proposed["result"]["steps"] = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert describe(proposed) == "ceo proposed a plan: 2 hires and 3 steps, $0.30 reserved"
    for status, why, line in [("done", "", "step write (w2) done"),
                              ("skipped", "it needs find, which failed", "step write (w2) skipped: it needs find, "
                                                                         "which failed")]:
        assert describe({"kind": "step", "agent": "ceo", "step": "write", "to": "w2", "status": status,
                         "why": why}) == line
    what = "ceo wants to hire w1 on gpt-4.1 with $0.30 reserved"
    assert describe({"kind": "approval", "action": "planned", "what": what}) == f"approved with your plan: {what}"
