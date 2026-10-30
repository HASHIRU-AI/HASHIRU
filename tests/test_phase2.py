"""Phase 2: third-party MCP servers (gateway), persistent memory, recall, arxiv_search and tool scoping."""
import asyncio
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from conftest import FakeOllama, ctx, make_settings
from mcp.server.mcpserver import MCPServer

from hashiru import runtime, web
from hashiru.config import McpServer, load_settings
from hashiru.gateway import hub_name
from hashiru.hub import Hub
from hashiru.memory import SessionMemory, SqliteMemory
from hashiru.trace import PRIVATE

ECHO = str(Path(__file__).parent / "servers" / "echo_server.py")


# ---- config ----

def test_mcp_servers_are_parsed_and_validated():
    s = load_settings(['+mcp_servers.echo={command: python, args: [x.py], env: {K: v}, labels: [private]}',
                       '+mcp_servers.remote={url: "https://example.com/mcp", headers: {Authorization: Bearer t}}'])
    echo, remote = s.mcp_servers
    assert (echo.name, echo.command, echo.args, echo.env, echo.labels) == (
        "echo", "python", ("x.py",), (("K", "v"),), ("private",))
    assert (remote.url, remote.headers, remote.labels) == (
        "https://example.com/mcp", (("Authorization", "Bearer t"),), ("untrusted",))
    for bad in ['+mcp_servers.x={args: [a]}',                                   # neither command nor url
                '+mcp_servers.x={command: a, url: "https://b"}',                # both
                '+mcp_servers.Bad={command: a}',                                # name
                '+mcp_servers.x={command: a, labels: [secret]}',                # label
                '+mcp_servers.x={command: a, shell: true}']:                    # unknown key
        with pytest.raises(ValueError):
            load_settings([bad])


def test_a_config_file_sits_between_defaults_and_the_command_line(tmp_path):
    path = tmp_path / "my.yaml"
    path.write_text("budget_usd: 2.0\nmax_depth: 3\nmcp_servers:\n  echo: {command: python, args: [e.py]}\n")
    s = load_settings(["max_depth=1"], config_file=str(path))
    assert (s.budget_usd, s.max_depth, [m.name for m in s.mcp_servers]) == (2.0, 1, ["echo"])


# ---- memory ----

def test_session_memory_ranks_by_shared_words():
    mem = SessionMemory({"units": "prefers metric units", "tea": "drinks green tea", "x": "unrelated"})
    assert [m["key"] for m in mem.search("which units and tea", 5)][:2] in (["units", "tea"], ["tea", "units"])
    assert len(mem.search("", 5)) == 3 and mem.delete("x") and not mem.delete("x")


def test_sqlite_memory_persists_ranks_and_ignores_query_syntax(tmp_path):
    db = tmp_path / "m.sqlite3"
    first = SqliteMemory(db)
    first.add("units", "prefers metric units")
    first.add("tea", "drinks green tea every morning")
    first.add("units", "prefers metric units, always")  # replaces
    again = SqliteMemory(db)
    assert again.keys() == (["units", "tea"], 2)  # newest first, and the total
    assert again.search("metric", 5) == [{"key": "units", "text": "prefers metric units, always", "labels": []}]
    assert again.search('tea" OR * NEAR( AND', 5)[0]["key"] == "tea"  # FTS5 syntax in the query is just words
    assert again.search("", 1)[0]["key"] == "units"  # most recent first
    assert again.delete("tea") and again.keys() == (["units"], 1)


def test_persistent_memory_is_shared_by_sessions(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, memory_db=str(tmp_path / "m.sqlite3")), ollama=ollama)  # type: ignore[arg-type]
    a, b = hub.new_session("a", trace=False), hub.new_session("b", trace=False)
    hub.tools.memory(ctx(a, "ceo"), "add", key="units", text="prefers metric")  # type: ignore[arg-type]
    assert hub.tools.memory(ctx(b, "ceo"), "search", query="units")["memories"][0]["key"] == "units"  # type: ignore[arg-type]
    assert "error" in hub.tools.memory(ctx(b, "ceo"), "add", key="", text="x")  # type: ignore[arg-type]


def test_recall_adds_memories_and_marks_the_ceo_private(tmp_path, ollama):
    # A local CEO: memories are private, so a cloud CEO gets none (test_privacy).
    hub = Hub(make_settings(tmp_path, ceo_model="ollama_chat/tiny", memory_autoinject=True),
              ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    session.memory.add("units", "the user prefers metric units")
    text = hub.recall(session, "convert 5 miles to the units I like")
    assert "the user prefers metric units" in text and session.trace.labels("ceo", session.id) == {PRIVATE}
    assert [e["kind"] for e in session.trace.events][-2:] == ["recall", "label"]
    assert hub.recall(session, "zzz") == "zzz"  # nothing relevant: unchanged
    off = Hub(make_settings(tmp_path, port=9101), ollama=ollama)  # type: ignore[arg-type]
    s2 = off.new_session("t", trace=False)
    s2.memory.add("units", "metric")
    assert off.recall(s2, "units") == "units"


# ---- arxiv_search ----

ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/abs/2506.04255v1</id><published>2025-06-04T00:00:00Z</published>
<title>HASHIRU: Hierarchical Agent System
 for Hybrid Intelligent Resource Utilization</title><summary>A CEO agent hires employees.</summary>
<author><name>Kunal Pai</name></author><author><name>Parth Shah</name></author>
<link title="pdf" href="http://arxiv.org/pdf/2506.04255v1"/></entry></feed>"""


def test_arxiv_search_parses_the_feed(hub, session, monkeypatch):
    async def public(url):
        return None  # skip DNS in tests; the host is fixed anyway
    monkeypatch.setattr(web, "non_public", public)
    seen = []

    def handler(request):
        seen.append(request.url.params["search_query"])
        return httpx.Response(200, text=ATOM)
    hub.tools.transport = httpx.MockTransport(handler)
    out = asyncio.run(hub.tools.arxiv_search(ctx(session, "ceo"), "hierarchical agents", 3))  # type: ignore[arg-type]
    paper = out["papers"][0]
    assert seen == ["all:hierarchical agents"]
    assert paper["title"] == "HASHIRU: Hierarchical Agent System for Hybrid Intelligent Resource Utilization"
    assert paper["authors"] == ["Kunal Pai", "Parth Shah"] and paper["published"] == "2025-06-04"
    assert paper["pdf"] == "http://arxiv.org/pdf/2506.04255v1"
    hub.tools.transport = httpx.MockTransport(lambda r: httpx.Response(200, text="<not xml"))
    assert "failed" in asyncio.run(hub.tools.arxiv_search(ctx(session, "ceo"), "x"))["error"]  # type: ignore[arg-type]


# ---- tool scoping ----

def hire(hub, session, caller="ceo", **kw):
    args = {"name": "m1", "role": "manager", "model": "gpt-4.1-mini", "prompt": "p", "budget_usd": 0.1} | kw
    return asyncio.run(hub.tools.hire_agent(ctx(session, caller), **args))  # type: ignore[arg-type]


def test_a_manager_can_be_limited_to_some_tools(hub, session):
    assert "hired" in hire(hub, session, tools=["memory", "ask_agent", "hire_agent"], budget_usd=0.3)
    assert session.ledger.agents["m1"].tools == ("ask_agent", "hire_agent", "memory")
    assert "cannot grant" in hire(hub, session, caller="m1", name="m2", tools=["read_file"])["error"]
    assert "hired" in hire(hub, session, caller="m1", name="m2", tools=["memory"])
    assert "workers have none" in hire(hub, session, name="w1", role="worker", tools=["memory"])["error"]
    assert "cannot grant" in hire(hub, session, name="m3", tools=["no_such_tool"])["error"]


def test_the_runtime_refuses_tools_outside_the_scope(hub, session):
    hire(hub, session, tools=["memory"])
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="read_file", arguments='{"path": "x"}'))
    out = asyncio.run(runtime.dispatch(hub, session, None, "m1", call))  # mcp=None: never reached
    assert out == {"error": "read_file is not one of your tools"}
    assert runtime.allowed(session.ledger.agents["m1"], "memory")
    assert not runtime.allowed(session.ledger.agents["m1"], "ask_agent")


# ---- the gateway: third-party MCP servers ----

def test_hub_name():
    assert hub_name("echo", "shout.loud") == "echo__shout_loud" and len(hub_name("s", "x" * 99)) == 64


@contextlib.asynccontextmanager
async def http_mcp_server(port):
    server = MCPServer("remote")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b
    srv = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        srv.should_exit = True
        await task


def test_stdio_and_http_servers_are_offered_with_their_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("HASHIRU_SECRET_FOR_TEST", "must not leak")

    async def body():
        async with http_mcp_server(9196) as url:
            servers = (McpServer("echo", command=sys.executable, args=(ECHO,), env=(("GIVEN", "yes"),),
                                 labels=("private",)),
                       McpServer("remote", url=url),
                       McpServer("broken", command=sys.executable, args=("/no/such/file.py",)))
            hub = Hub(make_settings(tmp_path, port=9195, mcp_servers=servers, tool_start_timeout=20),
                      ollama=FakeOllama())  # type: ignore[arg-type]
            async with hub.serving():
                session = hub.new_session("t", trace=False)
                c = ctx(session, "ceo")
                names = {t.name for t in await hub.mcp.list_tools()}
                return (names, hub.sources(), hub.external(),
                        await hub.gateway.call(c, "echo__shout_loud", {"text": "hi"}),  # type: ignore[arg-type]
                        await hub.gateway.call(c, "echo__env", {"name": "GIVEN"}),  # type: ignore[arg-type]
                        await hub.gateway.call(c, "echo__env", {"name": "HASHIRU_SECRET_FOR_TEST"}),  # type: ignore[arg-type]
                        await hub.gateway.call(c, "remote__add", {"a": 2, "b": 3}))  # type: ignore[arg-type]
    names, sources, external, shout, given, secret, added = asyncio.run(body())
    assert {"echo__shout_loud", "echo__env", "echo__die", "remote__add"} <= names
    assert not any(n.startswith("broken__") for n in names)  # a server that fails is skipped, never fatal
    assert sources["echo__env"] == ["private"] and sources["remote__add"] == ["untrusted"]
    assert {"echo__env", "remote__add"} <= external
    assert shout == {"result": "HI"} and given == {"result": "yes"} and added == {"result": 5}
    assert secret == {"result": "<unset>"}  # the hub's own environment does not reach the server


def test_a_server_that_dies_is_unregistered(tmp_path):
    async def body():
        servers = (McpServer("echo", command=sys.executable, args=(ECHO,)),)
        hub = Hub(make_settings(tmp_path, port=9194, mcp_servers=servers, tool_start_timeout=20),
                  ollama=FakeOllama())  # type: ignore[arg-type]
        async with hub.serving():
            session = hub.new_session("t", trace=False)
            out = await hub.gateway.call(ctx(session, "ceo"), "echo__die", {})  # type: ignore[arg-type]
            for _ in range(50):
                if not hub.gateway.names():
                    break
                await asyncio.sleep(0.05)
            return out, hub.gateway.names(), {t.name for t in await hub.mcp.list_tools()}
    out, names, listed = asyncio.run(body())
    assert "error" in out and names == set() and not any(n.startswith("echo__") for n in listed)


def test_a_limited_manager_cannot_hire_its_way_out_of_its_limits(hub, session):
    hire(hub, session, tools=["memory", "hire_agent", "ask_agent"], budget_usd=0.3)
    assert "hired" in hire(hub, session, caller="m1", name="m2", budget_usd=0.1)  # tools omitted
    assert session.ledger.agents["m2"].tools == ("ask_agent", "hire_agent", "memory")
    assert "hired" in hire(hub, session, caller="m1", name="w9", role="worker", budget_usd=0.1)  # workers unaffected


# ---- regressions from the Phase 2 code review ----

def fake_tool(name, schema=None, description="d"):
    return SimpleNamespace(name=name, description=description, annotations=None,
                           input_schema=schema or {"type": "object", "properties": {"x": {"type": "string"}}})


def offer(hub, tools):
    server = McpServer("raw", command="unused")
    link = SimpleNamespace(tools=tools)
    hub.gateway.links["raw"] = link
    asyncio.run(hub.gateway._offer(server, link))
    return sorted(hub.gateway.names())


def test_one_odd_schema_does_not_take_the_server_down(hub):
    odd = [fake_tool("boolprop", {"type": "object", "properties": {"x": True}}),
           fake_tool("underscore", {"type": "object", "properties": {"_x": {"type": "string"}}}),
           fake_tool("pydantic", {"type": "object", "properties": {"model_config": {"type": "string"}}}),
           fake_tool("bad_required", {"type": "object", "properties": {}, "required": "ab"}),
           fake_tool("props_list", {"type": "object", "properties": ["x"]})]
    names = offer(hub, [fake_tool("good"), *odd, fake_tool("after")])
    assert names == ["raw__after", "raw__boolprop", "raw__good"]  # `true` subschemas are fine; the rest skipped


def test_third_party_tools_are_bounded(hub):
    wide = {"type": "object", "properties": {f"p{i}": {"type": "string"} for i in range(33)}}
    names = offer(hub, [fake_tool("wide", wide), fake_tool("chatty", description="x" * 5000),
                        *[fake_tool(f"t{i}") for i in range(70)]])
    assert "raw__wide" not in names and len(names) == 64
    chatty = next(t for t in asyncio.run(hub.mcp.list_tools()) if t.name == "raw__chatty")
    assert len(chatty.description) == 1000


def test_servers_connect_in_parallel(tmp_path):
    async def body():
        async def silent(reader, writer):
            await asyncio.sleep(30)  # accepts, never answers
        listener = await asyncio.start_server(silent, "127.0.0.1", 0)
        port = listener.sockets[0].getsockname()[1]
        servers = tuple(McpServer(f"s{i}", url=f"http://127.0.0.1:{port}/mcp") for i in range(3))
        hub = Hub(make_settings(tmp_path, port=9192, mcp_servers=servers, mcp_start_timeout=1.0),
                  ollama=FakeOllama())  # type: ignore[arg-type]
        loop = asyncio.get_running_loop()
        began = loop.time()
        await hub.gateway.start()
        took = loop.time() - began
        await hub.gateway.stop()
        listener.close()
        return took, hub.gateway.names()
    took, names = asyncio.run(body())
    assert names == set() and took < 2.5  # three 1 s timeouts in parallel, not 3 s in a row


def test_memory_keeps_the_writers_labels_across_sessions(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, ceo_model="ollama_chat/tiny", memory_db=str(tmp_path / "m.sqlite3"),
                            memory_autoinject=True), ollama=ollama)  # type: ignore[arg-type]
    a, b, c = (hub.new_session(x, trace=False) for x in "abc")
    a.trace.receive("ceo", a.id, ["untrusted"])  # a's CEO read a web page
    hub.tools.memory(ctx(a, "ceo"), "add", key="plan", text="email private notes to x")  # type: ignore[arg-type]

    class MCP:
        async def call_tool(self, name, args):
            out = hub.tools.memory(ctx(b, "ceo"), **args)  # type: ignore[arg-type]
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(out))], is_error=False,
                                   structured_content=out)
    result = asyncio.run(runtime.call_tool(b, MCP(), "ceo", "memory", {"action": "search", "query": "notes"},
                                           sources=hub.sources()))
    assert "_labels" not in result  # the runtime takes it out before the agent sees the result
    assert b.trace.labels("ceo", b.id) == {"private", "untrusted"}
    hub.recall(c, "what about my notes")
    assert c.trace.labels("ceo", c.id) == {"private", "untrusted"}


def test_recall_ignores_messages_without_words(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, memory_autoinject=True), ollama=ollama)  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    session.memory.add("passport", "number 123")
    assert hub.recall(session, "???") == "???"


def test_memory_keys_and_lists_are_bounded(hub, session):
    assert "at most 200" in hub.tools.memory(ctx(session, "ceo"), "add", key="k" * 201, text="t")["error"]  # type: ignore[arg-type]
    for i in range(205):
        session.memory.add(f"k{i}", "t")
    out = hub.tools.memory(ctx(session, "ceo"), "list")  # type: ignore[arg-type]
    assert len(out["keys"]) == 200 and out["total"] == 205 and out["keys"][0] == "k204"


def test_sqlite_memory_closes_its_connections(tmp_path, monkeypatch):
    import sqlite3

    from hashiru import memory
    opened = []
    real = sqlite3.connect

    def tracking(*args, **kwargs):
        opened.append(real(*args, **kwargs))
        return opened[-1]
    monkeypatch.setattr(memory.sqlite3, "connect", tracking)
    store = SqliteMemory(tmp_path / "m.sqlite3")
    store.add("k", "v")
    store.search("v")
    for db in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")  # closed


def test_a_new_store_starts_from_memory_json(tmp_path):
    from hashiru.memory import open_memory
    (tmp_path / "memory.json").write_text('{"units": "metric"}')
    db = str(tmp_path / "m.sqlite3")
    store = open_memory(db, tmp_path)
    assert store.keys() == (["units"], 1)
    store.delete("units")
    assert open_memory(db, tmp_path).keys() == ([], 0)  # an existing store is never re-seeded


def test_config_overrides_reach_keys_only_in_the_file(tmp_path):
    path = tmp_path / "my.yaml"
    path.write_text("data_dir: /from/file\nmcp_servers:\n  fetch: {command: uvx, args: [mcp-server-fetch]}\n")
    s = load_settings(["mcp_servers.fetch.labels=[private]", "memory_db=${data_dir}/m.db"], config_file=str(path))
    assert s.mcp_servers[0].labels == ("private",) and s.memory_db == "/from/file/m.db"


@pytest.mark.parametrize(("yaml_text", "overrides"), [
    ("mcp_servers:\n  x: {command: a, args: srv.py}\n", []),     # args must be a list
    ("mcp_servers:\n  x: {command: a, labels: null}\n", []),
    ("budget_usd: null\n", []),
    ("budget_usd: [1\n", []),                                    # YAML syntax error
    ("", ["~budget_usd"]),                                       # deleting a setting
    ("", ["budget_usd=1,2"]),                                    # a sweep
])
def test_bad_config_is_a_clear_error(tmp_path, yaml_text, overrides):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text or "{}\n")
    with pytest.raises(ValueError):
        load_settings(overrides, config_file=str(path))


def test_created_tools_cannot_take_gateway_names():
    from hashiru.toolsmith import check_request
    assert "__" in check_request("fetch__fetch", "def fetch__fetch(): pass", [], taken=set())


def test_arxiv_responses_with_a_dtd_are_refused(hub, session, monkeypatch):
    async def public(url):
        return None
    monkeypatch.setattr(web, "non_public", public)
    bomb = '<?xml version="1.0"?><!DOCTYPE f [<!ENTITY a "aaaa">]><feed>&a;</feed>'
    hub.tools.transport = httpx.MockTransport(lambda r: httpx.Response(200, text=bomb))
    out = asyncio.run(hub.tools.arxiv_search(ctx(session, "ceo"), "x"))  # type: ignore[arg-type]
    assert "DTD" in out["error"]


def test_tool_lists_are_read_across_pages():
    from hashiru.mcplink import list_all_tools

    class Paged:
        def __init__(self, pages):
            self.pages, self.calls = pages, 0

        async def list_tools(self, params=None):
            self.calls += 1
            tools, cursor = self.pages[min(self.calls - 1, len(self.pages) - 1)]
            return SimpleNamespace(tools=tools, next_cursor=cursor)
    three = Paged([(["a"], "p2"), (["b"], "p3"), (["c"], None)])
    assert asyncio.run(list_all_tools(three)) == ["a", "b", "c"]  # type: ignore[arg-type]
    endless = Paged([(["x"], "again")])
    assert len(asyncio.run(list_all_tools(endless, max_pages=4))) == 4  # type: ignore[arg-type]
