"""The examples stay runnable: the MCP server and the A2A agent in examples/, hired through a hub, and one hub hiring
another through its public A2A agent (scripted LLMs, no tokens spent)."""
import asyncio
import contextlib
import importlib.metadata
import importlib.util
import re
import sys
import tomllib
import zoneinfo
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import uvicorn
from conftest import FakeOllama, ctx, make_settings

from hashiru import runtime
from hashiru.config import A2APeer, McpServer, load_settings
from hashiru.hub import Hub
from hashiru.peers import fetch_card, price_of

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def example(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"example_{name}", EXAMPLES / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoLLM:
    """For a hub whose agents should never think: every hire here is a remote agent."""

    async def chat(self, model, messages, tools=None):
        raise AssertionError("no LLM call expected")


class HeardLLM:
    """A CEO that repeats the newest message it was given (a continued conversation also shows it the earlier ones)."""

    async def chat(self, model, messages, tools=None):
        said = messages[1]["content"].split("The user's new message:")[-1].strip()
        return SimpleNamespace(content=f"team b heard: {said}", tool_calls=None), 0.002


@contextlib.asynccontextmanager
async def served(app, port):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           timeout_graceful_shutdown=1))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        await task


def test_one_hub_hires_another_through_its_public_a2a_agent(tmp_path):
    team_b = Hub(make_settings(tmp_path / "b", port=9340, budget_usd=0.05), llm=HeardLLM(),  # type: ignore[arg-type]
                 ollama=FakeOllama())  # type: ignore[arg-type]
    peers = (A2APeer("team_b", "http://127.0.0.1:9340/a2a"), A2APeer("slash", "http://127.0.0.1:9340/a2a/"))
    hub = Hub(make_settings(tmp_path / "a", port=9341, a2a_peers=peers), llm=NoLLM(),  # type: ignore[arg-type]
              ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with team_b.serving(), hub.serving():
            session = hub.new_session("t", trace=False)
            hired = await hub.tools.hire_agent(ctx(session, "ceo"), "b", "worker", "a2a/team_b", "",  # type: ignore[arg-type]
                                               0.2)
            first = await runtime.ask_agent(hub, session, "ceo", "b", "hello")
            second = await runtime.ask_agent(hub, session, "ceo", "b", "again")
            await hub.tools.hire_agent(ctx(session, "ceo"), "s", "worker", "a2a/slash", "", 0.05)  # type: ignore[arg-type]
            slash = await runtime.ask_agent(hub, session, "ceo", "s", "hi")
            return (hired, first, second, slash, len(team_b.conversations.open_),
                    session.ledger.agents["b"].spent)
    hired, first, second, slash, open_on_b, spent = asyncio.run(body())
    assert hired["price_usd_per_message"] == 0.05             # team B's card: a message may cost its whole budget
    assert (first, second) == ("team b heard: hello", "team b heard: again")
    assert slash == "team b heard: hi"                        # <base>/a2a/ works too
    assert open_on_b == 2                                     # one conversation per hire on team B
    assert spent == pytest.approx(0.1)                        # the declared price, twice, whatever B spent


# ---- examples/clock_mcp_server.py ----

def test_the_example_mcp_server_gives_agents_the_date(tmp_path):
    server = McpServer("clock", command=sys.executable, args=(str(EXAMPLES / "clock_mcp_server.py"),))
    hub = Hub(make_settings(tmp_path, port=9342, mcp_servers=(server,), mcp_start_timeout=20),
              ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with hub.serving():
            c = ctx(hub.new_session("t", trace=False), "ceo")
            return ({t.name for t in await hub.mcp.list_tools()},
                    await hub.gateway.call(c, "clock__days_between", {"start": "2026-01-01", "end": "2026-10-29"}),  # type: ignore[arg-type]
                    await hub.gateway.call(c, "clock__now", {}),  # type: ignore[arg-type]
                    await hub.gateway.call(c, "clock__now", {"timezone": "Asia/Tokyo"}),  # type: ignore[arg-type]
                    await hub.gateway.call(c, "clock__now", {"timezone": "../../etc/passwd"}),  # type: ignore[arg-type]
                    await hub.gateway.call(c, "clock__days_between", {"start": "today", "end": "2026-10-29"}))  # type: ignore[arg-type]
    names, days, utc, tokyo, bad_zone, bad_date = asyncio.run(body())
    assert {"clock__now", "clock__days_between"} <= names
    assert days == {"result": 301}
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00 \(\w+day\)", utc["result"])
    if "Asia/Tokyo" in zoneinfo.available_timezones():  # a host without a time zone database knows only UTC
        assert tokyo["result"].split(" ")[0].endswith("+09:00")
    assert "unknown time zone" in bad_zone["error"]          # a ToolError's text reaches the agent
    assert "use ISO dates" in bad_date["error"]


# ---- examples/calc_a2a_agent.py ----

HELP = "error: send one arithmetic expression, such as (17.5 / 100) * 2340"
TOO_LARGE = "error: the result would be too large"


@pytest.mark.parametrize(("text", "answer"), [
    ("What is (17.5 / 100) * 2340?", "(17.5 / 100) * 2340 = 409.5"),
    ("Compute exactly.\n\nWhat is 2 ** 10 - 3 * (4 + 1)?", "2 ** 10 - 3 * (4 + 1) = 1009"),  # a hire's prompt first
    ("-7 // 2 and 7 % 3.", "7 % 3 = 1"),                       # the last expression; a full stop is not a decimal
    ("-7 // 2", "-7 // 2 = -4"),
    ("0.1 + 0.2", "0.1 + 0.2 = 0.3"),                          # exact, not binary floating point
    ("12345678901234567890.5 + 1", "12345678901234567890.5 + 1 = 12345678901234567891.5"),
    ("1 / 10 ** 400", "1 / 10 ** 400 = 1e-400"),
    ("1e-400 * 10 ** 400", "1e-400 * 10 ** 400 = 1"),
    ("1 / 3", "1 / 3 ≈ 0.333333333333"),
    ("1e3 + 1", "1e3 + 1 = 1001"),
    ("1 / 0", "error: division by zero"),
    ("2 ** 0.5", "error: only whole-number powers"),
    ("9 ** 9 ** 9", TOO_LARGE),
    ("(1 + 1 / 10 ** 20) ** 10 ** 12", TOO_LARGE),              # a base near 1 still has a large numerator
    ("1e1001 + 1", TOO_LARGE),
    ("2 ** 4000 * 2 ** 4000 * 2 ** 4000 * 2 ** 4000", TOO_LARGE),
    ("How many days are left until 2026-12-25?", HELP),          # dates and names are not subtractions
    ("Is it ISO-8601?", HELP),
    ("hello there", HELP),
])
def test_the_example_a2a_agent_does_exact_arithmetic(text, answer):
    assert example("calc_a2a_agent").calculate(text) == answer


def test_the_example_a2a_agent_is_hired_and_paid_like_any_peer(tmp_path):
    calc = example("calc_a2a_agent")
    hub = Hub(make_settings(tmp_path, port=9344, a2a_peers=(A2APeer("calc", "http://127.0.0.1:9343/"),)),
              llm=NoLLM(), ollama=FakeOllama())  # type: ignore[arg-type]

    async def body():
        async with served(calc.build_app("http://127.0.0.1:9343/", 0.001), 9343), hub.serving():
            card = await fetch_card("http://127.0.0.1:9343/", {}, 5)
            session = hub.new_session("t", trace=False)
            await hub.tools.hire_agent(ctx(session, "ceo"), "calc", "worker", "a2a/calc", "", 0.01)  # type: ignore[arg-type]
            answer = await runtime.ask_agent(hub, session, "ceo", "calc", "What is (17.5 / 100) * 2340?")
            return card, answer, session.ledger.agents["calc"].spent
    card, answer, spent = asyncio.run(body())
    assert price_of(card).per_message_usd == 0.001  # type: ignore[union-attr]
    assert answer == "(17.5 / 100) * 2340 = 409.5" and spent == pytest.approx(0.001)


def test_the_example_a2a_agent_reads_only_the_end_of_a_long_message():
    calc = example("calc_a2a_agent")
    assert calc.calculate("1 + 1 " + "x" * 10_000 + " 2 * 3") == "2 * 3 = 6"
    assert calc.calculate("1 + 1 " + "x" * 10_000) == HELP


@pytest.mark.parametrize("name", ["clock_mcp_server", "calc_a2a_agent"])
def test_the_examples_pin_the_versions_hashiru_is_tested_with(name):
    """The tests run the examples with this project's packages, so their own pins must name the same versions."""
    block = re.search(r"^# /// script\n(.*?)^# ///$", (EXAMPLES / f"{name}.py").read_text(), re.S | re.M)
    assert block
    script = tomllib.loads("\n".join(line.removeprefix("#").removeprefix(" ") for line in block[1].splitlines()))
    for requirement in script["dependencies"]:
        package, _, pinned = re.sub(r"\[.*\]", "", requirement).partition("==")
        assert pinned, f"{requirement} is not pinned"
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):  # e.g. tzdata, where the host has a database
            assert importlib.metadata.version(package) == pinned, requirement


def test_the_example_config_wires_the_examples():
    settings = load_settings([], config_file=str(EXAMPLES / "hashiru.yaml"))
    assert {"clock", "fetch"} <= {s.name for s in settings.mcp_servers}
    assert [p.name for p in settings.a2a_peers] == ["calc"]
