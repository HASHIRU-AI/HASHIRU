"""The built-in MCP tools, called directly with a fake request context."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import FakeOllama, ctx

from hashiru.hub import Hub
from hashiru.session import Session


def hire(hub: Hub, session: Session, caller: str = "ceo", **kw: object) -> dict:
    args = {"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "prompt": "p", "budget_usd": 0.2} | kw
    return asyncio.run(hub.tools.hire_agent(ctx(session, caller), **args))  # type: ignore[arg-type]


def fire(hub: Hub, session: Session, caller: str, name: str) -> dict:
    return asyncio.run(hub.tools.fire_agent(ctx(session, caller), name))  # type: ignore[arg-type]


def test_hire_reserves_budget_from_hirer(hub, session):
    assert hire(hub, session) == {"hired": "w1", "tier": "cloud", "depth": 1, "reserved_usd": 0.2, "gpu_mb": 0}
    assert session.ledger.agents["w1"].parent == "ceo"
    assert round(session.ledger.left()["ceo"], 6) == 0.3  # 0.50 budget - 0.20 reserved


@pytest.mark.parametrize("bad", [{"model": "nope"}, {"role": "boss"}, {"name": "Bad Name"}, {"name": "ceo"},
                                 {"budget_usd": -0.5}, {"budget_usd": float("inf")}, {"budget_usd": float("nan")}])
def test_hire_rejects_structural_errors(hub, session, bad):
    assert "error" in hire(hub, session, **bad)


def test_depth_cap(hub, session):
    hire(hub, session, name="m1", role="manager")
    assert hire(hub, session, caller="m1", name="m2", role="manager")["depth"] == 2
    assert "error" in hire(hub, session, caller="m2", name="w3")


def test_only_managers_get_tokens_and_hire(hub, session):
    hire(hub, session, name="w1")
    assert "w1" not in session.tokens               # workers never call tools, so they get no token
    hub.issue(session, "w1")                        # even with one, a worker cannot hire
    assert hire(hub, session, caller="w1", name="w2") == {"error": "only managers can hire"}


def test_unknown_token_is_rejected(hub, session):
    stranger = SimpleNamespace(headers={"authorization": "Bearer not-a-token"})
    with pytest.raises(PermissionError):
        asyncio.run(hub.tools.check_usage(stranger, "ceo"))  # type: ignore[arg-type]


def test_concurrent_hires_with_same_name_only_one_wins(hub, session):
    hub.ollama = FakeOllama(delay=0.05)

    async def both():
        args = {"name": "w1", "role": "worker", "model": "ollama_chat/tiny", "prompt": "p", "budget_usd": 0.1}
        c = ctx(session, "ceo")
        return await asyncio.gather(hub.tools.hire_agent(c, **args), hub.tools.hire_agent(c, **args))

    results = asyncio.run(both())
    assert sorted("hired" in r for r in results) == [False, True]
    assert session.ledger.agents["w1"].gpu_mb == 100


def test_failed_model_load_releases_the_name(hub, session):
    hub.ollama = FakeOllama(fail=httpx.ConnectError("ollama is not running"))
    assert "error" in hire(hub, session, model="ollama_chat/tiny")
    assert session.ledger.get("w1") is None


def test_fire_takes_the_whole_subtree_and_unloads_unused_models(hub, session, ollama):
    hire(hub, session, name="m1", role="manager", budget_usd=0.3)
    hire(hub, session, caller="m1", name="w2", model="ollama_chat/tiny", budget_usd=0.1)
    assert fire(hub, session, "ceo", "m1") == {"fired": ["m1", "w2"]}
    assert not session.ledger.agents["m1"].live and not session.ledger.agents["w2"].live
    assert round(session.ledger.left()["ceo"], 6) == 0.5  # nothing was spent, so everything returns
    assert ollama.unloaded == ["ollama_chat/tiny"]
    paths = [getattr(r, "path", "") for r in hub.app.router.routes]
    assert not any(p.endswith(("/agents/m1", "/agents/w2")) for p in paths)
    assert "error" in fire(hub, session, "ceo", "ceo")


def test_a_model_shared_with_another_session_stays_loaded(hub, session, ollama):
    other = hub.new_session("other", trace=False)
    hire(hub, session, model="ollama_chat/tiny")
    hire(hub, other, model="ollama_chat/tiny")
    fire(hub, session, "ceo", "w1")
    assert ollama.unloaded == []


def test_only_an_ancestor_can_fire(hub, session):
    hire(hub, session, name="m1", role="manager", budget_usd=0.3)
    hire(hub, session, caller="m1", name="m2", role="manager", budget_usd=0.1)
    for caller, name in [("m2", "m1"), ("m1", "m1"), ("m2", "m2")]:
        assert "error" in fire(hub, session, caller, name)
    assert session.ledger.agents["m1"].live and session.ledger.agents["m2"].live
    assert fire(hub, session, "m1", "m2") == {"fired": ["m2"]}


def test_a_fired_agent_cannot_hire(hub, session):
    hire(hub, session, name="m1", role="manager")
    stale = ctx(session, "m1")
    fire(hub, session, "ceo", "m1")
    args = {"name": "w2", "role": "worker", "model": "gpt-4.1-mini", "prompt": "p", "budget_usd": 0.1}
    with pytest.raises(PermissionError):
        asyncio.run(hub.tools.hire_agent(stale, **args))  # type: ignore[arg-type]
    assert session.ledger.get("w2") is None


def test_check_usage_reports_money_and_memory(hub, session):
    out = asyncio.run(hub.tools.check_usage(ctx(session, "ceo"), "ceo"))  # type: ignore[arg-type]
    assert out["api_usd"] == 0.5 and out["free_mb"] > 0 and out["model_mb"] == {"tiny": 50}
    assert "error" in asyncio.run(hub.tools.check_usage(ctx(session, "ceo"), "nobody"))  # type: ignore[arg-type]


def test_read_file_stays_inside_data(hub, session, tmp_path):
    data = tmp_path / "data"
    (data / "private").mkdir(parents=True)
    (data / "private" / "notes.md").write_text("secret")
    (tmp_path / "outside.txt").write_text("nope")
    c = ctx(session, "ceo")
    assert hub.tools.read_file(c, "private/notes.md")["text"] == "secret"  # type: ignore[arg-type]
    for bad in ["../outside.txt", str(tmp_path / "outside.txt"), "private/../../outside.txt", "missing.md"]:
        assert "error" in hub.tools.read_file(c, bad)  # type: ignore[arg-type]


def test_session_memory_is_per_session_and_never_written_back(hub, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "memory.json").write_text('{"units": "prefers metric units"}')
    s1, s2 = hub.new_session("s1", trace=False), hub.new_session("s2", trace=False)
    mem = hub.tools.memory
    assert mem(ctx(s1, "ceo"), "add", key="k", text="likes tea") == {"saved": "k"}  # type: ignore[arg-type]
    assert mem(ctx(s1, "ceo"), "list") == {"keys": ["k", "units"], "total": 2}  # type: ignore[arg-type]
    assert mem(ctx(s1, "ceo"), "search", query="what units") == {  # type: ignore[arg-type]
        "memories": [{"key": "units", "text": "prefers metric units"}], "_labels": []}
    assert mem(ctx(s2, "ceo"), "list") == {"keys": ["units"], "total": 1}  # type: ignore[arg-type]
    assert json.loads((data / "memory.json").read_text()) == {"units": "prefers metric units"}


def test_get_website_rejects_non_http(hub, session):
    out = asyncio.run(hub.tools.get_website(ctx(session, "ceo"), "file:///etc/passwd"))  # type: ignore[arg-type]
    assert "error" in out


# ---- regressions from the Phase 1 code review ----

def test_reserved_names_cannot_be_hired(hub, session):
    for reserved in ("user", "ceo"):
        assert "error" in hire(hub, session, name=reserved, role="manager")
    assert hub.identity(session.tokens["user"])[1] == "user"


def test_fired_agents_lose_their_token_and_tools(hub, session):
    hire(hub, session, name="m1", role="manager")
    token = session.tokens["m1"]
    fire(hub, session, "ceo", "m1")
    assert hub.identity(token) is None and hub.asgi.check("/mcp", token)[0] == 401
    stale = SimpleNamespace(headers={"authorization": f"Bearer {token}"})
    with pytest.raises(PermissionError):
        hub.tools.deliver_to_user(stale, "still here")  # type: ignore[arg-type]


def test_a_dead_caller_cannot_use_tools(hub, session):
    hire(hub, session, name="m1", role="manager")
    session.ledger.agents["m1"].live = False
    with pytest.raises(PermissionError):
        hub.tools.memory(ctx(session, "m1"), "get")  # type: ignore[arg-type]


def test_fire_while_a_local_model_loads_does_not_resurrect_the_hire(hub, session):
    hub.ollama = loading = FakeOllama(delay=0.05)
    hire(hub, session, name="m1", role="manager", budget_usd=0.3)

    async def race():
        args = {"name": "w2", "role": "worker", "model": "ollama_chat/tiny", "prompt": "p", "budget_usd": 0.1}
        hiring = asyncio.create_task(hub.tools.hire_agent(ctx(session, "m1"), **args))
        await asyncio.sleep(0.01)  # w2 is claimed and its model is loading
        fired = await hub.tools.fire_agent(ctx(session, "ceo"), "m1")
        return fired, await hiring

    fired, hired = asyncio.run(race())
    assert fired == {"fired": ["m1", "w2"]} and "error" in hired
    assert not session.ledger.agents["w2"].live
    assert not any(getattr(r, "path", "").endswith("/agents/w2") for r in hub.app.router.routes)
    assert loading.unloaded == ["ollama_chat/tiny"]


def test_close_while_a_local_model_loads_leaks_nothing(hub, session):
    hub.ollama = FakeOllama(delay=0.05)

    async def race():
        args = {"name": "m2", "role": "manager", "model": "ollama_chat/tiny", "prompt": "p", "budget_usd": 0.1}
        hiring = asyncio.create_task(hub.tools.hire_agent(ctx(session, "ceo"), **args))
        await asyncio.sleep(0.01)
        await hub.close_session(session)
        return await hiring

    assert "error" in asyncio.run(race())
    assert not [r for r in hub.app.router.routes if getattr(r, "path", "").startswith("/s/")]
    assert hub._tokens == {} and "m2" not in session.tokens


@pytest.mark.parametrize("failure", [KeyError("models"), ValueError("bad json")])
def test_unexpected_load_failures_become_errors_without_a_phantom(hub, session, failure):
    hub.ollama = FakeOllama(fail=failure)
    assert "error" in hire(hub, session, model="ollama_chat/tiny")
    assert session.ledger.get("w1") is None and not hub.in_use("ollama_chat/tiny")


def test_a_crashing_load_releases_the_name_before_propagating(hub, session):
    hub.ollama = FakeOllama(fail=RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        hire(hub, session, model="ollama_chat/tiny")
    assert session.ledger.get("w1") is None


@pytest.mark.parametrize("url", ["http://127.0.0.1:11434/api/ps", "http://localhost/", "http://10.0.0.5/",
                                 "http://169.254.169.254/latest/meta-data/", "http://[::1]/", "http://0.0.0.0/"])
def test_get_website_refuses_private_addresses(hub, session, url):
    out = asyncio.run(hub.tools.get_website(ctx(session, "ceo"), url))  # type: ignore[arg-type]
    assert "error" in out and "public" in out["error"]


def test_get_website_checks_every_redirect_and_caps_the_body(hub, session):
    def handler(request):
        if request.url.path == "/hop":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:11434/api/ps"})
        return httpx.Response(200, text="<p>" + "x" * 5_000_000 + "</p>")
    hub.tools.transport = httpx.MockTransport(handler)
    c = ctx(session, "ceo")
    hop = asyncio.run(hub.tools.get_website(c, "http://93.184.215.14/hop"))  # type: ignore[arg-type]
    assert "error" in hop and "public" in hop["error"]
    big = asyncio.run(hub.tools.get_website(c, "http://93.184.215.14/big"))  # type: ignore[arg-type]
    assert big["status"] == 200 and len(big["text"]) == 20000
