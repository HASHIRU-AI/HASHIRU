"""Sessions on one hub: setup, isolation and teardown."""
import asyncio

import pytest
from conftest import FakeOllama, ctx, make_settings

import hashiru.hub as hub_module
from hashiru.hub import Hub
from hashiru.session import USER


def agent_paths(hub):
    return sorted(p for r in hub.app.router.routes if (p := getattr(r, "path", "")).startswith("/s/"))


def test_new_session_starts_with_a_ceo_holding_the_budget(hub, tmp_path):
    session = hub.new_session("ask")
    ceo = session.ledger.agents["ceo"]
    assert (ceo.role, ceo.depth, ceo.reserved, ceo.parent) == ("manager", 0, 0.5, None)
    assert set(session.tokens) == {USER, "ceo"}
    assert agent_paths(hub) == [f"/s/{session.id}/agents/ceo"]
    assert session.trace.path == tmp_path / "traces" / f"{session.id}.jsonl"
    assert session.trace.events[0]["kind"] == "start" and session.trace.events[0]["budget_usd"] == 0.5


def test_sessions_are_isolated(hub):
    a, b = hub.new_session("a", trace=False), hub.new_session("b", trace=False)
    args = {"name": "w1", "role": "worker", "model": "gpt-4.1-mini", "prompt": "p", "budget_usd": 0.2}
    for s in (a, b):  # the same agent name in two sessions
        assert "hired" in asyncio.run(hub.tools.hire_agent(ctx(s, "ceo"), **args))  # type: ignore[arg-type]
    assert a.ledger.agents["w1"] is not b.ledger.agents["w1"]
    assert hub.caller(ctx(a, "ceo").headers)[0] is a and hub.caller(ctx(b, "ceo").headers)[0] is b
    out = asyncio.run(hub.tools.check_usage(ctx(a, "ceo"), "ceo"))  # type: ignore[arg-type]
    assert out["api_usd"] == 0.3


def test_close_session_takes_everything_down(hub, session, ollama):
    args = {"name": "w1", "role": "worker", "model": "ollama_chat/tiny", "prompt": "p", "budget_usd": 0.1}
    asyncio.run(hub.tools.hire_agent(ctx(session, "ceo"), **args))  # type: ignore[arg-type]
    asyncio.run(hub.close_session(session))
    assert agent_paths(hub) == [] and session.id not in hub.sessions
    assert not any(a.live for a in session.ledger.agents.values())
    assert ollama.unloaded == ["ollama_chat/tiny"]


def test_unpriced_cloud_models_are_refused_at_startup(tmp_path):
    with pytest.raises(ValueError, match="no price"):
        Hub(make_settings(tmp_path, models=("made-up-provider/model-x",)))


def test_session_ids_never_collide(hub, monkeypatch):
    draws = iter(["aaaaaaaaaaaa", "aaaaaaaaaaaa", "bbbbbbbbbbbb"])
    monkeypatch.setattr("hashiru.hub.secrets.token_hex", lambda n: next(draws))
    monkeypatch.setattr("hashiru.hub.time.strftime", lambda fmt: "20261029-000000")
    a, b = hub.new_session("chat", trace=False), hub.new_session("chat", trace=False)
    assert a.id != b.id and hub.sessions[a.id] is a and hub.sessions[b.id] is b


@pytest.mark.parametrize("label", ["../x", "a/b", "{sid}", "Chat", ""])
def test_session_labels_are_validated(hub, label):
    with pytest.raises(ValueError):
        hub.new_session(label, trace=False)


def test_unload_rechecks_each_model_right_before_unloading(hub, session, ollama):
    other = hub.new_session("other", trace=False)

    async def hire_y_meanwhile(model):
        if model == "ollama_chat/a":  # while a is unloading, another session starts using b
            other.ledger.add("w9", "worker", "local", "ollama_chat/b", "p", "ceo", 1, 0.0)
    ollama.on_unload = hire_y_meanwhile
    asyncio.run(hub.release_models({"ollama_chat/a", "ollama_chat/b"}))
    assert ollama.unloaded == ["ollama_chat/a"]


def test_a_stuck_random_source_raises_instead_of_hanging(hub, monkeypatch):
    monkeypatch.setattr("hashiru.hub.secrets.token_hex", lambda n: "beef")
    monkeypatch.setattr("hashiru.hub.time.strftime", lambda fmt: "20261029-000000")
    hub.new_session("chat", trace=False)
    with pytest.raises(RuntimeError, match="unused session id"):
        hub.new_session("chat", trace=False)


def test_a_busy_port_is_a_clear_error(tmp_path):
    import socket
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        hub = Hub(make_settings(tmp_path, port=port))

        async def go():
            async with hub.serving():
                pass
        with pytest.raises(RuntimeError, match="cannot listen"):
            asyncio.run(go())


class SizedOllama(FakeOllama):
    async def sizes(self) -> dict[str, int]:
        return {"tiny": 50, "other": 50}  # each loads as 100 MB (FakeOllama.load)


def test_memory_cap_refuses_a_second_model_that_would_pass_it(tmp_path, monkeypatch):
    from test_tools import hire
    monkeypatch.setattr(hub_module, "free_mb", lambda: 10_000)
    models = ("gpt-4.1-mini", "ollama_chat/tiny", "ollama_chat/other")
    hub = Hub(make_settings(tmp_path, models=models, memory_mb=120), ollama=SizedOllama())  # type: ignore[arg-type]
    session = hub.new_session("t", trace=False)
    assert "hired" in hire(hub, session, name="w1", model="ollama_chat/tiny", budget_usd=0)
    assert hub.free_memory() == 20  # 120 MB cap, 100 MB held by tiny
    refused = hire(hub, session, name="w2", model="ollama_chat/other", budget_usd=0)
    assert "needs 50 MB" in refused["error"] and "20 MB" in refused["error"]
    assert "hired" in hire(hub, session, name="w3", model="ollama_chat/tiny", budget_usd=0)  # already loaded


def test_no_cap_keeps_measured_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(hub_module, "free_mb", lambda: 777)
    hub = Hub(make_settings(tmp_path), ollama=SizedOllama())  # type: ignore[arg-type]
    assert hub.free_memory() == 777
    capped = Hub(make_settings(tmp_path, memory_mb=500), ollama=SizedOllama())  # type: ignore[arg-type]
    assert capped.free_memory() == 500


def test_hubs_can_share_one_wallet(tmp_path):
    from hashiru.llm import Wallet
    wallet = Wallet(1.0)
    one = Hub(make_settings(tmp_path / "a"), ollama=FakeOllama(), wallet=wallet)  # type: ignore[arg-type]
    two = Hub(make_settings(tmp_path / "b"), ollama=FakeOllama(), wallet=wallet)  # type: ignore[arg-type]
    assert one.wallet is two.wallet is wallet and one.llm.wallet is wallet and two.llm.wallet is wallet
