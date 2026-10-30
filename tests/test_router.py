"""Phase 8b: hiring by difficulty. The hiring agent says how hard the job is; the router turns that into an ordered
list of models within the hire's privacy, the modes and the pool; a worker that fails moves up its list (fake LLMs,
no network)."""
import asyncio

import pytest
from conftest import FakeOllama, ctx, make_settings

from hashiru import hub as hub_module
from hashiru import runtime
from hashiru.config import load_settings
from hashiru.hub import Hub
from hashiru.interop.activity import describe
from hashiru.trace import PRIVATE

MODELS = ("gpt-4.1-mini", "gpt-4.1", "ollama_chat/tiny", "ollama_chat/big")


class SizedOllama(FakeOllama):
    """tiny is 50 MB and big 500 MB; loads report those sizes."""

    async def sizes(self):
        await asyncio.sleep(0)
        return {"tiny": 50, "big": 500}

    async def load(self, model):
        self.loaded.append(model)
        return {"ollama_chat/tiny": 50, "ollama_chat/big": 500}[model]


def make_hub(tmp_path, **settings):
    return Hub(make_settings(tmp_path, models=MODELS, **settings), ollama=SizedOllama())  # type: ignore[arg-type]


def routes(hub, session, boss="ceo", budget=0.1):
    return {d: asyncio.run(hub.router.candidates(session, session.ledger.agents[boss], d, budget))
            for d in ("easy", "medium", "hard")}


# ---- the router ----

def test_default_routes_order_models_by_size_and_price(tmp_path):
    hub = make_hub(tmp_path)
    assert routes(hub, hub.new_session("t", trace=False)) == {
        "easy": ["ollama_chat/tiny", "ollama_chat/big", "gpt-4.1-mini", "gpt-4.1"],
        "medium": ["ollama_chat/big", "ollama_chat/tiny", "gpt-4.1-mini", "gpt-4.1"],
        "hard": ["gpt-4.1", "gpt-4.1-mini", "ollama_chat/big", "ollama_chat/tiny"]}


def test_configured_routes_and_modes_win(tmp_path):
    hub = make_hub(tmp_path, routing=(("hard", ("gpt-4.1-mini", "ollama_chat/big")),), allow_local_agents=False)
    found = routes(hub, hub.new_session("t", trace=False))
    assert found["hard"] == ["gpt-4.1-mini"]                      # the table, minus what the modes forbid
    assert found["easy"] == ["gpt-4.1-mini", "gpt-4.1"]           # no table entry: the default order


def test_privacy_and_an_empty_pool_shape_the_list(tmp_path):
    strict = make_hub(tmp_path, private_mode="block")
    session = strict.new_session("t", trace=False)
    session.trace.receive("ceo", session.id, [PRIVATE])
    assert routes(strict, session)["hard"] == ["ollama_chat/big", "ollama_chat/tiny"]  # private: local only
    hub = make_hub(tmp_path)
    s2 = hub.new_session("t", trace=False)
    s2.trace.receive("ceo", s2.id, [PRIVATE])
    assert routes(hub, s2)["hard"] == ["ollama_chat/big", "ollama_chat/tiny", "gpt-4.1", "gpt-4.1-mini"]  # local first
    assert routes(hub, hub.new_session("t", trace=False), budget=0.0)["hard"] == ["ollama_chat/big", "ollama_chat/tiny"]


# ---- hiring by difficulty ----

def hire(hub, session, **kw):
    args = {"name": "w1", "role": "worker", "prompt": "p", "budget_usd": 0.1} | kw
    return asyncio.run(hub.tools.hire_agent(ctx(session, "ceo"), **args))  # type: ignore[arg-type]


def test_a_hire_by_difficulty_gets_the_first_model_that_fits(tmp_path, monkeypatch):
    hub = make_hub(tmp_path)
    session = hub.new_session("t", trace=False)
    out = hire(hub, session, difficulty="easy")
    assert out["hired"] == "w1" and out["model"] == "ollama_chat/tiny" and out["difficulty"] == "easy"
    assert session.ledger.agents["w1"].route == ("ollama_chat/tiny", "ollama_chat/big", "gpt-4.1-mini", "gpt-4.1")
    monkeypatch.setattr(hub_module, "free_mb", lambda: 100)       # big (500 MB) no longer fits
    assert hire(hub, session, name="w2", difficulty="medium")["model"] == "ollama_chat/tiny"  # in use: fits


def test_a_hire_needs_a_difficulty_or_a_model(tmp_path):
    hub = make_hub(tmp_path)
    session = hub.new_session("t", trace=False)
    assert "difficulty" in hire(hub, session)["error"]
    assert "difficulty" in hire(hub, session, difficulty="extreme")["error"]
    assert hire(hub, session, model="gpt-4.1")["model"] == "gpt-4.1"  # naming a model still works


# ---- escalation ----

def worker_replies(monkeypatch, session, answers):
    """The worker answers according to the model it runs on; record what each model was asked."""
    asked = []

    async def fake_send(hub_, session_, me, name, text):
        model = session.ledger.agents[name].model
        asked.append(model)
        return answers[model]
    monkeypatch.setattr(runtime, "a2a_send", fake_send)
    return asked


def test_a_failing_worker_moves_up_its_list(tmp_path, monkeypatch):
    hub = make_hub(tmp_path)
    session = hub.new_session("t", trace=False)
    hire(hub, session, difficulty="easy")
    asked = worker_replies(monkeypatch, session, {"ollama_chat/tiny": "failed: w1 gave an empty answer",
                                                  "ollama_chat/big": "the answer"})
    assert asyncio.run(runtime.ask_agent(hub, session, "ceo", "w1", "solve it")) == "the answer"
    assert asked == ["ollama_chat/tiny", "ollama_chat/big"] and session.ledger.agents["w1"].model == "ollama_chat/big"
    event = next(e for e in session.trace.events if e["kind"] == "escalate")
    assert (event["from_model"], event["to_model"]) == ("ollama_chat/tiny", "ollama_chat/big")
    assert describe(event).startswith("w1 moved from ollama_chat/tiny to ollama_chat/big")


def test_escalation_stops_at_its_limit_and_never_takes_private_data_to_the_cloud(tmp_path, monkeypatch):
    hub = make_hub(tmp_path, max_escalations=1)
    session = hub.new_session("t", trace=False)
    hire(hub, session, difficulty="easy")
    asked = worker_replies(monkeypatch, session, dict.fromkeys(MODELS, "failed: no"))
    assert asyncio.run(runtime.ask_agent(hub, session, "ceo", "w1", "solve it")) == "failed: no"
    assert asked == ["ollama_chat/tiny", "ollama_chat/big"]      # one step, then the failure is the answer
    private = make_hub(tmp_path / "p")
    s2 = private.new_session("t", trace=False)
    hire(private, s2, difficulty="easy")
    s2.trace.receive("w1", s2.id, [PRIVATE])                      # it read a private file
    asked = worker_replies(monkeypatch, s2, dict.fromkeys(MODELS, "failed: no"))
    asyncio.run(runtime.ask_agent(private, s2, "ceo", "w1", "solve it"))
    assert asked == ["ollama_chat/tiny", "ollama_chat/big"]      # big, then no cloud model


def test_routing_settings():
    s = load_settings(["+routing={hard: [gpt-4.1-mini]}", "models=[gpt-4.1-mini]", "max_escalations=3"])
    assert s.routing == (("hard", ("gpt-4.1-mini",)),) and s.max_escalations == 3
    for bad, match in (("+routing={extreme: [gpt-4.1-mini]}", "routing"), ("+routing={hard: [nope]}", "routing"),
                       ("max_escalations=-1", "max_escalations")):
        with pytest.raises(ValueError, match=match):
            load_settings([bad])
