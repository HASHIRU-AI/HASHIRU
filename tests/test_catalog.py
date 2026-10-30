"""Model recon: which models this machine can use (local from Ollama, cloud from LiteLLM's map where a key is set),
how they are classified and picked, and models=auto (fakes only: no Ollama, no provider, no download)."""
import asyncio
import json
import socket

import httpx
import pytest
from conftest import make_settings

from hashiru import recon
from hashiru.catalog import Card, base, cloud_cards, cloud_tier, local_cards, local_tier, picks, routing
from hashiru.config import load_settings
from hashiru.hub import Hub

TODAY = "2026-10-30"


def entry(out, provider="openai", mode="chat", tools=True, reasoning=False, dep=None, ctx=128000, inp=None):
    return {"litellm_provider": provider, "mode": mode, "output_cost_per_token": out / 1e6,
            "input_cost_per_token": (out / 4 if inp is None else inp) / 1e6, "supports_function_calling": tools,
            "supports_reasoning": reasoning, "max_input_tokens": ctx} | ({"deprecation_date": dep} if dep else {})


MAP = {
    "sample_spec": {"mode": "chat", "litellm_provider": "openai"},  # the real map's documentation entry
    "gpt-cheap": entry(1.6),
    "gpt-mid": entry(8, reasoning=True),
    "gpt-big": entry(15, reasoning=True),
    "gpt-big-2025-08-07": entry(15, reasoning=True),   # a snapshot of gpt-big: the alias is enough
    "gpt-old-2024-05-13": entry(5),                    # a snapshot with no alias: kept
    "text-embedding-x": entry(0.1, mode="embedding"),
    "gpt-notools": entry(1, tools=False),
    "gpt-free": entry(0, inp=0),
    "gpt-retired": entry(2, dep="2026-01-01"),
    "gpt-retiring": entry(2, dep="2027-01-01"),
    "ft:gpt-cheap-2024": entry(3),
    "gpt-audio-mini": entry(2),
    "gpt-4o-search-preview": entry(10),
    "chat-latest": entry(30),
    "gpt-x-exp-0827": entry(4),
    "claude-mid": entry(5, provider="anthropic", reasoning=True),
    "groq/llama": entry(0.8, provider="groq"),          # no groq key
    "not-a-dict": "x",
}


def keyed(model):
    return MAP[model]["litellm_provider"] in {"openai", "anthropic"}


def card(model, tier, out=0.0, reasoning=False, size=None, provider="openai"):
    local = model.startswith("ollama_chat/")
    return Card(model, "ollama" if local else provider, local, True, reasoning, False, None, out / 4, out, size, "",
                tier)


# ---- classifying ----

def test_cloud_cards_keep_only_usable_chat_models():
    cards = {c.model: c for c in cloud_cards(MAP, keyed, TODAY)}
    assert set(cards) == {"gpt-cheap", "gpt-mid", "gpt-big", "gpt-old-2024-05-13", "gpt-retiring", "claude-mid"}
    mid = cards["gpt-mid"]
    assert (mid.provider, mid.local, mid.tools, mid.reasoning, mid.context, mid.in_usd, mid.out_usd, mid.tier) == (
        "openai", False, True, True, 128000, 2.0, 8.0, "medium")
    assert cards["gpt-retiring"].deprecated == "2027-01-01" and cards["gpt-cheap"].deprecated == ""


def test_tiers_follow_price_and_size():
    assert [cloud_tier(x) for x in (0.4, 1.99, 2.0, 9.99, 10.0, 75.0)] == [
        "easy", "easy", "medium", "medium", "hard", "hard"]
    assert [local_tier(x) for x in (0.27, 4.7, 8.0, 12.2, 30.0, 70.6, None)] == [
        "easy", "easy", "medium", "medium", "hard", "hard", "easy"]


def test_local_cards_need_tools_and_skip_cloud_and_embedding():
    described = {
        "qwen3.5:4b": {"capabilities": ["completion", "tools", "thinking"], "details": {"parameter_size": "4.7B"},
                       "model_info": {"qwen35.context_length": 262144}, "size": 3_400_000_000},
        "gemma3:270m": {"capabilities": ["completion"], "details": {"parameter_size": "268.10M"}},
        "nomic-embed-text:latest": {"capabilities": ["embedding"], "details": {"parameter_size": "137M"}},
        "gpt-oss:120b-cloud": {"capabilities": ["completion", "tools"], "details": {"parameter_size": "116.8B"}},
        "llama3.3:70b": {"details": {"parameter_size": "70.6B"}},  # an older Ollama: no capabilities reported
        "gemma4:12b": {"capabilities": ["completion", "tools", "vision"], "details": {"parameter_size": "12.2B"}},
    }
    cards = {c.model: c for c in local_cards(described)}
    assert set(cards) == {"ollama_chat/qwen3.5:4b", "ollama_chat/llama3.3:70b", "ollama_chat/gemma4:12b"}
    qwen = cards["ollama_chat/qwen3.5:4b"]
    assert (qwen.provider, qwen.local, qwen.reasoning, qwen.context, qwen.size_b, qwen.tier, qwen.out_usd) == (
        "ollama", True, True, 262144, 4.7, "easy", 0.0)
    assert cards["ollama_chat/gemma4:12b"].vision and cards["ollama_chat/gemma4:12b"].tier == "medium"
    assert cards["ollama_chat/llama3.3:70b"].tier == "hard"


# ---- picking ----

PICKABLE = [card("ollama_chat/a", "easy", size=4), card("ollama_chat/b", "hard", size=70),
            card("o-e1", "easy", 0.4, reasoning=True), card("o-e2", "easy", 0.2),
            card("o-m1", "medium", 5, reasoning=True), card("o-m2", "medium", 3, reasoning=True),
            card("o-h1", "hard", 20, reasoning=True), card("o-h2", "hard", 75, reasoning=True),
            card("o-h3", "hard", 20, reasoning=True), card("a-h", "hard", 25, reasoning=True, provider="anthropic")]


def names(cards):
    return {c.model for c in cards}


def test_picks_one_per_provider_and_tier():
    assert names(picks(PICKABLE, 1, 30.0, {})) == {"ollama_chat/a", "ollama_chat/b", "o-e1", "o-m2", "o-h3", "a-h"}
    assert names(picks(PICKABLE, 2, 30.0, {})) == {"ollama_chat/a", "ollama_chat/b", "o-e1", "o-e2", "o-m2", "o-m1",
                                                   "o-h3", "o-h1", "a-h"}
    assert "o-h2" in names(picks(PICKABLE, 1, 100.0, {}))  # dearest hard model, once the cap allows it


def test_a_failed_check_drops_that_provider_only():
    assert names(picks(PICKABLE, 1, 30.0, {"openai": []})) == {"ollama_chat/a", "ollama_chat/b", "a-h"}


def test_listed_names_match_without_prefix_or_date():
    assert base("claude-sonnet-4-5-20250929") == base("claude-sonnet-4-5") == "claude-sonnet-4-5"
    assert base("groq/openai/gpt-oss-20b") == base("openai/gpt-oss-20b") == "gpt-oss-20b"
    assert base("gpt-4o-2024-08-06") == "gpt-4o" and base("gpt-4-0613") == "gpt-4"
    cards = [card("claude-x", "hard", 25, provider="anthropic"), card("claude-y", "hard", 20, provider="anthropic")]
    assert names(picks(cards, 2, 30.0, {"anthropic": ["claude-x-20250929"]})) == {"claude-x"}


def test_routing_moves_up_then_down():
    cards = [card("ollama_chat/small", "easy", size=4), card("ollama_chat/big", "medium", size=12),
             card("ollama_chat/huge", "hard", size=70), card("cheap", "easy", 0.4), card("mid", "medium", 5),
             card("hard1", "hard", 15), card("hard2", "hard", 25)]
    assert dict(routing(cards)) == {
        "easy": ("ollama_chat/small", "cheap", "ollama_chat/big", "mid", "ollama_chat/huge", "hard1", "hard2"),
        "medium": ("ollama_chat/big", "mid", "ollama_chat/huge", "hard1", "hard2", "ollama_chat/small", "cheap"),
        "hard": ("hard2", "hard1", "ollama_chat/huge", "mid", "ollama_chat/big", "cheap", "ollama_chat/small")}


# ---- recon: the cache, refresh, check and models=auto ----

class DescribingOllama:
    """Ollama with one pulled model that calls tools; `down` makes it unreachable."""

    def __init__(self, down=False):
        self.down = down

    async def describe(self):
        if self.down:
            raise httpx.ConnectError("connection refused")
        return {"tiny:1b": {"capabilities": ["completion", "tools"], "details": {"parameter_size": "1.2B"}}}


def served(payload, status=200, seen=None):
    def handle(request):
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(status, json=payload) if not isinstance(payload, str) else \
            httpx.Response(status, text=payload)
    return httpx.MockTransport(handle)


BIG = {f"model-{i}": entry(1 + i % 20) for i in range(120)}


def test_refresh_caches_a_valid_map(tmp_path):
    path, seen = str(tmp_path / "catalog.json"), []
    download = BIG | {"sample_spec": {"mode": "chat"}, "negative": entry(-1), "words": entry(1) | {
        "input_cost_per_token": "cheap"}, "nested-bad": entry(1) | {"search_context_cost_per_query": {"low": -2}}}
    assert asyncio.run(recon.refresh(path, served(download, seen=seen))) == 120
    cache = recon.load(path)
    assert seen == [recon.MAP_URL] and set(cache["map"]) == set(BIG) and cache["fetched"]


def test_a_bad_download_keeps_the_old_cache(tmp_path):
    path = str(tmp_path / "catalog.json")
    asyncio.run(recon.refresh(path, served(BIG)))
    before = recon.load(path)
    for transport, why in [(served("<html>rate limited</html>"), "could not download"),
                           (served(["not", "a", "map"]), "not a model map"),
                           (served(dict(list(BIG.items())[:5])), "only 5 usable"),
                           (served(BIG, status=500), "could not download")]:
        with pytest.raises(ValueError, match=why):
            asyncio.run(recon.refresh(path, transport))
        assert recon.load(path) == before
    (tmp_path / "broken.json").write_text("{not json")
    assert recon.load(str(tmp_path / "broken.json")) == {} and recon.load(str(tmp_path / "absent.json")) == {}


def test_check_saves_each_providers_list(tmp_path, monkeypatch):
    path = str(tmp_path / "catalog.json")
    asyncio.run(recon.refresh(path, served(BIG)))
    monkeypatch.setattr(recon, "listed", lambda provider: {"openai": ["gpt-cheap"], "anthropic": []}[provider])
    assert asyncio.run(recon.check(path, ["openai", "anthropic"])) == {"openai": ["gpt-cheap"], "anthropic": []}
    cache = recon.load(path)
    assert cache["listed"]["openai"]["models"] == ["gpt-cheap"] and cache["listed"]["anthropic"]["at"]
    assert set(cache["map"]) == set(BIG)  # checking keeps the refreshed map


def test_keys_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert recon.has_key("openai", "gpt-4.1-mini") and not recon.has_key("anthropic", "claude-sonnet-4-5")
    assert not recon.has_key("no-such-provider", "x") and not recon.has_key("bedrock_converse", "x")  # unknowable
    assert recon.os.environ["OPENAI_API_KEY"] == "sk-test"  # the environment is back


def test_the_hub_prices_models_from_the_cached_map(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"map": {"recon-fresh-model": entry(3)}}))
    with pytest.raises(ValueError, match="no price"):
        Hub(make_settings(tmp_path, models=("gpt-4.1-mini", "recon-fresh-model")))
    hub = Hub(make_settings(tmp_path, models=("gpt-4.1-mini", "recon-fresh-model"), model_catalog=str(path)))
    assert "recon-fresh-model" in hub.hireable()


def auto(tmp_path, monkeypatch, keys=("openai",), **overrides):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"map": {k: v for k, v in MAP.items() if isinstance(v, dict)}}))
    monkeypatch.setattr(recon, "has_key", lambda provider, model: provider in keys)
    return make_settings(tmp_path, models=("auto",), model_catalog=str(path), **overrides)


def test_settle_turns_auto_into_models_and_routing(tmp_path, monkeypatch):
    settled = asyncio.run(recon.settle(auto(tmp_path, monkeypatch), DescribingOllama()))
    assert set(settled.models) == {"ollama_chat/tiny:1b", "gpt-cheap", "gpt-mid", "gpt-big"}  # reasoning first
    assert dict(settled.routing)["easy"][:2] == ("ollama_chat/tiny:1b", "gpt-cheap")
    assert dict(settled.routing)["hard"][0] == "gpt-big"
    pinned = make_settings(tmp_path)
    assert asyncio.run(recon.settle(pinned, DescribingOllama(down=True))) is pinned  # only auto is settled


def test_settle_without_ollama_uses_cloud_picks(tmp_path, monkeypatch, caplog):
    settled = asyncio.run(recon.settle(auto(tmp_path, monkeypatch), DescribingOllama(down=True)))
    assert settled.models and not any(m.startswith("ollama_chat/") for m in settled.models)
    assert "Ollama is not reachable" in caplog.text


def test_settle_with_nothing_usable_says_why(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="models=auto found no usable model"):
        asyncio.run(recon.settle(auto(tmp_path, monkeypatch, keys=()), DescribingOllama(down=True)))


def test_settle_opens_no_socket(tmp_path, monkeypatch):
    connects = []

    def offline(self, address):
        connects.append(address)
        raise OSError("offline")

    settings = auto(tmp_path, monkeypatch)
    monkeypatch.setattr(socket.socket, "connect", offline)
    asyncio.run(recon.settle(settings, DescribingOllama()))
    assert connects == []


def test_auto_with_routing_is_a_config_error(tmp_path):
    assert load_settings(["models=auto"]).models == ("auto",)
    config = tmp_path / "auto.yaml"
    config.write_text("models: auto\nrouting:\n  easy: [ollama_chat/tiny]\n")
    with pytest.raises(ValueError, match="routing cannot be set with models=auto"):
        load_settings([], str(config))
    for bad in ("auto_per_tier=0", "auto_max_usd=-1"):
        with pytest.raises(ValueError, match=bad.split("=")[0]):
            load_settings([bad])


def test_an_unsettled_auto_hub_says_so(tmp_path):
    with pytest.raises(ValueError, match="settled first"):
        Hub(make_settings(tmp_path, models=("auto",)))
