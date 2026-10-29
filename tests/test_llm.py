"""LiteLLM wrapper: tiers, prices, costs and the wallet cap (LiteLLM itself is faked)."""
import asyncio
import json
from types import SimpleNamespace

import pytest

from hashiru import llm


def fake_response(text="hi"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))])


def fake_litellm(monkeypatch, cost=0.01):
    seen: list[dict] = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        return fake_response("hello")
    monkeypatch.setattr(llm.litellm, "acompletion", acompletion)
    monkeypatch.setattr(llm.litellm, "completion_cost", lambda completion_response: cost)
    return seen


def test_is_local():
    assert llm.is_local("ollama_chat/qwen3.5:4b") and llm.is_local("ollama/gemma4:12b")
    assert not llm.is_local("gemini/gemini-2.5-flash")


def test_check_priced_rejects_unknown_cloud_model():
    llm.check_priced(["ollama_chat/anything", "gpt-4.1-mini"])
    with pytest.raises(ValueError, match="no price"):
        llm.check_priced(["made-up-provider/model-x"])


def test_prices_file_fills_gaps_in_litellms_price_map(tmp_path):
    model = "hashiru_test/priced-model"
    assert not llm.has_price(model)
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({model: {"litellm_provider": "openai", "mode": "chat",
                                          "input_cost_per_token": 1e-7, "output_cost_per_token": 2e-7}}))
    llm.register_prices(prices)
    assert llm.has_price(model)


def test_chat_returns_message_and_cost(monkeypatch):
    fake_litellm(monkeypatch, cost=0.01)
    model = llm.LLM("http://ollama", llm.Wallet(2.0))
    msg, cost = asyncio.run(model.chat("gpt-4.1-mini", [{"role": "user", "content": "x"}]))
    assert msg.content == "hello" and cost == 0.01 and model.wallet.spent == 0.01


def test_local_calls_cost_nothing_and_use_ollama(monkeypatch):
    seen = fake_litellm(monkeypatch, cost=9.9)
    model = llm.LLM("http://ollama", llm.Wallet(2.0))
    _, cost = asyncio.run(model.chat("ollama_chat/qwen3.5:4b", []))
    assert cost == 0.0 and model.wallet.spent == 0.0
    assert seen[0]["api_base"] == "http://ollama" and seen[0]["temperature"] == 0 and seen[0]["num_retries"] == 2


def test_wallet_cap_stops_all_further_calls(monkeypatch):
    seen = fake_litellm(monkeypatch, cost=1.5)
    model = llm.LLM("http://ollama", llm.Wallet(2.0))
    asyncio.run(model.chat("gpt-4.1-mini", []))   # $1.50 spent
    asyncio.run(model.chat("gpt-4.1-mini", []))   # $3.00: crosses the cap, but this answer is still returned
    for _ in range(2):                            # every later call, local or cloud, is refused before the provider
        with pytest.raises(llm.WalletExceeded):
            asyncio.run(model.chat("ollama_chat/qwen3.5:4b", []))
    assert len(seen) == 2


def test_local_models_get_a_context_window_big_enough_to_answer(monkeypatch):
    seen = fake_litellm(monkeypatch)
    model = llm.LLM("http://ollama", llm.Wallet(2.0), num_ctx=32768)
    asyncio.run(model.chat("ollama_chat/qwen3.5:4b", []))
    asyncio.run(model.chat("gpt-4.1-mini", []))
    assert seen[0]["num_ctx"] == 32768 and "num_ctx" not in seen[1]  # cloud providers would reject it


def test_loading_uses_the_same_context_window_and_reports_the_loaded_size(monkeypatch):
    from hashiru.ollama import Ollama

    calls: list[tuple[str, object]] = []
    loaded = {"models": [{"name": "qwen3.5:4b", "size": 3537 * 2**20}]}

    async def request(path, body=None):
        calls.append((path, body))
        return loaded if path == "/api/ps" else {"models": [{"name": "qwen3.5:4b", "size": 3232 * 2**20}]}
    ollama = Ollama("http://ollama", ("ollama_chat/qwen3.5:4b",), num_ctx=16384)
    monkeypatch.setattr(ollama, "request", request)
    assert asyncio.run(ollama.load("ollama_chat/qwen3.5:4b")) == 3537   # weights plus context, not the file size
    assert calls[0][1]["options"] == {"num_ctx": 16384}
    loaded["models"] = []  # Ollama did not report it: fall back to the file size
    assert asyncio.run(ollama.load("ollama_chat/qwen3.5:4b")) == 3232
