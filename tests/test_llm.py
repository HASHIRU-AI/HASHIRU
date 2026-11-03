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


def test_check_priced_rejects_a_cloud_model_litellm_names_but_does_not_price():
    # LiteLLM can know a model without knowing its price (gpt-6-luna under the pinned 1.103.4 map): its info then
    # says $0, and every call would be billed by the provider but never charged to a pool or the wallet
    model = "hashiru_test/name-only"
    llm.litellm.register_model({model: {"litellm_provider": "openai", "mode": "chat"}})
    assert not llm.has_price(model)
    with pytest.raises(ValueError, match="no price"):
        llm.check_priced([model])


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


def test_a_model_with_an_endpoint_is_sent_there_and_priced_as_cloud(monkeypatch):
    seen = fake_litellm(monkeypatch, cost=0.003)
    url = "http://127.0.0.1:8001/v1"
    model = llm.LLM("http://ollama", llm.Wallet(2.0), endpoints={"hosted_vllm/org/small": url})
    _, cost = asyncio.run(model.chat("hosted_vllm/org/small", []))
    asyncio.run(model.chat("gpt-4.1-mini", []))
    assert cost == 0.003 and seen[0]["api_base"] == url and "num_ctx" not in seen[0]
    assert "api_base" not in seen[1]  # other cloud models go to their provider


def rate_limited(monkeypatch, failures: int, cost: float = 0.01) -> tuple[list[dict], list[float]]:
    """acompletion refuses the first `failures` calls with a 429; asyncio.sleep records its waits."""
    seen: list[dict] = []
    waits: list[float] = []

    async def acompletion(**kwargs):
        seen.append(kwargs)
        if len(seen) <= failures:
            raise llm.litellm.RateLimitError("Rate limit reached on tokens per min (TPM)", "openai", "gpt-6-luna")
        return fake_response("hello")

    async def sleep(seconds):
        waits.append(seconds)
    monkeypatch.setattr(llm.litellm, "acompletion", acompletion)
    monkeypatch.setattr(llm.litellm, "completion_cost", lambda completion_response: cost)
    monkeypatch.setattr(llm.asyncio, "sleep", sleep)
    return seen, waits


def test_a_rate_limited_call_waits_and_is_sent_again(monkeypatch):
    # a provider's per-minute limit is no fault of the agent: the call waits for the window to pass, and is charged
    # once, for the attempt that ran (a refused attempt is not billed)
    seen, waits = rate_limited(monkeypatch, failures=3)
    model = llm.LLM("http://ollama", llm.Wallet(2.0))
    msg, cost = asyncio.run(model.chat("gpt-6-luna", []))
    assert msg.content == "hello" and cost == 0.01 and model.wallet.spent == 0.01
    assert len(seen) == 4 and len(waits) == 3 and waits == sorted(waits) and waits[0] < waits[-1]


def test_a_call_still_rate_limited_after_every_wait_raises(monkeypatch):
    seen, waits = rate_limited(monkeypatch, failures=100)
    model = llm.LLM("http://ollama", llm.Wallet(2.0))
    with pytest.raises(llm.litellm.RateLimitError):
        asyncio.run(model.chat("gpt-6-luna", []))
    assert len(waits) == len(llm.RATE_LIMIT_WAITS) and len(seen) == len(waits) + 1
    assert model.wallet.spent == 0.0
    assert sum(llm.RATE_LIMIT_WAITS) >= 60  # long enough for a per-minute window to pass


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


def test_temperature_zero_is_dropped_only_where_a_model_rejects_it(monkeypatch):
    # gpt-6-luna rejects temperature=0 while it reasons (found on the first real call, 2026-11-01): LiteLLM drops
    # the parameter for such models when asked, and keeps it for every other model
    seen = fake_litellm(monkeypatch)
    asyncio.run(llm.LLM("http://ollama", llm.Wallet(2.0)).chat("gpt-4.1-mini", []))
    assert seen[0]["temperature"] == 0 and seen[0]["drop_params"] is True


def test_chat_caps_output_tokens_only_when_asked(monkeypatch):
    seen = fake_litellm(monkeypatch)
    model = llm.LLM("http://ollama", llm.Wallet(2.0))
    token = llm.OUTPUT_CAP.set(300)
    try:
        asyncio.run(model.chat("gpt-4.1-mini", []))
    finally:
        llm.OUTPUT_CAP.reset(token)
    asyncio.run(model.chat("gpt-4.1-mini", []))
    assert seen[0]["max_tokens"] == 300 and "max_tokens" not in seen[1]


def test_affordable_output_is_what_is_left_after_reading_the_input():
    model = "hashiru_test/capped"
    llm.litellm.register_model({model: {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 1e-6,
                                        "output_cost_per_token": 1e-5}})
    messages = [{"role": "user", "content": "hello " * 200}]
    tokens, cost_in = llm.affordable(model, messages, None, 0.01)
    expected_in = llm.litellm.token_counter(model=model, messages=messages) * 1e-6 * llm.INPUT_MARGIN
    assert abs(cost_in - expected_in) < 1e-12 and tokens == int((0.01 - expected_in) / 1e-5)
    assert llm.affordable(model, messages, None, 0.0)[0] == 0
    assert llm.affordable("ollama_chat/anything", messages, None, 0.01)[0] is None  # local: no dollar cap


def test_affordable_output_never_exceeds_what_the_model_can_write():
    model = "hashiru_test/short"
    llm.litellm.register_model({model: {"litellm_provider": "openai", "mode": "chat", "input_cost_per_token": 1e-7,
                                        "output_cost_per_token": 4e-7, "max_output_tokens": 128000}})
    messages = [{"role": "user", "content": "hi"}]
    assert llm.affordable(model, messages, None, 5.0)[0] == 128000  # $5 pays for 12.5M tokens; the model writes 128K
    assert llm.affordable(model, messages, None, 0.001)[0] < 128000  # a small pool still binds first
