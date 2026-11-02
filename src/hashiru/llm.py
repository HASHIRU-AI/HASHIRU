"""Every model call goes through LiteLLM, so any provider can be plugged in."""
import json
import math
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # use the bundled price map; no network at import

import litellm  # must come after the env var above

__all__ = ["INPUT_MARGIN", "LLM", "OUTPUT_CAP", "Wallet", "WalletExceeded", "affordable", "check_priced", "has_price",
           "is_local", "litellm_output_price", "register_prices"]

litellm.suppress_debug_info = True

# The most output tokens the current call may produce: set by runtime.think so a call cannot cost more than its
# agent's pool holds. A context variable, so concurrent calls in one process each carry their own.
OUTPUT_CAP: ContextVar[int | None] = ContextVar("hashiru_output_cap", default=None)
INPUT_MARGIN = 1.1  # token counts are estimates: price the input a little high


class WalletExceeded(RuntimeError):
    """Real spend passed max_usd. A guard on the user's money, separate from the agents' pools."""


class Wallet:
    """Process-wide cap on real API spend, shared by every session in the hub.

    The cap is checked before a call and charged after it, so calls already in flight when it is crossed all
    complete: spend can exceed max_usd by up to one call per concurrently running agent.
    """

    def __init__(self, max_usd: float) -> None:
        self.max_usd = max_usd
        self.spent = 0.0

    def check(self) -> None:
        if self.spent > self.max_usd:
            raise WalletExceeded(f"spent ${self.spent:.4f}; the cap is max_usd=${self.max_usd}")


def is_local(model: str) -> bool:
    """Served by Ollama on this machine. Ollama's -cloud tags (e.g. gpt-oss:120b-cloud) run on ollama.com: cloud."""
    return model.startswith(("ollama_chat/", "ollama/")) and not model.endswith("-cloud")


def has_price(model: str) -> bool:
    """LiteLLM knows what a token of `model` costs. Knowing its name is not enough: for a model newer than its map it
    answers with $0, and the provider would bill calls that no pool or wallet ever sees."""
    try:
        info = litellm.get_model_info(model)
    except Exception:  # LiteLLM raises a plain Exception for models it has not mapped
        return False
    return (info.get("input_cost_per_token") or 0) > 0 or (info.get("output_cost_per_token") or 0) > 0


def affordable(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None,
               left_usd: float) -> tuple[int | None, float]:
    """(the most output tokens `left_usd` pays for after reading the input, the input's estimated cost). None tokens
    means no cap: a local model, or one without an output price."""
    if is_local(model):
        return None, 0.0
    info = litellm.get_model_info(model)
    tokens = litellm.token_counter(model=model, messages=messages) + (len(json.dumps(tools)) // 4 if tools else 0)
    cost_in = tokens * (info.get("input_cost_per_token") or 0) * INPUT_MARGIN
    out = info.get("output_cost_per_token") or 0
    if out <= 0:
        return None, cost_in
    return max(0, math.floor((left_usd - cost_in) / out)), cost_in


def litellm_output_price(model: str) -> float:
    """Dollars per output token, or 0 where LiteLLM has no price (local models)."""
    try:
        return float(litellm.get_model_info(model).get("output_cost_per_token") or 0)
    except Exception:  # LiteLLM raises a plain Exception for models it has not mapped
        return 0.0


def register_prices(path: Path) -> None:
    """Add LiteLLM price entries (same format as its model map) for cloud models the map lacks."""
    litellm.register_model(json.loads(path.read_text()))


def check_priced(models: list[str]) -> None:
    """Fail fast: neither the pools nor the wallet can see the cost of an unpriced cloud model."""
    unpriced = [m for m in models if not is_local(m) and not has_price(m)]
    if unpriced:
        raise ValueError(f"LiteLLM has no price for {unpriced}; run `hashiru models --refresh` for its newest "
                         "prices, or add one to a prices file (prices_file=...)")


class LLM:
    """Chat completions at temperature 0, with cost from LiteLLM's price map (local calls cost $0)."""

    def __init__(self, ollama_url: str, wallet: Wallet, num_ctx: int = 16384) -> None:
        self.ollama_url = ollama_url
        self.wallet = wallet
        self.num_ctx = num_ctx  # Ollama's own default (4096) leaves a thinking model no room to answer

    async def chat(self, model: str, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]] | None = None) -> tuple[Any, float]:
        """One model call. Returns (message, cost_usd).

        The wallet is checked before each call, so the call that crosses the cap is still returned and
        traced, and every later call by any agent in any session is refused.
        """
        self.wallet.check()
        extra: dict[str, Any] = {"api_base": self.ollama_url, "num_ctx": self.num_ctx} if is_local(model) else {}
        if (cap := OUTPUT_CAP.get()) is not None:
            extra["max_tokens"] = cap  # LiteLLM sends max_completion_tokens to reasoning models
        # drop_params: a reasoning model that rejects temperature=0 (gpt-6-luna) runs at its default instead
        resp = await litellm.acompletion(model=model, messages=messages, tools=tools or None, temperature=0,
                                         drop_params=True, num_retries=2, **extra)  # ride out a 429/503
        cost = 0.0 if is_local(model) else float(litellm.completion_cost(completion_response=resp))
        self.wallet.spent += cost
        return resp.choices[0].message, cost
