"""Every model call goes through LiteLLM, so any provider can be plugged in."""
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # use the bundled price map; no network at import

import litellm  # must come after the env var above

__all__ = ["LLM", "Wallet", "WalletExceeded", "check_priced", "has_price", "is_local", "register_prices"]

litellm.suppress_debug_info = True


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
    return model.startswith(("ollama_chat/", "ollama/"))


def has_price(model: str) -> bool:
    try:
        litellm.get_model_info(model)
    except Exception:  # LiteLLM raises a plain Exception for models it has not mapped
        return False
    return True


def register_prices(path: Path) -> None:
    """Add LiteLLM price entries (same format as its model map) for cloud models the map lacks."""
    litellm.register_model(json.loads(path.read_text()))


def check_priced(models: list[str]) -> None:
    """Fail fast: neither the pools nor the wallet can see the cost of an unpriced cloud model."""
    unpriced = [m for m in models if not is_local(m) and not has_price(m)]
    if unpriced:
        raise ValueError(f"LiteLLM has no price for {unpriced}; add one to a prices file (prices_file=...)")


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
        extra = {"api_base": self.ollama_url, "num_ctx": self.num_ctx} if is_local(model) else {}
        resp = await litellm.acompletion(model=model, messages=messages, tools=tools or None,
                                         temperature=0, num_retries=2, **extra)  # ride out a 429/503
        cost = 0.0 if is_local(model) else float(litellm.completion_cost(completion_response=resp))
        self.wallet.spent += cost
        return resp.choices[0].message, cost
