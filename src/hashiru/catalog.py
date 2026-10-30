"""Which models this machine can use, and how they are classified (recon.py does the I/O).

A cloud model counts when LiteLLM's map lists it as a chat model that calls tools (so it can be a manager), priced
above $0, not past its deprecation date, with its provider's key set, and not a special-purpose or floating name
(fine-tunes, audio, search, realtime, ..., -latest, -exp). A dated snapshot gives way to its undated alias. A local
model counts when Ollama has it pulled, it is not an ollama.com -cloud tag or embedding-only, and it calls tools.

Each model gets a tier: a cloud model by its price per million output tokens, a local one by its parameter count.
models=auto keeps every usable local model and, per provider and tier, the best few cloud ones (picks), and builds
routing from the tiers: a job tries its own tier first, then the ones above (so a failing worker moves up), then the
ones below.
"""
import math
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from hashiru.config import DIFFICULTIES

__all__ = ["Card", "base", "cloud_cards", "cloud_tier", "local_cards", "local_tier", "picks", "routing"]

TIER_USD = (2.0, 10.0)  # $ per million output tokens: under the first is easy, under the second medium, else hard
TIER_B = (8.0, 30.0)    # billions of parameters, the same way, for local models
SPECIAL = ("audio", "search", "realtime", "computer-use", "robotics", "tts", "transcribe", "image", "embed",
           "moderation", "safeguard", "customtools")
SNAPSHOT = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8}|\d{4})$")  # -2025-08-07, -20250929, -0613
SIZE = re.compile(r"([\d.]+)\s*([KMBT])", re.IGNORECASE)     # Ollama's parameter_size: 4.7B, 268.10M
SCALE = {"K": 1e-6, "M": 1e-3, "B": 1.0, "T": 1e3}


@dataclass(frozen=True)
class Card:
    """One usable model. Prices are $ per million tokens (0 for local models)."""

    model: str
    provider: str
    local: bool
    tools: bool
    reasoning: bool
    vision: bool
    context: int | None
    in_usd: float
    out_usd: float
    size_b: float | None  # billions of parameters, for local models
    deprecated: str       # the date LiteLLM retires it, if it gives one
    tier: str


def cloud_tier(out_usd: float) -> str:
    return "easy" if out_usd < TIER_USD[0] else "medium" if out_usd < TIER_USD[1] else "hard"


def local_tier(size_b: float | None) -> str:
    if size_b is None:
        return "easy"
    return "easy" if size_b < TIER_B[0] else "medium" if size_b < TIER_B[1] else "hard"


def base(model: str) -> str:
    """A model's name without provider prefix or snapshot date, to match what a provider lists against the map."""
    return SNAPSHOT.sub("", model.rsplit("/", 1)[-1])


def special(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("ft:") or lower.endswith("-latest") or "-exp" in lower or any(s in lower for s in SPECIAL)


def per_million(entry: Mapping[str, Any], side: str) -> float:
    """$ per million tokens, or 0 when the map gives no usable price."""
    try:
        value = float(entry.get(f"{side}_cost_per_token") or 0) * 1e6
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def cloud_cards(model_map: Mapping[str, Any], keyed: Callable[[str], bool], today: str) -> list[Card]:
    """The cloud models in a LiteLLM-style map that this machine can use (`keyed`: its provider's key is set)."""
    found: dict[str, Card] = {}
    for name, entry in model_map.items():
        if not isinstance(entry, dict) or entry.get("mode") != "chat" or entry.get("supports_function_calling") \
                is not True:
            continue
        inp, out, retired = per_million(entry, "input"), per_million(entry, "output"), entry.get("deprecation_date")
        deprecated = retired if isinstance(retired, str) else ""
        if not (inp and out) or (deprecated and deprecated <= today) or special(name) or not keyed(name):
            continue
        context = entry.get("max_input_tokens")
        found[name] = Card(name, str(entry.get("litellm_provider", "")), False, True,
                           entry.get("supports_reasoning") is True, entry.get("supports_vision") is True,
                           context if isinstance(context, int) else None, inp, out, None, deprecated, cloud_tier(out))
    return [c for name, c in found.items() if not (SNAPSHOT.search(name) and SNAPSHOT.sub("", name) in found)]


def local_cards(described: Mapping[str, Mapping[str, Any]]) -> list[Card]:
    """The local models Ollama describes (tag -> its /api/show answer) that can work as agents."""
    cards = []
    for tag, info in described.items():
        caps = info.get("capabilities")
        if tag.endswith("-cloud") or (isinstance(caps, list) and not {"completion", "tools"} <= set(caps)):
            continue  # an Ollama that reports no capabilities is trusted to call tools
        details, model_info = info.get("details") or {}, info.get("model_info") or {}
        found = SIZE.search(str(details.get("parameter_size", "")))
        size = round(float(found.group(1)) * SCALE[found.group(2).upper()], 4) if found else None
        context = next((v for k, v in model_info.items() if k.endswith(".context_length") and isinstance(v, int)),
                       None)
        caps = caps if isinstance(caps, list) else []
        cards.append(Card(f"ollama_chat/{tag}", "ollama", True, True, "thinking" in caps, "vision" in caps, context,
                          0.0, 0.0, size, "", local_tier(size)))
    return cards


def picks(cards: Sequence[Card], per_tier: int, max_usd: float, listed: Mapping[str, Collection[str]]) -> list[Card]:
    """What models=auto hires from: every local card, and per provider and tier the `per_tier` best cloud cards
    costing at most `max_usd` per million output tokens: reasoning models first, then the cheapest (easy, medium) or
    the dearest (hard), ties by name descending. A provider in `listed` was checked: only what it listed counts, and
    nothing when it listed nothing (its key failed)."""
    seen = {provider: {base(m) for m in models} for provider, models in listed.items()}
    chosen = [c for c in cards if c.local]
    groups: dict[tuple[str, str], list[Card]] = {}
    for c in cards:
        if c.local or c.out_usd > max_usd or (c.provider in seen and base(c.model) not in seen[c.provider]):
            continue
        groups.setdefault((c.provider, c.tier), []).append(c)
    for (_, tier), group in sorted(groups.items()):
        group.sort(key=lambda c: c.model, reverse=True)  # the tie-break; the next sort is stable
        group.sort(key=lambda c: (not c.reasoning, -c.out_usd if tier == "hard" else c.out_usd))
        chosen += group[:per_tier]
    return chosen


def order(cards: Sequence[Card], difficulty: str) -> list[Card]:
    """One tier's cards in the order a `difficulty` job tries them (as router.Router.default orders models)."""
    local = sorted((c for c in cards if c.local), key=lambda c: c.size_b or 0.0, reverse=difficulty != "easy")
    cloud = sorted((c for c in cards if not c.local), key=lambda c: c.out_usd, reverse=difficulty == "hard")
    return cloud + local if difficulty == "hard" else local + cloud


def routing(cards: Sequence[Card]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The routing setting for these cards: per difficulty its own tier, then the tiers above, then those below."""
    out = []
    for i, difficulty in enumerate(DIFFICULTIES):
        tiers = [*DIFFICULTIES[i:], *reversed(DIFFICULTIES[:i])]
        out.append((difficulty, tuple(c.model for t in tiers for c in order([c for c in cards if c.tier == t],
                                                                            difficulty))))
    return tuple(out)
