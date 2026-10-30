"""Model recon: what this machine can use now (catalog.py classifies it), and models=auto.

The model map is LiteLLM's: the copy bundled with LiteLLM, or a newer one that `hashiru models --refresh` downloaded
into the model_catalog file. The keys are the environment's (litellm.validate_environment, no network); `--check`
also asks each keyed provider which models it lists (a listing call: nothing runs, nothing is billed) and keeps the
answer in the same file. Local models come from this machine's Ollama. Nothing else touches the network: starting
the hub, with or without models=auto, reads only that file, the environment and Ollama.
"""
import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
import litellm

from hashiru.catalog import Card, cloud_cards, local_cards, picks, routing
from hashiru.config import AUTO, Settings
from hashiru.ollama import Ollama

__all__ = ["MAP_URL", "Survey", "check", "has_key", "listed", "load", "refresh", "settle", "survey", "use_cached_map"]

log = logging.getLogger(__name__)
MAP_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
MIN_ENTRIES = 100  # fewer usable models than this is not LiteLLM's map: keep the one we have


def load(path: str) -> dict[str, Any]:
    """The model_catalog file: {"map", "fetched", "listed": {provider: {"at", "models"}}}; {} if absent or broken."""
    try:
        data = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        log.warning("ignoring the model catalog %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def save(path: str, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(data))
    os.replace(partial, target)  # a reader never sees half a file


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def priced(value: Any) -> bool:
    return value is None or (isinstance(value, int | float) and not isinstance(value, bool)
                             and math.isfinite(value) and value >= 0)


def sane(entry: Any) -> bool:
    """A map entry whose costs are all numbers >= 0 (some costs are tables of numbers)."""
    if not isinstance(entry, dict):
        return False
    for key, value in entry.items():
        if "cost" in key and not all(map(priced, value.values() if isinstance(value, dict) else [value])):
            return False
    return True


async def refresh(path: str, transport: httpx.AsyncBaseTransport | None = None) -> int:
    """Download LiteLLM's newest model map into the catalog file and return how many models it kept. Entries with a
    cost that is not a number >= 0 are dropped; a download that is not a model map changes nothing (ValueError)."""
    try:
        async with httpx.AsyncClient(transport=transport, timeout=30, follow_redirects=True) as http:
            resp = await http.get(MAP_URL)
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:  # a JSON error is a ValueError
        raise ValueError(f"could not download LiteLLM's model map: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("the download is not a model map (a JSON object of models)")
    kept = {name: entry for name, entry in data.items() if name != "sample_spec" and sane(entry)}
    if len(kept) < MIN_ENTRIES:
        raise ValueError(f"the download has only {len(kept)} usable models; keeping the map you have")
    save(path, load(path) | {"map": kept, "fetched": now()})
    return len(kept)


def listed(provider: str) -> list[str]:
    """The models a provider lists for the key in the environment. LiteLLM returns [] on any error (a wrong key,
    no network), and its own list for a provider it cannot ask."""
    return list(litellm.get_valid_models(check_provider_endpoint=True, custom_llm_provider=provider))


async def check(path: str, providers: Iterable[str]) -> dict[str, list[str]]:
    """Ask each provider which models it lists, and keep the answers in the catalog file."""
    found = {p: await asyncio.to_thread(listed, p) for p in providers}
    cache = load(path)
    save(path, cache | {"listed": (cache.get("listed") or {}) | {p: {"at": now(), "models": models}
                                                                 for p, models in found.items()}})
    return found


def has_key(provider: str, model: str) -> bool:
    """Whether the environment holds the keys `provider` needs (asked of LiteLLM for one of its models). LiteLLM
    answers yes for a provider whose keys it does not know, so it is asked again with no environment: a provider it
    cannot tell about does not count. Call it before the hub runs (it empties the environment for a moment)."""
    probe = model if model.startswith(f"{provider}/") else f"{provider}/{model}"
    try:
        found = litellm.validate_environment(probe)
        with mock.patch.dict(os.environ, {}, clear=True):
            bare = litellm.validate_environment(probe)
    except Exception:  # LiteLLM raises plain Exceptions for providers it cannot place
        return False
    return bool(found["keys_in_environment"]) and not bare["keys_in_environment"]


def use_cached_map(path: str, models: Iterable[str]) -> None:
    """Price `models` from the refreshed map, when it has them (LiteLLM's bundled map prices the rest)."""
    fresh = load(path).get("map") or {}
    wanted = {m: fresh[m] for m in models if isinstance(fresh.get(m), dict) and sane(fresh[m])}
    if wanted:
        litellm.register_model(wanted)


@dataclass(frozen=True)
class Survey:
    cards: list[Card]  # every usable model
    picks: list[Card]  # what models=auto hires from
    keyed: list[str]   # cloud providers whose keys are set
    listed: dict[str, list[str]]  # what each checked provider listed
    source: str        # where the model map came from
    notes: list[str]


async def survey(settings: Settings, ollama: Any) -> Survey:
    """Everything this machine can use now: the map (refreshed or bundled), the keys, the checks and Ollama."""
    cache = load(settings.model_catalog)
    model_map: Mapping[str, Any] = cache.get("map") or litellm.model_cost
    source = f"refreshed {cache.get('fetched', '')}".strip() if cache.get("map") else "bundled with LiteLLM"
    keys: dict[str, bool] = {}

    def keyed(name: str) -> bool:
        provider = str(model_map[name].get("litellm_provider", ""))
        if provider not in keys:
            keys[provider] = has_key(provider, name)
        return keys[provider]

    cloud = cloud_cards(model_map, keyed, date.today().isoformat())
    notes = []
    try:
        local = local_cards(await ollama.describe())
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        local = []
        notes.append(f"Ollama is not reachable at {settings.ollama_url} ({exc or type(exc).__name__}): no local models")
    checked = {p: [str(m) for m in v["models"]] for p, v in (cache.get("listed") or {}).items()
               if isinstance(v, dict) and isinstance(v.get("models"), list)}
    notes += [f"{p} listed no models when checked: its key may be wrong, so none of its models are picked"
              for p, models in sorted(checked.items()) if not models]
    cards = local + cloud
    return Survey(cards, picks(cards, settings.auto_per_tier, settings.auto_max_usd, checked),
                  sorted(p for p, ok in keys.items() if ok), checked, source, notes)


async def settle(settings: Settings, ollama: Any = None) -> Settings:
    """models=auto made concrete: the picks, and routing built from their tiers. Other settings come back as they
    are."""
    if settings.models != AUTO:
        return settings
    found = await survey(settings, ollama or Ollama(settings.ollama_url, ()))
    for note in found.notes:
        log.warning(note)
    if not found.picks:
        raise ValueError("models=auto found no usable model: pull a local model that calls tools (ollama pull ...) "
                         "or set a provider's API key; `hashiru models` shows what it sees")
    return replace(settings, models=tuple(c.model for c in found.picks), routing=routing(found.picks))
