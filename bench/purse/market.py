"""What can be hired for a job, at what price, and what one attempt came to."""
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from purse.jobs import Job

__all__ = ["ALL_OPTIONS", "API_TIERS", "CEO_MODEL", "MARKET", "OPEN", "PROVIDER_ERROR", "SMALL", "SMALL_OPTION",
           "UNRELIABLE", "Market", "Option", "Outcome", "Tiers", "calibrated", "lookup", "priced", "serves"]

CEO_MODEL = "gpt-6-luna"
PROVIDER_ERROR = "provider error: "  # an attempt the provider or the hub failed, not the model: calibration retries it
SMALL = "gpt-5-nano"  # the small tier: cheaper per token than gpt-6-luna


@dataclass(frozen=True)
class Option:
    """A local model (costs memory), a cloud model (costs dollars) or an outside agent (a price per message)."""

    name: str
    kind: Literal["local", "cloud", "agent"]
    per_message_usd: float = 0.0
    families: tuple[str, ...] = ()  # empty: it serves every family
    input_usd_per_m: float = 0.0  # a cloud model's list prices per million tokens, as allocators are told them
    output_usd_per_m: float = 0.0
    price_factor: float = 1.0  # an outside agent's price relative to the others' (priced)


# The small tier is out of the 2026-11-01 runs: Gemma 4 31B on NVIDIA NIM did not answer then. Round 2 uses
# gpt-5-nano ($0.05 / $0.40, priced in LiteLLM's map), cheaper per token than gpt-6-luna. Set aside: gpt-4.1-nano
# (OpenAI lists it but answers that it is deprecated), Muse Glimmer 30B on NIM (dearer per token than gpt-6-luna, and
# NIM's trial rate limits) and Gemma 4 31B on OpenRouter (no credits); the last two stay priced in prices.json. Local
# models are out: a 4B thinking model took 6 minutes a job on the runs' laptop.
SMALL_OPTION = Option(SMALL, "cloud", input_usd_per_m=0.05, output_usd_per_m=0.40)
MARKET: tuple[Option, ...] = (
    Option("gpt-6-luna", "cloud", input_usd_per_m=0.10, output_usd_per_m=0.50),
    Option("meta/muse-spark-1.3", "cloud", input_usd_per_m=1.25, output_usd_per_m=4.25),
    Option("a2a/calc", "agent", 0.001, ("arithmetic",)),
    Option("a2a/csp", "agent", 0.001, ("grids",)),
)


# A cheap, unreliable seller beside csp: the same grids at a quarter of the price, wrong on 40% of them (run it with
# `purse.peers.csp_agent --name gridfast --wrong 0.4`). Out of MARKET: a table holds it only once it is calibrated.
UNRELIABLE = Option("a2a/gridfast", "agent", 0.001, ("grids",), price_factor=0.25)

# The open-weight market (the pilot of plan/2026-11-02-finetuning-scope.md): Apache-2.0 models served by vLLM on rented
# GPUs and reached with --endpoint, priced at OpenRouter's list prices (read 2026-11-02; bench/prices.json), so a pool
# pays for them as it would through a provider. Smaller than the API market, so that hard jobs can fail.
OPEN: tuple[Option, ...] = (
    Option("hosted_vllm/mistralai/Ministral-3-3B-Instruct-2512", "cloud", input_usd_per_m=0.10, output_usd_per_m=0.10),
    Option("hosted_vllm/Qwen/Qwen3.5-9B", "cloud", input_usd_per_m=0.17, output_usd_per_m=0.25),
    Option("hosted_vllm/google/gemma-4-31B-it", "cloud", input_usd_per_m=0.20, output_usd_per_m=0.40),
    Option("hosted_vllm/Qwen/Qwen3.5-27B", "cloud", input_usd_per_m=0.26, output_usd_per_m=2.60),
)

ALL_OPTIONS = (SMALL_OPTION, *MARKET, UNRELIABLE, *OPEN)


@dataclass(frozen=True)
class Tiers:
    """Which models are a market's small, middle and top tiers: the validity checks, the outside agents' reference
    price (the middle tier's median cost) and the hybrid policy are defined by tier, not by model."""

    small: str
    mid: str
    top: str


API_TIERS = Tiers(SMALL, "gpt-6-luna", "meta/muse-spark-1.3")


def calibrated(names: set[str]) -> tuple[Option, ...]:
    """The options an outcome table holds records for: the market a check or a run should use."""
    return tuple(o for o in ALL_OPTIONS if o.name in names)


@dataclass(frozen=True)
class Outcome:
    answer: str
    correct: bool
    cost_usd: float
    seconds: float
    refused: str = ""  # why the attempt could not run (no record, a refused hire, ...); a refusal is a failure


class Market(Protocol):
    async def attempt(self, job: Job, option: str) -> Outcome:
        """Send `job` to one option and report what came back."""
        ...


def serves(o: Option, job: Job) -> bool:
    return not o.families or job.family in o.families


def priced(market: Sequence[Option], agent_price_usd: float) -> tuple[Option, ...]:
    """The market with every outside agent at `agent_price_usd` a message (times its price_factor)."""
    return tuple(replace(o, per_message_usd=agent_price_usd * o.price_factor) if o.kind == "agent" else o
                 for o in market)


def lookup(market: Sequence[Option], name: str) -> Option:
    for o in market:
        if o.name == name:
            return o
    raise KeyError(f"{name!r} is not in the market")
