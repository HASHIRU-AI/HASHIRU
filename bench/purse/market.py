"""What can be hired for a job, at what price, and what one attempt came to."""
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from purse.jobs import Job

__all__ = ["CEO_MODEL", "MARKET", "Market", "Option", "Outcome", "lookup", "priced", "serves"]

CEO_MODEL = "gpt-6-luna"


@dataclass(frozen=True)
class Option:
    """A local model (costs memory), a cloud model (costs dollars) or an outside agent (a price per message)."""

    name: str
    kind: Literal["local", "cloud", "agent"]
    per_message_usd: float = 0.0
    families: tuple[str, ...] = ()  # empty: it serves every family
    input_usd_per_m: float = 0.0  # a cloud model's list prices per million tokens, as allocators are told them
    output_usd_per_m: float = 0.0


MARKET: tuple[Option, ...] = (
    Option("ollama_chat/qwen3.5:4b", "local"),
    Option("ollama_chat/gemma4:12b", "local"),
    Option("gpt-6-luna", "cloud", input_usd_per_m=0.10, output_usd_per_m=0.50),
    Option("meta/muse-spark-1.3", "cloud", input_usd_per_m=1.25, output_usd_per_m=4.25),
    Option("a2a/calc", "agent", 0.001, ("arithmetic",)),
    Option("a2a/csp", "agent", 0.001, ("grids",)),
)


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
    """The market with every outside agent at `agent_price_usd` a message."""
    return tuple(replace(o, per_message_usd=agent_price_usd) if o.kind == "agent" else o for o in market)


def lookup(market: Sequence[Option], name: str) -> Option:
    for o in market:
        if o.name == name:
            return o
    raise KeyError(f"{name!r} is not in the market")
