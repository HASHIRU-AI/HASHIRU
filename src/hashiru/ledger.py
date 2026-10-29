"""Who exists in a session and what they hold. Accounting only: nothing here ever blocks."""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import psutil

__all__ = ["Account", "AgentRecord", "Ledger", "free_mb", "left", "replay"]


class Account(Protocol):
    parent: str | None
    reserved: float
    spent: float
    live: bool


@dataclass
class AgentRecord:
    """One hired agent. role is "manager" (tool loop, may hire) or "worker" (one LLM call, no tools)."""

    name: str
    role: str
    tier: str  # "local" (Ollama, holds memory) or "cloud" (costs money)
    model: str
    prompt: str
    parent: str | None
    depth: int
    reserved: float
    spent: float = 0.0
    gpu_mb: int = 0
    live: bool = True
    tools: tuple[str, ...] | None = None  # None: every tool; else only these (managers)


@dataclass
class _Replayed:
    parent: str | None
    reserved: float
    spent: float = 0.0
    live: bool = True


def left(accounts: Mapping[str, Account]) -> dict[str, float]:
    """Dollars left per agent: reservation minus own spend minus what its children hold.

    A live child holds its reservation; a fired child holds what it and everyone below it spent
    (API dollars are not refunded). Nobody has overspent while every value is >= 0.
    """
    children: dict[str, list[str]] = {}
    for name, acct in accounts.items():
        if acct.parent is not None:
            children.setdefault(acct.parent, []).append(name)

    def spent_below(name: str) -> float:
        return accounts[name].spent + sum(spent_below(k) for k in children.get(name, []))

    def hold(name: str) -> float:
        return accounts[name].reserved if accounts[name].live else spent_below(name)

    return {name: acct.reserved - acct.spent - sum(hold(k) for k in children.get(name, []))
            for name, acct in accounts.items()}


def replay(events: list[dict[str, Any]]) -> dict[str, _Replayed]:
    """Rebuild the accounts from a trace, so checks need nothing but the trace.

    A hire_agent event without a result is a proposed hire and counts as made.
    """
    accounts: dict[str, _Replayed] = {}
    for e in events:
        kind, tool = e["kind"], e.get("name")
        if kind == "start":
            accounts["ceo"] = _Replayed(parent=None, reserved=float(e["budget_usd"]))
        elif kind == "llm" and e["agent"] in accounts:
            accounts[e["agent"]].spent += e["cost_usd"]
        elif kind == "tool" and tool == "hire_agent" and "error" not in e.get("result", {}):
            accounts[e["args"].get("name", "?")] = _Replayed(
                parent=e["agent"], reserved=float(e["args"].get("budget_usd", 0)))
        elif kind == "tool" and tool == "fire_agent":
            for fired in e.get("result", {}).get("fired", []):
                if fired in accounts:  # an agent fired while its model loaded never got a successful hire
                    accounts[fired].live = False
    return accounts


class Ledger:
    """The agents of one session, keyed by name."""

    def __init__(self) -> None:
        self.agents: dict[str, AgentRecord] = {}

    def add(self, name: str, role: str, tier: str, model: str, prompt: str,
            parent: str | None, depth: int, reserved: float, tools: tuple[str, ...] | None = None) -> AgentRecord:
        """Register an agent holding a dollar reservation taken from its parent's pool."""
        self.agents[name] = AgentRecord(name, role, tier, model, prompt, parent, depth, reserved, tools=tools)
        return self.agents[name]

    def get(self, name: str) -> AgentRecord | None:
        return self.agents.get(name)

    def left(self) -> dict[str, float]:
        return left(self.agents)

    def ancestors(self, name: str) -> list[str]:
        """Everyone above an agent: its hirer, its hirer's hirer, ... up to the CEO."""
        parent = self.agents[name].parent
        return [parent, *self.ancestors(parent)] if parent else []

    def subtree(self, name: str) -> list[str]:
        """An agent and every live agent below it, parents first."""
        out = [name]
        for a in list(self.agents.values()):
            if a.parent == name and a.live:
                out += self.subtree(a.name)
        return out


def free_mb() -> int:
    """Measured free memory in MB (unified memory on Apple silicon, so it covers the GPU)."""
    return int(psutil.virtual_memory().available // 2**20)
