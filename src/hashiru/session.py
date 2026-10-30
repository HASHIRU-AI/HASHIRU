"""A session: one CEO, its pool tree, its trace and its memory. Many sessions share one Hub."""
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from hashiru.approvals import Approvals
from hashiru.ledger import AgentRecord, Ledger
from hashiru.memory import MemoryStore, SessionMemory
from hashiru.obfuscation.vault import Vault
from hashiru.trace import Trace

if TYPE_CHECKING:
    from hashiru.planning import Plan

__all__ = ["USER", "Session"]

USER = "user"  # the human at the top; talks only to the CEO


@dataclass
class Session:
    """State for one user conversation. ``id`` is also the A2A context id and the trace's task id."""

    id: str
    trace: Trace
    ledger: Ledger = field(default_factory=Ledger)
    memory: MemoryStore = field(default_factory=SessionMemory)
    tokens: dict[str, str] = field(default_factory=dict)  # caller name -> bearer token
    vault: Vault = field(default_factory=Vault)  # placeholders for what cloud models see of private data
    approvals: Approvals = field(default_factory=Approvals)  # big spends waiting for the user
    planning: bool = False  # this turn only reads and plans (planning.py); its actions are refused
    plan: "Plan | None" = None  # the plan proposed this turn, or waiting for the user's approval
    approved: "Plan | None" = None  # an approved plan the CEO's run this turn carries out as a graph (plangraph.py)

    def may_ask(self, me: str, name: str) -> bool:
        """Only the hirer may message an agent, and only the user may message the CEO."""
        target = self.ledger.get(name)
        return bool(target and target.live and target.parent == (None if me == USER else me))

    def live_hires(self, parent: str) -> list[AgentRecord]:
        return [a for a in self.ledger.agents.values() if a.parent == parent and a.live]
