"""A session: one CEO, its pool tree, its trace and its memory. Many sessions share one Hub."""
from dataclasses import dataclass, field

from hashiru.ledger import AgentRecord, Ledger
from hashiru.trace import Trace

__all__ = ["USER", "Session"]

USER = "user"  # the human at the top; talks only to the CEO


@dataclass
class Session:
    """State for one user conversation. ``id`` is also the A2A context id and the trace's task id."""

    id: str
    trace: Trace
    ledger: Ledger = field(default_factory=Ledger)
    memory: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, str] = field(default_factory=dict)  # caller name -> bearer token

    def may_ask(self, me: str, name: str) -> bool:
        """Only the hirer may message an agent, and only the user may message the CEO."""
        target = self.ledger.get(name)
        return bool(target and target.live and target.parent == (None if me == USER else me))

    def live_hires(self, parent: str) -> list[AgentRecord]:
        return [a for a in self.ledger.agents.values() if a.parent == parent and a.live]
