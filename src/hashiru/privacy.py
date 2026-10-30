"""R3, private stays local: HASHIRU-lite's rule, enforced while agents run.

Data labeled private (read_file, memory and recalled memories, MCP servers configured with labels: [private], and
everything the user says in a private chat) may reach only local agents and the user. A cloud model or a remote
agent never gets it: not in a message, a reply, a hire prompt or a tool result. An agent holding it does not call a
tool that sends data off the machine, unless private_egress is "allow". Every refusal is a 'policy' event.

The runtime and the hub call these checks at their chokepoints (runtime.call_tool and ask_agent, Tools.hire_agent,
Hub.recall); Hub.make_private opens the front door for a private chat.
"""
from collections.abc import Collection, Mapping

from hashiru.ledger import AgentRecord
from hashiru.session import USER, Session
from hashiru.trace import PRIVATE

__all__ = ["EGRESS", "RULE", "holds_private", "may_hold", "refuse", "tool_refusal", "where"]

RULE = "R3"
SAFE = frozenset({"local", USER})  # who may see private data: local agents and the user
# Built-in tools that send data off the machine. Created tools' code is shared with every session, so writing one
# from private data leaks it too.
EGRESS = frozenset({"get_website", "arxiv_search", "create_tool"})


def may_hold(tier: str) -> bool:
    """Whether an agent of this tier (or the user) may see private data."""
    return tier in SAFE


def holds_private(session: Session, agent: str) -> bool:
    return PRIVATE in session.trace.labels(agent, session.id)


def where(tier: str) -> str:
    """Why an agent of this tier may not see private data, for the refusal text."""
    return "is a remote agent" if tier == "remote" else "runs on a cloud model"


def refuse(session: Session, agent: str, action: str, why: str) -> str:
    """Record a refusal and return what the agent is told."""
    session.trace.emit("policy", task=session.id, agent=agent, rule=RULE, action=action, why=why)
    return f"{RULE} (private stays local): {why}"


def tool_refusal(session: Session, rec: AgentRecord, tool: str, sources: Mapping[str, list[str]],
                 egress: Collection[str]) -> str | None:
    """Why `rec` may not call `tool` at all, or None. Checked before the call, so the tool never runs."""
    if PRIVATE in sources.get(tool, ()) and not may_hold(rec.tier):
        return refuse(session, rec.name, "tool", f"{tool} returns private data and {rec.name} {where(rec.tier)}; "
                                                 "hire a local agent to use it")
    if tool in egress and holds_private(session, rec.name):
        return refuse(session, rec.name, "egress", f"{rec.name} holds private data and {tool} sends data off this "
                                                   "machine")
    return None
