"""R3, private stays local: HASHIRU-lite's rule, enforced while agents run.

Data labeled private (read_file, memory and recalled memories, MCP servers configured with labels: [private], and
everything the user says in a private chat) may reach only local agents and the user. A cloud model or a remote
agent never gets it: not in a message, a reply, a hire prompt or a tool result. An agent holding it does not call a
tool that sends data off the machine, unless private_egress is "allow". Every refusal is a 'policy' event.

The runtime and the hub call these checks at their chokepoints (runtime.call_tool and ask_agent, Tools.hire_agent,
Hub.recall); Hub.make_private opens the front door for a private chat.
"""
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

from hashiru.ledger import AgentRecord
from hashiru.session import USER, Session
from hashiru.trace import OBFUSCATED, PRIVATE

if TYPE_CHECKING:
    from hashiru.obfuscation import Obfuscator

__all__ = ["EGRESS", "RULE", "cross", "holds_private", "may_hold", "refuse", "tool_refusal", "where"]
MISSED = "and its private details could not be found to hide them"

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
                 egress: Collection[str], obfuscating: bool = False) -> str | None:
    """Why `rec` may not call `tool` at all, or None. Checked before the call, so the tool never runs. When
    obfuscating, a cloud agent may call a tool that returns private data: it gets the result obfuscated."""
    if PRIVATE in sources.get(tool, ()) and not may_hold(rec.tier) and not obfuscating:
        return refuse(session, rec.name, "tool", f"{tool} returns private data and {rec.name} {where(rec.tier)}; "
                                                 "hire a local agent to use it")
    if tool in egress and holds_private(session, rec.name):
        return refuse(session, rec.name, "egress", f"{rec.name} holds private data and {tool} sends data off this "
                                                   "machine")
    return None


async def cross(session: Session, obfuscator: "Obfuscator | None", text: str, labels: list[str], sender_tier: str,
                receiver_tier: str, agent: str, action: str, why: str) -> tuple[str, list[str], str | None]:
    """R3 for text moving between two parties: (what the receiver gets, its labels, why it was refused or None).

    Private text for a cloud or remote receiver is obfuscated, or refused (in block mode, or if its details could not
    be found). Text from a cloud or remote sender to a local agent or the user gets its placeholders restored, and is
    then private.
    """
    if PRIVATE in labels and not may_hold(receiver_tier):
        if obfuscator is None:
            return text, labels, refuse(session, agent, action, why)
        masked = await obfuscator.mask(session, text)
        if masked is None:
            return text, labels, refuse(session, agent, action, f"{why}, {MISSED}")
        return masked, sorted((set(labels) - {PRIVATE}) | {OBFUSCATED}), None
    if may_hold(receiver_tier):  # a local receiver holds real values, never obfuscated ones
        labels = [label for label in labels if label != OBFUSCATED]
        if obfuscator is not None and not may_hold(sender_tier):
            restored, changed = obfuscator.unmask(session, text)
            if changed:
                return restored, sorted(set(labels) | {PRIVATE}), None
    return text, labels, None
