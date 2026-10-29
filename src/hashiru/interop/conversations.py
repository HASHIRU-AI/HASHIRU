"""Conversations: what the chat page, OpenAI-style clients, A2A clients and MCP clients talk to.

Each conversation is one hub session (a CEO, its hires, its trace) plus the chat so far. OpenAI-style clients
resend the whole chat with every message, so a request continues a conversation only when its earlier messages
match that conversation's history; anything else (a new chat, an edited message, a regenerated reply) gets a new
session that is shown the chat so far as text.
"""
import asyncio
import hashlib
import json
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hashiru.runtime import DEADLINE, ask_agent
from hashiru.session import USER, Session

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["Busy", "Conversation", "Conversations", "History", "ShuttingDown", "delivered", "fingerprint",
           "normalize", "with_history"]

History = list[tuple[str, str]]  # (what the user said, what they were shown)
FOLDED = (("<think>", "</think>"), ("<details", "</details>"))  # reasoning blocks chat apps fold into replies
KEEPALIVE = 15.0  # seconds without events before ask() yields an "idle" event


class Busy(RuntimeError):
    """The conversation is still answering, or every conversation slot is in use."""


class ShuttingDown(Busy):
    """The hub is closing its conversations and opens no new ones."""


def _folded_start(lower: str, tag: str, start: int) -> int:
    """Where the next opening `tag` begins at or after `start`, or -1. `<details` must end its name there."""
    while (at := lower.find(tag, start)) >= 0:
        after = lower[at + len(tag): at + len(tag) + 1]
        if tag != "<details" or after in (">", "") or after.isspace():
            return at
        start = at + 1
    return -1


def normalize(text: str) -> str:
    """What a chat app may change when it sends our reply back: folded reasoning blocks and whitespace.

    One pass over the text (a regex here was quadratic on unclosed tags): each tag's search only moves forward, and
    an opening tag without its closing tag ends the scan.
    """
    lower, out, at = text.lower(), [], 0
    starts = {opening: _folded_start(lower, opening, 0) for opening, _ in FOLDED}
    while True:
        for opening, _ in FOLDED:
            if 0 <= starts[opening] < at:
                starts[opening] = _folded_start(lower, opening, at)
        found = [(starts[o], o, c) for o, c in FOLDED if starts[o] >= 0]
        if not found:
            break
        begin, opening, closing = min(found)
        end = lower.find(closing, begin)
        if end < 0:
            break
        out.append(text[at:begin])
        at = end + len(closing)
    out.append(text[at:])
    return " ".join(" ".join(out).split())


def fingerprint(history: History) -> str:
    return hashlib.sha256(json.dumps([[normalize(u), normalize(a)] for u, a in history]).encode()).hexdigest()


def with_history(session: Session, history: History, text: str) -> str:
    """The CEO keeps no memory between messages, so each one carries the chat so far and who is still hired."""
    if not history:
        return text
    past = "\n".join(f"User: {said}\nYou: {reply}" for said, reply in history)
    live = [f"{a.name} ({a.role}, {a.tier} {a.model})" for a in session.live_hires("ceo")]
    return (f"Our conversation so far:\n{past}\nAgents you hired that are still live: {', '.join(live) or 'none'}."
            f"\n\nThe user's new message: {text}")


def delivered(event: dict[str, Any]) -> str | None:
    """What deliver_to_user sent straight to the user (not through the CEO's reply), as shown to them."""
    if event["kind"] != "tool" or event.get("name") != "deliver_to_user" or "error" in event.get("result", {}):
        return None
    return f"**{event['agent']}:** {event['args'].get('text', '')}"


@dataclass
class Conversation:
    session: Session
    history: History = field(default_factory=list)
    busy: bool = False
    used: float = field(default_factory=time.monotonic)
    context: str = ""  # the A2A context it answers, if an A2A client opened it
    head: str = ""     # fingerprint(history), kept current

    @property
    def id(self) -> str:
        return self.session.id


class Conversations:
    """Open conversations, at most max_sessions; idle ones close after session_ttl seconds."""

    def __init__(self, hub: "Hub") -> None:
        self.hub = hub
        self.open_: dict[str, Conversation] = {}
        self.shutting_down = False
        self._heads: dict[str, str] = {}      # fingerprint of a conversation's history -> its id
        self._contexts: dict[str, str] = {}   # A2A context id -> conversation id
        self._closing: set[asyncio.Task[None]] = set()
        self._opening = asyncio.Lock()        # makes room and opens one conversation at a time

    def get(self, cid: str) -> Conversation | None:
        return self.open_.get(cid)

    def find(self, past: History, cid: str = "") -> Conversation | None:
        """The open conversation whose history is exactly `past` (preferring the one named `cid`)."""
        if not past:
            return None
        head = fingerprint(past)
        named = self.open_.get(cid) if cid else None
        if named is not None and named.head == head:
            return named
        return self.open_.get(self._heads.get(head, ""))

    def for_context(self, context: str) -> Conversation | None:
        return self.open_.get(self._contexts.get(context, ""))

    async def open(self, label: str, history: History | None = None, context: str = "") -> Conversation:
        """A new conversation on a fresh session. `history` is the chat so far, if it began elsewhere."""
        async with self._opening:
            if self.shutting_down:
                raise ShuttingDown("the hub is shutting down")
            await self.sweep()
            if len(self.open_) >= self.hub.settings.max_sessions:
                idle = [c for c in self.open_.values() if not c.busy]
                if not idle:
                    raise Busy(f"all {len(self.open_)} conversations are busy; try again shortly")
                await self.close(min(idle, key=lambda c: c.used))
            past = list(history or [])
            conv = Conversation(self.hub.new_session(label), past, context=context, head=fingerprint(past))
            self.open_[conv.id] = conv
            if context:
                self._contexts[context] = conv.id
            return conv

    def ask(self, conv: Conversation, text: str, instructions: str = "",
            deadline: float | None = None) -> AsyncGenerator[dict[str, Any]]:
        """Send one user message to the CEO. Yields the turn's trace events as they happen ({"kind": "idle"} after
        KEEPALIVE quiet seconds), then {"kind": "answer", "text": what the user is shown, "cost_usd": ...}.

        The conversation is claimed at once (Busy if it already is), so two requests cannot both start a turn.
        `deadline` (Unix time) can only shorten task_timeout. Consumers iterate it under contextlib.aclosing: if they
        stop before the answer, the turn is stopped and the conversation closed. A reply that is an error also closes
        it: the CEO's run may still be going, and a new turn would race it.
        """
        if conv.busy:
            raise Busy("this conversation is still answering the last message")
        if conv.id not in self.open_:
            raise Busy("this conversation has been closed")
        conv.busy, conv.used = True, time.monotonic()
        return self._turn(conv, text, instructions, deadline)

    async def _turn(self, conv: Conversation, text: str, instructions: str,
                    deadline: float | None) -> AsyncGenerator[dict[str, Any]]:
        session, queue = conv.session, asyncio.Queue[dict[str, Any]]()
        start = len(session.trace.events)
        stop_listening = session.trace.listen(queue.put_nowait)
        prompt = self.hub.recall(session, text)
        if instructions:
            prompt = f"The user's chat app adds these instructions:\n{instructions}\n\n{prompt}"
        limit = time.time() + self.hub.settings.task_timeout
        token = DEADLINE.set(limit if deadline is None else min(limit, deadline))  # the run's task copies it
        try:
            run = asyncio.create_task(
                ask_agent(self.hub, session, USER, "ceo", with_history(session, conv.history, prompt)))
        finally:
            DEADLINE.reset(token)
        run.add_done_callback(lambda _: queue.put_nowait({"kind": "_done"}))
        answered = False
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), KEEPALIVE)
                except TimeoutError:
                    yield {"kind": "idle"}
                    continue
                if event["kind"] == "_done":
                    break
                yield event
            reply = run.result()
            turn = session.trace.events[start:]
            shown = "\n\n".join([*filter(None, map(delivered, turn)), reply])
            cost = sum(e.get("cost_usd", 0.0) for e in turn if e["kind"] == "llm")
            self._remember(conv, text, shown)
            answered = True
            if reply.startswith("error: "):
                self.close_soon(conv)
            yield {"kind": "answer", "text": shown, "reply": reply, "cost_usd": round(cost, 8)}
        finally:
            stop_listening()
            conv.busy, conv.used = False, time.monotonic()
            if not answered:
                run.cancel()
                self.close_soon(conv)

    def _remember(self, conv: Conversation, said: str, shown: str) -> None:
        conv.history.append((said, shown))
        if self._heads.get(conv.head) == conv.id:  # another conversation with the same history may own it now
            del self._heads[conv.head]
        conv.head = fingerprint(conv.history)
        if conv.id in self.open_:  # closed during the turn (DELETE, end_session, sweep): nothing to find
            self._heads[conv.head] = conv.id

    async def close(self, conv: Conversation) -> None:
        if self.open_.pop(conv.id, None) is None:
            return
        self._heads = {h: c for h, c in self._heads.items() if c != conv.id}
        self._contexts = {x: c for x, c in self._contexts.items() if c != conv.id}
        await self.hub.close_session(conv.session)

    def close_soon(self, conv: Conversation) -> None:
        """close() from a path that cannot wait for it (a cancelled stream, a finished turn)."""
        task = asyncio.get_running_loop().create_task(self.close(conv))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    def expired(self, conv: Conversation) -> bool:
        """Idle longer than session_ttl. A conversation still busy long after any turn's deadline lost its turn
        before it started (a client that left before its stream began)."""
        s = self.hub.settings
        return time.monotonic() - conv.used > s.session_ttl + (s.task_timeout + 60 if conv.busy else 0)

    async def sweep(self) -> None:
        """Close expired conversations, checking each one again right before closing it: while an earlier close
        was awaited, a client may have started a turn in it."""
        for conv in list(self.open_.values()):
            if conv.id in self.open_ and self.expired(conv):
                await self.close(conv)

    async def sweeper(self) -> None:
        """Sweep every so often while the hub serves, so idle conversations expire without new ones opening."""
        while True:
            await asyncio.sleep(min(60.0, self.hub.settings.session_ttl / 2))
            await self.sweep()

    async def close_all(self) -> None:
        """At shutdown: open no more conversations and close every one at once (each may wait for runs to stop)."""
        self.shutting_down = True
        await asyncio.gather(*(self.close(c) for c in list(self.open_.values())))
        if self._closing:
            await asyncio.wait(list(self._closing), timeout=10)

    def state(self, conv: Conversation) -> dict[str, Any]:
        """What the chat page's agents panel draws: every agent with its pool, and the session's totals."""
        s, ledger = self.hub.settings, conv.session.ledger
        left = ledger.left()
        agents = [{"name": a.name, "role": a.role, "tier": a.tier, "model": a.model, "parent": a.parent,
                   "depth": a.depth, "live": a.live, "reserved_usd": round(a.reserved, 6),
                   "spent_usd": round(a.spent, 6), "left_usd": round(left[a.name], 6), "gpu_mb": a.gpu_mb}
                  for a in ledger.agents.values()]
        return {"session": conv.id, "busy": conv.busy, "turns": len(conv.history), "budget_usd": s.budget_usd,
                "spent_usd": round(sum(a.spent for a in ledger.agents.values()), 6),
                "wallet": {"max_usd": self.hub.wallet.max_usd, "spent_usd": round(self.hub.wallet.spent, 6)},
                "agents": agents}
