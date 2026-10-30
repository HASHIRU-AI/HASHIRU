"""Approvals before a big spend, of dollars (api) or of memory (gpu).

A hire that would reserve approve_hire_usd or more for a cloud model or a remote agent, a hire or an escalation that
would load a local model needing approve_hire_mb of memory or more, and a turn whose calls have cost approve_turn_usd
(and each further approve_turn_usd) wait for the user. A setting of 0 never asks.

The question is asked in-band, so every interface works: the turn's answer says what waits ("Waiting for your
approval: ..."), the waiting work stays paused, and the user's next message in that conversation decides it.
"approve" (or yes, ok) allows what was asked, once; anything else denies it, and a hirer that is denied hears the
user's words as the reason. A question not answered by the turn's deadline is denied, and so is every question in a
session nobody can answer (no conversation, and no terminal: `hashiru ask` in a script).
"""
import asyncio
import itertools
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from hashiru.ledger import free_mb
from hashiru.llm import is_local
from hashiru.ollama import tag

if TYPE_CHECKING:
    from hashiru.hub import Hub
    from hashiru.ledger import AgentRecord
    from hashiru.session import Session

__all__ = ["Approvals", "Decision", "Request", "approves", "hire", "load", "question", "turn"]

log = logging.getLogger(__name__)
YES = {"approve", "approved", "yes", "y", "ok", "okay", "allow", "go ahead"}
NO = {"deny", "denied", "no", "n", "reject", "stop", "cancel"}


def words(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def approves(text: str) -> bool:
    return words(text) in YES


def dollars(usd: float) -> str:
    return f"${usd:.2f}" if usd == 0 or abs(usd) >= 0.1 else f"${usd:.4f}"


@dataclass(frozen=True)
class Request:
    """Something waiting for the user. kind is "api" (a cloud reservation), "gpu" (a local model to load) or "turn"
    (more cloud spending in this turn); agent is who waits."""

    id: str
    kind: str
    agent: str
    what: str

    def public(self) -> dict[str, str]:
        return {"id": self.id, "kind": self.kind, "agent": self.agent, "what": self.what}


@dataclass(frozen=True)
class Decision:
    approved: bool
    said: str = ""                # the user's words, when they said more than yes or no
    labels: tuple[str, ...] = ()  # what those words carry (the user's labels)
    why: str = ""                 # set when nobody decided: "expired" or "unattended"

    @classmethod
    def of(cls, text: str, labels: list[str]) -> "Decision":
        """The user's answer to the questions they were shown."""
        plain = words(text) in YES | NO
        return cls(approves(text), "" if plain else text.strip(), () if plain else tuple(labels))


Answer = Callable[[Request], Awaitable[Decision]]


def question(requests: list[Request]) -> str:
    """What the user is shown while work waits for them."""
    lines = "\n".join(f"- {r.what}" for r in requests)
    this, it = ("this", "it") if len(requests) == 1 else ("these", "them")
    return (f"Waiting for your approval:\n{lines}\n\nReply **approve** to allow {this} once. Anything else denies "
            f"{it}; say why or what to do instead, and the agent that asked hears it.")


class Approvals:
    """One session's questions for the user."""

    def __init__(self) -> None:
        self.attended = False              # a conversation shows the questions and passes the user's answer on
        self.answer: Answer | None = None  # or someone is asked directly (the terminal)
        self.deadline: float | None = None  # the turn's deadline (Unix time)
        self.turn_base = 0.0               # dollars the session had spent when the turn began
        self.turn_steps = 1                # the turn may spend approve_turn_usd times this before asking again
        self.turn_denied = False
        self.planned: dict[str, tuple[str, float]] = {}  # the CEO's hires in an approved plan: name -> (model, usd)
        self._pending: dict[str, tuple[Request, asyncio.Future[Decision]]] = {}  # by key
        self._shown: set[str] = set()
        self._ids = itertools.count(1)

    def new_turn(self, spent: float, deadline: float) -> None:
        self.turn_base, self.turn_steps, self.turn_denied, self.deadline = spent, 1, False, deadline

    def in_plan(self, boss: str, name: str, model: str, budget_usd: float) -> bool:
        """Whether the user approved this hire with a plan: the CEO's, by the same name and model, reserving no more.
        It covers one hire."""
        planned = self.planned.get(name) if boss == "ceo" else None
        if planned is None or planned[0] != model or budget_usd > planned[1] + 1e-9:
            return False
        del self.planned[name]
        return True

    def show(self) -> list[Request]:
        """The questions not shown to the user yet, now marked shown."""
        fresh = [r for r, _ in self._pending.values() if r.id not in self._shown]
        self._shown.update(r.id for r in fresh)
        return fresh

    def decide(self, ids: list[str], decision: Decision) -> None:
        """The user's answer to the questions `ids` (ones already decided, e.g. expired, are left alone)."""
        for request, future in list(self._pending.values()):
            if request.id in ids and not future.done():
                future.set_result(decision)

    def drop(self) -> None:
        """The session is closing: every waiting question is denied."""
        for _, future in list(self._pending.values()):
            if not future.done():
                future.set_result(Decision(False, why="expired"))

    async def ask(self, session: "Session", kind: str, agent: str, what: str, key: str = "") -> Decision:
        """Wait for the user's decision. Callers asking with the same `key` share one question."""
        if key and key in self._pending:
            return await asyncio.shield(self._pending[key][1])
        request = Request(str(next(self._ids)), kind, agent, what)
        future: asyncio.Future[Decision] = asyncio.get_running_loop().create_future()
        slot = key or f"#{request.id}"
        self._pending[slot] = (request, future)
        session.trace.emit("approval", task=session.id, id=request.id, action="asked", approval=kind, agent=agent,
                           what=what)
        try:
            decision = await self._wait(request, future)
        except BaseException:
            if not future.done():
                future.set_result(Decision(False, why="expired"))  # others asking the same question stop too
            raise
        finally:
            self._pending.pop(slot, None)
        action = decision.why or ("approved" if decision.approved else "denied")
        session.trace.emit("approval", task=session.id, id=request.id, action=action, approval=kind, agent=agent,
                           what=what)
        return decision

    async def _wait(self, request: Request, future: "asyncio.Future[Decision]") -> Decision:
        if self.answer is not None:
            future.set_result(await self.answer(request))
        elif not self.attended:
            future.set_result(Decision(False, why="unattended"))
        else:
            left = (self.deadline or time.time()) - time.time()
            try:
                await asyncio.wait_for(asyncio.shield(future), max(left, 0.0))
            except TimeoutError:
                if not future.done():
                    future.set_result(Decision(False, why="expired"))
        return future.result()


def refusal(decision: Decision, what: str) -> dict[str, Any]:
    """What a denied hirer gets: why, with the user's words (and their labels, so R3 applies to them)."""
    if decision.why == "unattended":
        why = f"{what} needs the user's approval, and nobody can approve in this session"
    elif decision.why == "expired":
        why = f"{what} needs the user's approval, and it did not come in time"
    else:
        why = f"the user did not approve: {what}" + (f". They said: {decision.said}" if decision.said else "")
    return {"error": why, "denied": True, "_labels": list(decision.labels)}


async def need_mb(hub: "Hub", model: str) -> int:
    """Memory a local model would newly take: 0 if a live agent already has it loaded, or if Ollama cannot say (the
    load then fails on its own)."""
    if hub.in_use(model):
        return 0
    try:
        return (await hub.ollama.sizes()).get(tag(model), 0)
    except httpx.HTTPError as exc:
        log.warning("could not read %s's size for approval: %s", model, exc)
        return 0


async def load(hub: "Hub", session: "Session", agent: str, model: str, what: str) -> Decision | None:
    """Ask before loading a big local model: None if it may go ahead without asking, else the decision. `what` is
    how the user reads it, without the memory."""
    limit = hub.settings.approve_hire_mb
    if not limit or (mb := await need_mb(hub, model)) < limit:
        return None
    if hub.settings.enforce_budget and mb > free_mb():
        return None  # it cannot load anyway: the memory check says so, without a question first
    return await session.approvals.ask(session, "gpu", agent, f"{what}, loading {mb} MB of memory")


async def hire(hub: "Hub", session: "Session", boss: "AgentRecord", name: str, model: str,
               budget_usd: float) -> dict[str, Any] | None:
    """Ask before a big hire: None if it may go ahead, else the error the hirer gets. A hire in a plan the user
    approved goes ahead without asking again."""
    what = f"{boss.name} wants to hire {name} on {model}"
    if session.approvals.in_plan(boss.name, name, model, budget_usd):
        session.trace.emit("approval", task=session.id, id="plan", action="planned", approval="plan", agent=boss.name,
                           what=f"{what} with {dollars(budget_usd)} reserved")
        return None
    if is_local(model):
        decision = await load(hub, session, boss.name, model, what)
        return None if decision is None or decision.approved else refusal(decision, what)
    limit = hub.settings.approve_hire_usd
    if not limit or budget_usd < limit:
        return None
    what = f"{what} with {dollars(budget_usd)} reserved"
    decision = await session.approvals.ask(session, "api", boss.name, what)
    return None if decision.approved else refusal(decision, what)


async def turn(hub: "Hub", session: "Session", rec: "AgentRecord") -> str | None:
    """Ask before a cloud call once the turn has spent approve_turn_usd (and again at each multiple). Returns why the
    call may not happen, or None. Every agent that waits meanwhile shares the one question."""
    limit, state = hub.settings.approve_turn_usd, session.approvals
    if not limit or rec.tier == "local":
        return None
    while not state.turn_denied:
        spent = sum(a.spent for a in session.ledger.agents.values()) - state.turn_base
        steps = state.turn_steps
        if spent < limit * steps:
            return None
        what = f"this turn has spent {dollars(spent)} on cloud models; continue, up to {dollars(limit * (steps + 1))}"
        decision = await state.ask(session, "turn", rec.name, what, key=f"turn-{steps}")
        if decision.approved:
            state.turn_steps = max(state.turn_steps, steps + 1)
        else:
            state.turn_denied = True
    return f"failed: the user did not approve more spending this turn, so {rec.name} stopped"
