"""Hiring by difficulty: the hiring agent says how hard a job is, and the router turns that into the models the hire
may run on, best first. Where success rates have been measured for a kind of job (`success_rates`), the agent may
give the kind instead: the hire goes to the cheapest model that succeeds at least `route_min_rate` of the time, then
to the others, most reliable first.

The order comes from the `routing` setting or, for a difficulty it does not list, from `models`: an easy job tries
small local models first, then cheap cloud ones; a medium job large local models, then cheap cloud ones; a hard job
the dearest cloud models first, then large local ones. The list keeps only what the modes allow and what the hire's
privacy and pool allow: a hirer holding private data hires only local models in block mode and local ones first when
obfuscating, and a pool of $0 cannot pay a cloud model. A worker that fails moves up its list (Hub.escalate).
"""
import logging
import re
from typing import TYPE_CHECKING

import httpx

from hashiru import approvals
from hashiru.config import DIFFICULTIES
from hashiru.ledger import AgentRecord
from hashiru.llm import is_local
from hashiru.ollama import tag
from hashiru.peers import PEER_PREFIX, model_price
from hashiru.privacy import holds_private
from hashiru.session import Session

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["DIFFICULTIES", "Router", "escalate", "kind_key"]

log = logging.getLogger(__name__)


class Router:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub

    async def candidates(self, session: Session, boss: AgentRecord, difficulty: str, budget_usd: float) -> list[str]:
        """The models a hire by `boss` for a `difficulty` job may run on, best first (empty if none may)."""
        table = dict(self.hub.settings.routing)
        order = list(table[difficulty]) if difficulty in table else await self.default(difficulty)
        return self.allowed(session, boss, order, budget_usd)

    def kinds(self) -> list[str]:
        """The kinds of job with measured success rates, as configured."""
        return [kind for kind, _ in self.hub.settings.success_rates]

    def by_rates(self, session: Session, boss: AgentRecord, kind: str, budget_usd: float) -> list[str]:
        """The models for a `kind` of job by measured success: the cheapest of those at or above route_min_rate
        first, then the rest, most reliable first. Empty for a kind without measurements."""
        rates = {kind_key(k): rows for k, rows in self.hub.settings.success_rates}.get(kind_key(kind), ())
        floor = self.hub.settings.route_min_rate
        good = sorted((r for r in rates if r[1] >= floor), key=lambda r: (r[2], -r[1]))
        rest = sorted((r for r in rates if r[1] < floor), key=lambda r: -r[1])
        return self.allowed(session, boss, [m for m, _, _ in good + rest], budget_usd)

    def allowed(self, session: Session, boss: AgentRecord, order: list[str], budget_usd: float) -> list[str]:
        """`order` minus what the modes, the hire's pool and the hirer's private data rule out."""
        allowed = set(self.hub.hireable())
        order = [m for m in order if m in allowed]
        if self.hub.settings.enforce_budget and budget_usd <= 0:
            order = [m for m in order if is_local(m)]  # a pool of $0 cannot pay for a single cloud call
        if holds_private(session, boss.name):
            local = [m for m in order if is_local(m)]
            order = local if self.hub.obfuscating() is None else local + [m for m in order if not is_local(m)]
        return order

    async def default(self, difficulty: str) -> list[str]:
        models = [m for m in self.hub.settings.models if not m.startswith(PEER_PREFIX)]
        try:
            sizes = await self.hub.ollama.sizes()
        except httpx.HTTPError as exc:  # Ollama down: local models keep their configured order
            log.warning("could not read local model sizes for routing: %s", exc)
            sizes = {}
        local = sorted((m for m in models if is_local(m)), key=lambda m: sizes.get(tag(m), 0))  # small first
        cloud = sorted((m for m in models if not is_local(m)), key=price)                      # cheap first
        return {"easy": local + cloud, "medium": local[::-1] + cloud, "hard": cloud[::-1] + local[::-1]}[difficulty]


def kind_key(kind: str) -> str:
    """'Arithmetic, level 4' and 'arithmetic level 4' are one kind."""
    return " ".join(re.sub(r"[^\w\s]", " ", kind).casefold().split())


def price(model: str) -> float:
    p = model_price(model)
    return p.per_message_usd + p.per_input_token_usd + p.per_output_token_usd


async def escalate(hub: "Hub", session: Session, rec: AgentRecord, why: str) -> bool:
    """Move a worker that failed (or answered empty) to the next model on its route that it may use, and say so in the
    trace. A worker holding private data stays local; a cloud model needs money left in its pool; a local one must fit
    in free memory and load, and a big one the user's approval. Returns whether it moved."""
    if rec.role != "worker" or rec.model not in rec.route:
        return False
    for model in rec.route[rec.route.index(rec.model) + 1:]:
        local = is_local(model)
        if model.startswith(PEER_PREFIX) or model not in hub.hireable():
            continue
        if not local and (holds_private(session, rec.name)
                          or (hub.settings.enforce_budget and session.ledger.left()[rec.name] <= 0)):
            continue
        gpu_mb = 0
        if local:
            asked = await approvals.load(hub, session, rec.name, model, f"{rec.name} failed on {rec.model}; move it to "
                                                                        f"{model}")
            if asked is not None and not asked.approved:
                continue
            try:
                async with hub.memory_lock:
                    if await hub.fits(model):
                        continue
                    async with hub.model_lock(model):
                        gpu_mb = await hub.ollama.load(model)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                log.warning("could not move %s to %s: %s", rec.name, model, exc)
                continue
        old = rec.model
        rec.model, rec.tier, rec.gpu_mb = model, "local" if local else "cloud", gpu_mb
        hub.unmount(session, rec.name)
        hub.mount(session, rec)  # its card names the new model
        await hub.release_models({old})
        session.trace.emit("escalate", task=session.id, agent=rec.name, from_model=old, to_model=model,
                           why=why[:300])
        return True
    return False
