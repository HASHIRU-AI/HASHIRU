"""PURSE through a live HASHIRU hub: the hub's settings for a project, a Market that hires through the hub as the
CEO (calibration and live replays of the solvers), HASHIRU's own run (agents allocate during the task), and the
up-front LLM split (a plan in plan mode, turned into a fixed allocation)."""
import contextlib
import re
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from hashiru import planning
from hashiru.config import A2APeer, McpServer, Settings
from hashiru.hub import Hub
from hashiru.llm import is_local
from hashiru.planning import Plan
from hashiru.runtime import ask_agent, bearer, perform
from hashiru.session import USER, Session
from purse.jobs import WORKER_PROMPT, Job, family
from purse.market import CEO_MODEL, PROVIDER_ERROR, Option, Outcome
from purse.projects import Project, parse_answers
from purse.solvers import Allocation
from purse.table import Stats

__all__ = ["PORT", "HubMarket", "LiveRun", "bench_settings", "plan_to_allocation", "propose_split", "run_hashiru",
           "success_rates", "variant_text"]

Rates = tuple[tuple[str, tuple[tuple[str, float, float], ...]], ...]

PORT = 9250
BENCH = Path(__file__).resolve().parents[1]
PRICES = BENCH / "prices.json"  # prices for market models LiteLLM does not price (the small tier)
VARIANTS = {"": "", "names-models": "Name a model for every hire; do not give a difficulty.",
            "split": "Plan one worker per job you delegate, named after the job id."}


def bench_settings(p: Project, market: Sequence[Option], *, out_dir: Path, max_usd: float, port: int = PORT,
                   max_escalations: int = 2, peer_urls: Mapping[str, str] | None = None,
                   ceo_model: str = CEO_MODEL, turn_reserve_usd: float = 0.01, max_turns: int = 40,
                   success_rates: Rates = (), by_reference: bool = False,
                   endpoints: Mapping[str, str] | None = None) -> Settings:
    """The hub for one project: its budget, memory cap and deadline; the market's models and outside agents; the
    grid_check tool; no web tools and no approvals (nobody is there to answer). Where the CEO never thinks
    (calibration, live replays of the solvers), turn_reserve_usd=0 lets a hire take the whole pool. A project needs a
    hire, an ask and a fire for up to eight jobs, so max_turns is 40 (chat's 12 ran out in the first live runs).
    `success_rates` lets hires route on measured success by kind of job; `by_reference` holds long sections of the
    project back from the CEO and attaches what a message names (the roadmap's two ablations). `endpoints` sends each
    open-weight model to the vLLM server it runs on."""
    urls = peer_urls or {}
    peers = tuple(A2APeer(o.name.removeprefix("a2a/"), urls[o.name], price_usd_per_message=o.per_message_usd)
                  for o in market if o.kind == "agent" and o.name in urls)
    grid = McpServer("purse", command=sys.executable, args=("-m", "purse.peers.grid_check"),
                     env=(("PYTHONPATH", str(BENCH)),), labels=("trusted",), egress=False)
    return Settings(ceo_model=ceo_model, models=tuple(o.name for o in market if o.kind != "agent"), a2a_peers=peers,
                    mcp_servers=(grid,), budget_usd=p.budget_usd, max_usd=max_usd, memory_mb=p.memory_mb,
                    task_timeout=p.deadline_s, web_tools=False, approve_hire_usd=0, approve_hire_mb=0,
                    approve_turn_usd=0, max_escalations=max_escalations, turn_reserve_usd=turn_reserve_usd,
                    max_turns=max_turns, success_rates=success_rates, request_by_reference=by_reference,
                    trace_dir=str(out_dir / "traces"),
                    data_dir=str(out_dir / "data"), port=port, obfuscate_with="patterns", prices_file=str(PRICES),
                    endpoints=tuple((endpoints or {}).items()))


def success_rates(stats: Stats) -> Rates:
    """The hub's success_rates setting from the statistics an informed allocator is told: per (family, level), each
    option's success rate and mean cost. Hidden-bin cells are not told, so they are left out."""
    kinds: dict[str, list[tuple[str, float, float]]] = {}
    for (fam, level, hidden, option), s in sorted(stats.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3])):
        if hidden is None:
            kinds.setdefault(f"{fam} level {level}", []).append((option, s.rate, s.cost_usd))
    return tuple((kind, tuple(rows)) for kind, rows in kinds.items())


class HubMarket:
    """A Market that hires one worker per attempt through the hub, as the session's CEO (its own token, the hub's
    /mcp), asks it the job's canonical prompt and fires it. A refused hire is a refused attempt. Each cloud hire
    reserves `reserve_usd`, or with None all the CEO has left (attempts run one at a time, and firing refunds)."""

    def __init__(self, hub: Hub, session: Session, reserve_usd: float | None) -> None:
        self.hub, self.session, self.reserve_usd = hub, session, reserve_usd
        self._stack = contextlib.AsyncExitStack()
        self._mcp: Any = None
        self._n = 0

    async def __aenter__(self) -> "HubMarket":
        timeout = httpx2.Timeout(30, read=self.hub.settings.task_timeout)
        http = await self._stack.enter_async_context(
            httpx2.AsyncClient(headers=bearer(self.session, "ceo"), timeout=timeout))
        read, write, *_ = await self._stack.enter_async_context(
            streamable_http_client(f"{self.hub.settings.internal_url}/mcp", http_client=http))
        self._mcp = await self._stack.enter_async_context(ClientSession(read, write))
        await self._mcp.initialize()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._stack.aclose()

    async def attempt(self, job: Job, option: str) -> Outcome:
        self.hub.wallet.check()  # past max_usd every call would be refused: stop rather than record failures
        self._n += 1
        name, start = f"w{self._n}", time.monotonic()
        left = max(0.0, self.session.ledger.left()["ceo"] - self.hub.settings.turn_reserve_usd)
        reserve = left if self.reserve_usd is None else self.reserve_usd
        args = {"name": name, "role": "worker", "model": option, "prompt": WORKER_PROMPT,
                "budget_usd": 0.0 if is_local(option) else reserve}
        hired = await perform(self.hub, self.session, self._mcp, "ceo", "hire_agent", args)
        if "error" in hired:
            return Outcome("", False, 0.0, time.monotonic() - start, refused=str(hired["error"]))
        try:
            answer = await ask_agent(self.hub, self.session, "ceo", name, job.prompt)
            spent = self.session.ledger.agents[name].spent
        finally:
            await perform(self.hub, self.session, self._mcp, "ceo", "fire_agent", {"name": name})
        if answer.startswith(f"error: {name} failed:"):  # the call raised (a 4xx/5xx, a timeout): no answer of its
            return Outcome("", False, spent, time.monotonic() - start, refused=PROVIDER_ERROR + answer)
        return Outcome(answer, family(job.family).correct(job.answer, answer), spent, time.monotonic() - start)


@dataclass(frozen=True)
class LiveRun:
    answers: dict[str, str]
    spent_usd: float  # everything the session spent, the CEO's own calls included
    ceo_usd: float  # the allocator's overhead: the CEO's own calls
    trace: str
    reply: str


async def run_hashiru(hub: Hub, p: Project, text: str) -> LiveRun:
    """HASHIRU allocating during the task: one act turn from the user to the CEO."""
    session = hub.new_session("purse")
    try:
        reply = await ask_agent(hub, session, USER, "ceo", planning.begin(hub, session, text, "act"))
        agents = session.ledger.agents
        return LiveRun(parse_answers(reply), sum(a.spent for a in agents.values()), agents["ceo"].spent,
                       str(session.trace.path), reply)
    finally:
        await hub.close_session(session)


def plan_to_allocation(plan: Plan, p: Project, ceo_model: str = CEO_MODEL) -> Allocation:
    """A hire named after a job (case-insensitive) gives that job its model; the CEO answers the rest itself."""
    planned = {h.name.strip().lower(): h.model for h in plan.hires}
    return {j.id: planned.get(j.id, ceo_model) for j in p.jobs}


async def propose_split(hub: Hub, p: Project, text: str) -> tuple[Allocation, float]:
    """The up-front LLM split: one plan-mode turn; the plan's hires become a fixed allocation. Returns it with the
    CEO's spend on planning."""
    session = hub.new_session("split")
    try:
        reply = await ask_agent(hub, session, USER, "ceo", planning.begin(hub, session, text, "plan"))
        planning.end(session, reply)
        alloc = (plan_to_allocation(session.plan, p, hub.settings.ceo_model) if session.plan is not None
                 else {j.id: hub.settings.ceo_model for j in p.jobs})
        return alloc, session.ledger.agents["ceo"].spent
    finally:
        await hub.close_session(session)


def variant_text(base: str, variant: str) -> str:
    """The project text for a HASHIRU variant: "" as is, "names-models" (v1's way), "split" (the up-front plan). The
    instruction goes before the first job's heading, so it is not part of a job's section (project_text)."""
    if not VARIANTS[variant]:
        return base
    first = re.search(r"^#+ ", base, re.MULTILINE)
    if not first:
        return f"{base}\n\n{VARIANTS[variant]}"
    return f"{base[:first.start()]}{VARIANTS[variant]}\n\n{base[first.start():]}"
