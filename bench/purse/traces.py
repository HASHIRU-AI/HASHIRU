"""What a live HASHIRU run did, read from its JSONL trace: who hired whom on which model with what reservation, what
each spent, what was fired and escalated, which hires got which job, and the difficulty each hire was declared at."""
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from purse.jobs import ANSWER_NOTE, Job
from purse.market import Market, Option
from purse.projects import Project
from purse.solvers import RunResult, run_upfront

__all__ = ["Hire", "declared", "escalations", "hires", "jobs_of", "live_allocation", "prompt_fidelity", "read",
           "replay_live", "slack"]

Event = dict[str, Any]


@dataclass(frozen=True)
class Hire:
    name: str
    boss: str
    model: str
    budget_usd: float
    difficulty: str
    spent_usd: float
    fired: bool


def read(path: Path) -> list[Event]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hires(events: Sequence[Event]) -> list[Hire]:
    """Every successful hire in order. A name hired again after a fire is a new hire; calls are charged to the
    latest hire of their agent's name. A hire by difficulty takes its model from its first call or escalation."""
    made: list[dict[str, Any]] = []
    current: dict[str, dict[str, Any]] = {}
    for e in events:
        kind = e.get("kind")
        if kind == "tool" and e.get("name") == "hire_agent" and "error" not in (e.get("result") or {}):
            args = e.get("args") or {}
            h = {"name": args.get("name", "?"), "boss": e.get("agent", ""), "model": args.get("model") or "",
                 "budget_usd": float(args.get("budget_usd") or 0), "difficulty": args.get("difficulty") or "",
                 "spent_usd": 0.0, "fired": False}
            made.append(h)
            current[h["name"]] = h
        elif kind == "llm" and (cur := current.get(e.get("agent", ""))) is not None:
            cur["spent_usd"] += float(e.get("cost_usd") or 0)
            cur["model"] = cur["model"] or e.get("model", "")
        elif kind == "escalate" and (cur := current.get(e.get("agent", ""))) is not None:
            cur["model"] = cur["model"] or e.get("from_model", "")
        elif kind == "tool" and e.get("name") == "fire_agent":
            for name in (e.get("result") or {}).get("fired", []):
                if name in current:
                    current[name]["fired"] = True
    return [Hire(**h) for h in made]


def escalations(events: Sequence[Event]) -> list[tuple[str, str, str]]:
    return [(e["agent"], e["from_model"], e["to_model"]) for e in events if e.get("kind") == "escalate"]


def _task(job: Job) -> str:
    return job.prompt.removesuffix(f"\n\n{ANSWER_NOTE}")


def _to_hires(events: Sequence[Event]) -> list[tuple[str, str]]:
    """(hire name, text) for every message a boss sent a hire, and every hire's instructions."""
    out = []
    for e in events:
        if e.get("kind") == "msg" and not e.get("reply") and e.get("to") not in (None, "ceo", "user"):
            out.append((e["to"], e.get("text", "")))
        elif e.get("kind") == "tool" and e.get("name") == "hire_agent" and "error" not in (e.get("result") or {}):
            out.append(((e.get("args") or {}).get("name", "?"), (e.get("args") or {}).get("prompt", "")))
    return out


def jobs_of(events: Sequence[Event], p: Project) -> dict[str, list[str]]:
    """Job id -> the hires that were sent its task text or its id, in order."""
    out: dict[str, list[str]] = {}
    for name, text in _to_hires(events):
        for j in p.jobs:
            if (_task(j) in text or re.search(rf"\b{re.escape(j.id)}\b", text, re.I)) and name not in out.get(j.id, []):
                out.setdefault(j.id, []).append(name)
    return out


def live_allocation(events: Sequence[Event], p: Project) -> dict[str, list[str]]:
    """Job id -> every model tried on it: each of its hires' models, then where each escalated to."""
    made = {h.name: h for h in hires(events)}
    moves = escalations(events)
    out: dict[str, list[str]] = {}
    for job_id, names in jobs_of(events, p).items():
        for name in names:
            if name in made and made[name].model:  # a hire that never ran a model tried nothing
                out.setdefault(job_id, []).append(made[name].model)
                out[job_id] += [to for agent, _, to in moves if agent == name]
    return out


def declared(events: Sequence[Event], p: Project) -> dict[str, str]:
    """Job id -> the difficulty its first hire was declared at ("" when the hirer named a model)."""
    made = {h.name: h for h in hires(events)}
    return {job_id: made[names[0]].difficulty for job_id, names in jobs_of(events, p).items() if names[0] in made}


def prompt_fidelity(events: Sequence[Event], p: Project) -> float:
    """Share of jobs whose canonical task text reached some hire verbatim."""
    texts = [text for _, text in _to_hires(events)]
    return sum(any(_task(j) in t for t in texts) for j in p.jobs) / len(p.jobs) if p.jobs else 0.0


def slack(events: Sequence[Event]) -> float:
    """Reserved minus spent, over the hires that were fired: the money refunds returned."""
    return sum(h.budget_usd - h.spent_usd for h in hires(events) if h.fired)


async def replay_live(p: Project, tried: Mapping[str, Sequence[str]], market: Market,
                      options: Sequence[Option]) -> RunResult:
    """HASHIRU's decisions scored on the outcome table: each job on the last model it was tried on."""
    return await run_upfront(p, {job_id: models[-1] for job_id, models in tried.items() if models}, market, options)
