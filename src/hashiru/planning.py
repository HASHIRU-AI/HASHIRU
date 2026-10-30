"""Plan and act modes.

In plan mode the CEO researches and proposes but does not act. It is offered only the tools that declare MCP's
readOnlyHint (and memory, for search and list) plus propose_plan, and runtime.dispatch refuses anything else in code,
whatever the model calls. propose_plan previews each hire as hire_agent would make it (the router's model, the
reservation, the memory a local model takes) and checks the totals against the CEO's pool, without hiring or
reserving anything. The plan's steps (program.py) say who gets which task and which results each step needs; under
plan_execution=graph they compile into the graph the hub runs on approval (plangraph.py). The turn's answer shows the
plan under the CEO's reply and the plan waits: the user's "approve" carries it out in act mode, and approving the plan
approves its hires (Approvals.planned); anything else is feedback for another planning turn, and act mode drops the
plan.

A message starting with /plan or /act picks its own mode. Otherwise the client's mode applies (the chat page's
switch, x-hashiru-mode, A2A hashiru_mode, MCP ask mode), else plan while a plan waits, else default_mode.
"""
import math
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from hashiru.approvals import approves, dollars, need_mb
from hashiru.config import DIFFICULTIES
from hashiru.ledger import AgentRecord, free_mb
from hashiru.llm import is_local
from hashiru.plangraph import mermaid
from hashiru.privacy import may_hold
from hashiru.program import Program, Step, compile_steps, default_steps, parse_steps
from hashiru.tools import NAME, RESERVED, ROLES

if TYPE_CHECKING:
    from hashiru.hub import Hub
    from hashiru.session import Session

__all__ = ["APPROVE", "MODES", "PLAN_RULES", "PROPOSE_TOOL", "Hire", "Plan", "begin", "carried_out", "carry_out",
           "end", "propose", "readable", "refusal", "split"]

MODES = ("act", "plan")
PREFIX = re.compile(r"\s*/(plan|act)\s+(\S.*)", re.IGNORECASE | re.DOTALL)
READ_ACTIONS = {"memory": ("search", "list")}  # tools that read with some actions and write with others
APPROVE = "Reply **approve** to carry out this plan, or say what to change."

PLAN_RULES = (
    "You are in plan mode: research the request and propose a plan, but do not act yet. You can use only tools that "
    "read (such as check_usage and memory search); hiring, messaging agents and everything else are refused until "
    "the user approves your plan. Call propose_plan with each agent you would hire (name, role, difficulty or model, "
    "budget_usd and its task), the steps (each step sends one agent a task; after names the steps whose results it "
    "needs, and they are added to its message) and how you will combine their work; if it reports problems, fix "
    "them and call it again. When the user approves, the steps run in that order, at the same time where they can, "
    "and you answer with their results. Then reply with the plan in a few sentences. The hub shows the user each "
    "hire, its model and what it reserves under your reply, and the user approves the plan or says what to change. "
    "If you proposed a plan earlier in the conversation, revise it with what the user said. A request that needs no "
    "agents gets a plan with no hires.")

PROPOSE_TOOL = {"type": "function", "function": {
    "name": "propose_plan",
    "description": ("Propose your plan for the user to approve. The hub previews each hire as hire_agent would make "
                    "it (model, reservation, memory) and checks the totals against your pool; nothing is hired or "
                    "reserved."),
    "parameters": {"type": "object", "properties": {
        "summary": {"type": "string", "description": "How you will split the request and combine the agents' work."},
        "hires": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "role": {"type": "string", "enum": list(ROLES)},
            "difficulty": {"type": "string", "enum": list(DIFFICULTIES)},
            "model": {"type": "string", "description": "Only when the job needs this model."},
            "budget_usd": {"type": "number"}, "task": {"type": "string", "description": "What it will do."},
            "prompt": {"type": "string", "description": "Its instructions when hired; default: its task."},
            "tools": {"type": "array", "items": {"type": "string"}, "description": "A manager's tools, if limited."}},
            "required": ["name", "role", "task"]}},
        "steps": {"type": "array", "description": "The work, in steps. Default: each hire gets its task once.",
                  "items": {"type": "object", "properties": {
                      "id": {"type": "string", "description": "a-z, 0-9 and _"},
                      "agent": {"type": "string", "description": "A hire in this plan, or an agent you hired before."},
                      "task": {"type": "string", "description": "The message it gets."},
                      "after": {"type": "array", "items": {"type": "string"},
                                "description": "Steps whose results it needs."}},
                      "required": ["id", "agent", "task"]}}},
        "required": ["summary", "hires"]}}}


@dataclass(frozen=True)
class Hire:
    """One hire of a plan, as hire_agent would make it now."""

    name: str
    role: str
    model: str
    tier: str
    budget_usd: float
    gpu_mb: int
    task: str
    difficulty: str = ""  # set when the router picked the model
    tools: tuple[str, ...] | None = None
    prompt: str = ""  # its instructions when hired; empty: its task

    def public(self) -> dict[str, Any]:
        return {"name": self.name, "role": self.role, "model": self.model, "tier": self.tier,
                "difficulty": self.difficulty, "budget_usd": self.budget_usd, "gpu_mb": self.gpu_mb,
                "task": self.task} | ({"tools": list(self.tools)} if self.tools is not None else {}) | (
            {"prompt": self.prompt} if self.prompt else {})


@dataclass(frozen=True)
class Plan:
    """What the CEO proposed. `proposed` is False when it only described a plan in its reply. `program` is set when
    the hub will run the steps as a graph (plan_execution=graph and they compile)."""

    summary: str = ""
    hires: tuple[Hire, ...] = ()
    problems: tuple[str, ...] = ()
    left_usd: float = 0.0
    free_mb: int = 0
    proposed: bool = True
    steps: tuple[Step, ...] = ()
    program: Program | None = None

    @property
    def total_usd(self) -> float:
        return sum(h.budget_usd for h in self.hires)

    @property
    def total_mb(self) -> int:
        return sum({h.model: h.gpu_mb for h in self.hires}.values())  # one local model loads once

    def public(self) -> dict[str, Any]:
        return {"summary": self.summary, "hires": [h.public() for h in self.hires],
                "total_usd": round(self.total_usd, 6), "total_mb": self.total_mb, "left_usd": round(self.left_usd, 6),
                "free_mb": self.free_mb, "problems": list(self.problems), "steps": [s.public() for s in self.steps]}

    def covers(self) -> dict[str, tuple[str, float]]:
        """The hires approving this plan approves: name -> (model, most it may reserve)."""
        return {h.name: (h.model, h.budget_usd) for h in self.hires}

    def table(self) -> str:
        """The plan as the user reads it under the CEO's reply (Markdown)."""
        if not self.proposed:
            return ""
        if not self.hires:
            lines = ["No new hires." if self.steps else "No hires: the CEO answers by itself."]
        else:
            lines = ["| Hire | Role | Model | Reserves | Memory | Task |", "|---|---|---|---|---|---|"]
            lines += [f"| {h.name} | {h.role} | {h.model}{f' ({h.difficulty})' if h.difficulty else ''} | "
                      f"{dollars(h.budget_usd)} | {f'{h.gpu_mb} MB' if h.gpu_mb else 'none'} | {cell(h.task)} |"
                      for h in self.hires]
            memory = f" and {self.total_mb} MB of memory ({self.free_mb} MB free)" if self.total_mb else ""
            lines += ["", f"Reserves {dollars(self.total_usd)} of the {dollars(self.left_usd)} left in the CEO's "
                          f"pool{memory}. Approving the plan approves these hires."]
        if self.steps != tuple(default_steps((h.name, h.task) for h in self.hires)):  # not just each hire's task
            lines += ["", "| Step | Agent | After | Task |", "|---|---|---|---|"]
            lines += [f"| {s.id} | {s.agent} | {', '.join(s.after)} | {cell(s.task)} |" for s in self.steps]
        if self.program is not None:
            lines += ["", "Approving runs this plan as a graph: the hires, each step once the steps it needs are done "
                          "(at the same time where they can), each hire fired after its last step, then the CEO "
                          "answers with every result."]
        if self.problems:
            lines += ["", "Problems the hub found:", *(f"- {p}" for p in self.problems)]
        return "\n".join(lines)


def cell(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split()).replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def split(text: str) -> tuple[str | None, str]:
    """A message's own mode (/plan ... or /act ...) and its words without the prefix."""
    found = PREFIX.fullmatch(text)
    return (found.group(1).lower(), found.group(2).strip()) if found else (None, text)


def readable(tool: Any) -> bool:
    """Whether plan mode offers an MCP tool: it declares readOnlyHint, or it reads with some of its actions."""
    hints = getattr(tool, "annotations", None)
    return tool.name in READ_ACTIONS or bool(hints and hints.read_only_hint)


async def refusal(hub: "Hub", name: str, args: dict[str, Any]) -> str | None:
    """Why plan mode refuses this call, or None. Checked against the hub's own tool list, whatever was offered."""
    if name in READ_ACTIONS:
        allowed = READ_ACTIONS[name]
        action = args.get("action")
        return None if action in allowed else (f"{name} {action} is not allowed while planning: only "
                                               f"{' and '.join(allowed)} until the user approves a plan")
    tools = {t.name: t for t in await hub.mcp.list_tools()}
    if name in tools and readable(tools[name]):
        return None
    return (f"{name} is not allowed while planning: until the user approves a plan you may only read, and propose "
            "the plan with propose_plan")


def begin(hub: "Hub", session: "Session", text: str, mode: str | None = None) -> str:
    """Start a turn in plan or act mode and return the message for the CEO.

    A waiting plan that the user approves is carried out, with its hires approved; otherwise it is dropped, and a
    planning turn revises it with the user's words.
    """
    own, words = split(text)
    mode = own or mode
    waiting, session.plan, session.approved = session.plan, None, None  # an approved plan runs in this turn only
    session.approvals.planned = {}
    if waiting is not None and approves(words):
        session.planning = False
        session.approvals.planned = waiting.covers()
        if waiting.program is not None:
            session.approved = waiting  # the CEO's run carries it out as a graph (runtime.run_agent)
            return carried_out(waiting)
        return carry_out(waiting)
    session.planning = mode == "plan" or (mode is None and (waiting is not None or
                                                            hub.settings.default_mode == "plan"))
    return hub.recall(session, words)


def carry_out(plan: Plan) -> str:
    """What the CEO is told when the user approves its plan."""
    if not plan.proposed:
        return "The user approved the plan you proposed above. Carry it out now, then answer their request."
    hires = [f"- {h.name}: a {h.role}, "
             + (f"difficulty={h.difficulty} (planned on {h.model})" if h.difficulty else f"model={h.model}")
             + f", budget_usd={h.budget_usd:g}" + (f", tools={list(h.tools)}" if h.tools is not None else "")
             + f". Its task: {h.task}" for h in plan.hires]
    steps = "\n".join(f"- {s.id}: send {s.agent}: {s.task}" + (f" (after {', '.join(s.after)}, with their results)"
                                                                 if s.after else "") for s in plan.steps)
    return ("The user approved your plan. Carry it out now: make these hires as planned (with the same names, models "
            "and budgets they need no further approval), give them their work, and answer the user's request.\n"
            + ("\n".join(hires) or "- no hires: answer by yourself")
            + (f"\nThe steps, in order:\n{steps}" if steps else "")
            + f"\nHow you planned to combine it: {plan.summary}")


def carried_out(plan: Plan) -> str:
    """What the CEO is told when the hub has run its approved plan as a graph; each step's result follows."""
    return ("The user approved your plan and the hub carried it out: it made your hires, ran each step once the steps "
            "it needs were done, and fired each hire after its last step. Answer the user's request with the results "
            f"below, in your own words.\nHow you planned to combine it: {plan.summary}")


def end(session: "Session", reply: str) -> tuple[str, dict[str, Any] | None]:
    """After a turn: what the user is shown and, after a planning turn, the plan that now waits (as they see it)."""
    if not session.planning:
        return reply, None
    session.planning = False
    if reply.startswith(("error: ", "failed")):
        session.plan = None
        return reply, None
    plan = session.plan = session.plan or Plan(proposed=False)
    table, public = plan.table(), plan.public() | ({"graph": mermaid(plan.program)} if plan.program else {})
    if not may_hold(session.ledger.agents["ceo"].tier):  # a cloud CEO wrote placeholders; the user reads the values
        table, _ = session.vault.unmask(table)
        public, _ = session.vault.unmask_value(public)
    return "\n\n".join(filter(None, [reply, table, APPROVE])), public


async def propose(hub: "Hub", session: "Session", me: str, args: dict[str, Any]) -> dict[str, Any]:
    """The propose_plan tool: preview the hires without making them, and keep the plan for the user to approve."""
    boss = session.ledger.agents[me]
    raw = args.get("hires", [])
    if not isinstance(raw, list):
        return {"error": "hires must be a list of objects"}
    hires: list[Hire] = []
    problems: list[str] = []
    seen: set[str] = set()
    for item in raw:
        hire = await preview(hub, session, boss, item if isinstance(item, dict) else {}, seen)
        if isinstance(hire, Hire):
            hires.append(hire)
        else:
            problems.append(hire)
    hired = [h.name for h in hires]
    if "steps" in args:
        steps, wrong = parse_steps(args["steps"])
    else:
        steps, wrong = default_steps((h.name, h.task) for h in hires), []
    program, wrong_too = compile_steps(steps, hired, [a.name for a in session.live_hires(me)])
    problems += wrong + wrong_too
    plan = Plan(str(args.get("summary", "")), tuple(hires), (), session.ledger.left()[me], free_mb(),
                steps=tuple(steps), program=program if hub.settings.plan_execution == "graph" else None)
    if raw and not hub.settings.allow_hiring:
        problems.append("hiring is off on this hub (allow_hiring=false): plan to answer without agents")
    if hub.settings.enforce_budget:
        margin = 0.0 if boss.tier == "local" else hub.settings.turn_reserve_usd
        if plan.left_usd + 1e-9 < plan.total_usd + margin:
            keep = f" and keep {dollars(margin)} for your own turns" if margin else ""
            problems.append(f"the hires reserve {dollars(plan.total_usd)} in all, but your pool has "
                            f"{dollars(plan.left_usd)} left{keep}")
        if plan.total_mb > plan.free_mb:
            problems.append(f"the local hires need {plan.total_mb} MB of memory and {plan.free_mb} MB is free")
    session.plan = plan = replace(plan, problems=tuple(problems))
    then = ("fix the problems and call propose_plan again, or explain them to the user" if problems else
            "reply to the user with the plan; the hub shows them these hires and costs under your reply")
    return plan.public() | {"next": then}


async def preview(hub: "Hub", session: "Session", boss: AgentRecord, item: dict[str, Any],
                  seen: set[str]) -> Hire | str:
    """One planned hire as hire_agent would make it now, or the problem it would hit."""
    name, role = str(item.get("name", "")), str(item.get("role", ""))
    if name in seen:
        return f"the plan names {name} twice"
    seen.add(name)
    if not NAME.fullmatch(name) or name in RESERVED:
        return f"{name!r}: a name must not be one of {sorted(RESERVED)} and must use only a-z, 0-9 and _"
    if session.ledger.get(name) is not None:
        return f"{name} is taken: an agent by that name was hired already"
    if role not in ROLES:
        return f"{name}: role must be 'manager' or 'worker'"
    try:
        budget = float(item.get("budget_usd") or 0)
    except (TypeError, ValueError):
        budget = math.nan
    if not (math.isfinite(budget) and budget >= 0):
        return f"{name}: budget_usd must be a finite number >= 0"
    model, difficulty = str(item.get("model") or ""), str(item.get("difficulty") or "")
    if model:
        if model not in hub.hireable():
            return f"{name}: unknown or disabled model {model!r}; choose from {hub.hireable()}"
        difficulty = ""
    elif difficulty in DIFFICULTIES:
        model = await route(hub, session, boss, difficulty, budget)
        if not model:
            return (f"{name}: no model you may use fits a {difficulty} job (check the modes, and give cloud models a "
                    "budget_usd above 0)")
    else:
        return f"{name}: say how hard its job is with difficulty (easy, medium or hard), or name a model"
    peer = hub.peer(model)
    if peer is not None and role != "worker":
        return f"{name}: a remote agent can only be hired as a worker"
    tier = "local" if is_local(model) else "remote" if peer else "cloud"
    tools = item.get("tools")
    scope = tuple(sorted(map(str, tools))) if isinstance(tools, list) and role == "manager" else None
    return Hire(name, role, model, tier, budget, await need_mb(hub, model) if tier == "local" else 0,
                str(item.get("task", "")), difficulty, scope, str(item.get("prompt") or ""))


async def route(hub: "Hub", session: "Session", boss: AgentRecord, difficulty: str, budget: float) -> str:
    """The model the router would hire on now: its first candidate that fits in free memory, or ''."""
    for model in await hub.router.candidates(session, boss, difficulty, budget):
        if is_local(model) and await hub.fits(model):
            continue
        return model
    return ""
