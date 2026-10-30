"""A plan's steps, compiled into the program the hub runs when the user approves the plan (plangraph.py).

A step sends one agent a task and names the steps whose results it needs (`after`). Compiling adds what the CEO
would otherwise have to remember: a hire node for each agent the plan hires (they start at once: propose_plan checked
their money and memory together), a fire node after each hire's last step (its unspent money goes back and a local
model is freed), and one answer node, where the CEO answers with every step's result. Agents hired in earlier turns
can take steps too; they are neither hired nor fired. A step whose agent or inputs are unknown, steps that wait for
each other, and a hire no step uses are plan problems, and such a plan does not compile.
"""
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from hashiru.tools import NAME

__all__ = ["Node", "Program", "Step", "compile_steps", "default_steps", "parse_steps"]


@dataclass(frozen=True)
class Step:
    """One message in a plan: `task` goes to `agent` once the steps in `after` are done, with their results."""

    id: str
    agent: str
    task: str
    after: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "agent": self.agent, "task": self.task, "after": list(self.after)}


@dataclass(frozen=True)
class Node:
    """One node of the program. kind is hire, step, fire or answer; it starts once every node in `waits` is done.
    The id is its LangGraph name: hire-<agent>, step-<id>, fire-<agent> or answer."""

    id: str
    kind: str
    agent: str = ""
    step: Step | None = None
    waits: tuple[str, ...] = ()


@dataclass(frozen=True)
class Program:
    nodes: tuple[Node, ...]

    def steps(self) -> list[Step]:
        """The steps, in the order they can start."""
        return [n.step for n in self.nodes if n.step is not None]


def parse_steps(raw: Any) -> tuple[list[Step], list[str]]:
    """propose_plan's `steps` argument: the steps that are well formed, and what is wrong with the others."""
    if not isinstance(raw, list):
        return [], ["steps must be a list of objects"]
    steps: list[Step] = []
    problems: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            problems.append("each step must be an object with id, agent, task and after")
            continue
        sid, after = str(item.get("id", "")), item.get("after", [])
        if not NAME.fullmatch(sid):
            problems.append(f"step {sid!r}: an id uses only a-z, 0-9 and _ (at most 32)")
        elif any(s.id == sid for s in steps):
            problems.append(f"the plan names step {sid} twice")
        elif not isinstance(after, list):
            problems.append(f"step {sid}: after must be a list of step ids")
        else:
            steps.append(Step(sid, str(item.get("agent", "")), str(item.get("task", "")),
                              tuple(dict.fromkeys(map(str, after)))))
    return steps, problems


def default_steps(hires: Iterable[tuple[str, str]]) -> list[Step]:
    """A plan without steps: each hire (name, task) gets one step, its task, with no inputs."""
    return [Step(name, name, task) for name, task in hires]


def ordered(steps: Sequence[Step]) -> tuple[list[Step], list[Step]]:
    """The steps in an order they can start (inputs first, otherwise as given), and those that never can."""
    ids = {s.id for s in steps}
    done: set[str] = set()
    order: list[Step] = []
    left = list(steps)
    while True:
        ready = [s for s in left if all(d in done or d not in ids for d in s.after)]  # unknown inputs: reported apart
        if not ready:
            return order, left
        order += ready
        done.update(s.id for s in ready)
        left = [s for s in left if s.id not in done]


def compile_steps(steps: Sequence[Step], hired: Sequence[str],
                  live: Collection[str]) -> tuple[Program | None, list[str]]:
    """The program for these steps, given the agents the plan hires and the ones the CEO hired before (`live`), or
    None and the problems that keep it from running. No steps: no program, nothing to run."""
    known, ids = set(hired) | set(live), [s.id for s in steps]
    problems = [f"the plan names step {sid} twice" for sid in sorted({i for i in ids if ids.count(i) > 1})]
    for s in steps:
        if s.agent not in known:
            problems.append(f"step {s.id}: {s.agent or 'no agent'} is neither hired in this plan nor an agent you "
                            "hired before")
        problems += [f"step {s.id} waits for {d}, which is not a step in this plan" for d in s.after if d not in ids]
    order, stuck = ordered(steps)
    if stuck:
        problems.append(f"steps {', '.join(sorted(s.id for s in stuck))} can never start: they wait for each other")
    used = {s.agent for s in steps}
    problems += [f"{name} is hired but no step gives it work" for name in hired if name not in used]
    if problems or not steps:
        return None, problems
    needed = {d for s in steps for d in s.after}
    hires = [Node(f"hire-{name}", "hire", name) for name in hired]
    runs = [Node(f"step-{s.id}", "step", s.agent, s,
                 ((f"hire-{s.agent}",) if s.agent in hired else ()) + tuple(f"step-{d}" for d in s.after))
            for s in order]
    fires = [Node(f"fire-{name}", "fire", name, waits=tuple(f"step-{s.id}" for s in order if s.agent == name))
             for name in hired]
    ends = [f.id for f in fires] + [f"step-{s.id}" for s in order if s.id not in needed and s.agent not in hired]
    return Program((*hires, *runs, *fires, Node("answer", "answer", waits=tuple(ends)))), []
