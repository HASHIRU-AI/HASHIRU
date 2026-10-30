"""An approved plan, run as a LangGraph graph.

Each node of the compiled program (program.py) becomes a graph node, and a node that waits for several others gets
one edge from all of them, so it starts once they are all done. The graph does nothing by itself: a Runner (the CEO,
runtime.PlanRunner) hires, asks, fires and answers, through the same paths the CEO's own tool calls take. A step whose
hire was refused fails; a step that needs a step that failed or was skipped is skipped; everything else still runs,
and the answer gets every outcome.

LangGraph runs in supersteps: the nodes that are ready start together, and the next ones start when all of those are
done. LangSmith tracing stays off, whatever the environment says: it would send every step's result off the machine.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypedDict, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import tracing_context

from hashiru.program import Node, Program, Step

__all__ = ["Outcome", "Runner", "build", "mermaid", "message", "run"]

SKIPPED_BECAUSE = {"failed": "failed", "skipped": "was skipped"}


@dataclass(frozen=True)
class Outcome:
    """What a step came to: done (text is the reply), failed (text says why) or skipped (text says which input)."""

    status: str
    text: str


class Runner(Protocol):
    async def hire(self, agent: str) -> str | None:
        """Hire a planned agent. None when it was hired, else why not."""

    async def ask(self, agent: str, text: str) -> str:
        """Send an agent one message and return its reply."""

    async def fire(self, agent: str) -> None:
        """Fire an agent the graph hired."""

    async def answer(self, results: list[tuple[Step, Outcome]]) -> str:
        """The answer to the user, from every step's outcome."""

    def note(self, node: Node, outcome: Outcome) -> None:
        """Record how a step ended."""


def merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    return old | new


class State(TypedDict):
    results: Annotated[dict[str, Outcome], merge]  # by step id
    hired: Annotated[dict[str, str], merge]        # by agent: "" when hired, else why not
    answer: str


def message(step: Step, results: Mapping[str, Outcome]) -> str:
    """What a step sends: its task, then what each step it builds on returned."""
    if not step.after:
        return step.task
    inputs = "\n\n".join(f"[{d}]\n{results[d].text}" for d in step.after)
    return f"{step.task}\n\nWhat the steps this one builds on returned:\n\n{inputs}"


async def outcome(step: Step, state: State, runner: Runner) -> Outcome:
    if why := state["hired"].get(step.agent):  # agents hired in earlier turns are not in `hired`
        return Outcome("failed", f"{step.agent} was not hired: {why}")
    for dep in step.after:
        if (got := state["results"][dep]).status != "done":
            return Outcome("skipped", f"it needs {dep}, which {SKIPPED_BECAUSE[got.status]}")
    reply = (await runner.ask(step.agent, message(step, state["results"]))).strip()
    if not reply:
        return Outcome("failed", f"{step.agent} gave an empty answer")
    return Outcome("failed" if reply.startswith(("error: ", "failed")) else "done", reply)


def action(program: Program, node: Node, runner: Runner) -> Any:
    """The graph node for one program node."""
    async def hire(state: State) -> dict[str, Any]:
        return {"hired": {node.agent: await runner.hire(node.agent) or ""}}

    async def step(state: State) -> dict[str, Any]:
        assert node.step is not None
        done = await outcome(node.step, state, runner)
        runner.note(node, done)
        return {"results": {node.step.id: done}}

    async def fire(state: State) -> dict[str, Any]:
        if state["hired"].get(node.agent) == "":
            await runner.fire(node.agent)
        return {}

    async def answer(state: State) -> dict[str, Any]:
        return {"answer": await runner.answer([(s, state["results"][s.id]) for s in program.steps()])}

    return {"hire": hire, "step": step, "fire": fire, "answer": answer}[node.kind]


def build(program: Program, runner: Runner) -> CompiledStateGraph[State, None, State, State]:
    graph = StateGraph(State)
    for node in program.nodes:
        graph.add_node(node.id, action(program, node, runner))
        if not node.waits:
            graph.add_edge(START, node.id)
        elif len(node.waits) == 1:
            graph.add_edge(node.waits[0], node.id)
        else:
            graph.add_edge(list(node.waits), node.id)  # a join: it starts once every one of them is done
    graph.add_edge("answer", END)
    return graph.compile()


async def run(program: Program, runner: Runner) -> str:
    """Run the program to the end and return the answer."""
    with tracing_context(enabled=False):  # never send the plan's data to LangSmith (R3), whatever the environment says
        state = await build(program, runner).ainvoke(
            {"results": {}, "hired": {}, "answer": ""},
            {"recursion_limit": len(program.nodes) + 2})  # enough for any program, whatever LangGraph's default
    return str(state["answer"])


def mermaid(program: Program) -> str:
    """The graph the hub would run, as a Mermaid flowchart."""
    return build(program, cast("Runner", None)).get_graph().draw_mermaid()  # drawing runs no node
