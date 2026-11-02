"""An outside agent for PURSE's grid jobs: it reads the last logic-grid puzzle in a message, solves it exactly and
answers with the JSON solution. Like the example calc agent, it needs no LLM, declares its price through HASHIRU's
cost extension and reports what each request cost. With --wrong it is an unreliable seller (PURSE's
gridfast): it answers every puzzle just as surely, but swaps two houses' values in that share of them, chosen by
the puzzle so the same puzzle always gets the same reply.

    uv run python -m purse.peers.csp_agent --port 9302 --price 0.001
    uv run python -m purse.peers.csp_agent --port 9303 --price 0.00025 --name gridfast --wrong 0.4
"""
import argparse
import hashlib
import json

import uvicorn
from a2a.helpers import get_message_text, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentExtension,
    AgentInterface,
    AgentSkill,
    Task,
    TaskState,
    TaskStatus,
)
from google.protobuf.struct_pb2 import Struct
from starlette.applications import Starlette

from purse.families.grids import parse, solve

__all__ = ["agent_card", "build_app", "main", "solve_text"]

COST_EXTENSION = "https://github.com/HASHIRU-AI/HASHIRU/blob/main/docs/a2a-cost-extension.md"
COST_KEY = "hashiru_cost_usd"
HELP = "send a logic-grid puzzle (houses in a row, attributes and numbered clues)"


def solve_text(text: str, wrong_rate: float = 0.0, seed: int = 0) -> str:
    try:
        solutions, _ = solve(parse(text))
    except ValueError as exc:
        return f"error: {exc}; {HELP}"
    if len(solutions) != 1:
        return f"error: the puzzle has {'no' if not solutions else 'more than one'} solution"
    solution = solutions[0]
    if wrong_rate and unlucky(json.dumps(solution, sort_keys=True), wrong_rate, seed):
        solution = swapped(solution)
    return f"<answer>{json.dumps(solution, sort_keys=True)}</answer>"


def unlucky(puzzle: str, rate: float, seed: int) -> bool:
    """Whether this puzzle gets a wrong answer: a hash of the seed and the puzzle, below `rate`."""
    digest = hashlib.sha256(f"{seed}:{puzzle}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < rate


def swapped(solution: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """The first attribute of houses 1 and 2 exchanged: a well-formed answer that breaks the puzzle's only solution."""
    first, second = sorted(solution, key=int)[:2]
    attr = sorted(solution[first])[0]
    out = {house: dict(values) for house, values in solution.items()}
    out[first][attr], out[second][attr] = solution[second][attr], solution[first][attr]
    return out


def agent_card(url: str, price_usd: float, name: str = "csp") -> AgentCard:
    price = Struct()
    price.update({"currency": "USD", "per_message_usd": price_usd})
    what = "Exact logic-grid solving" if name == "csp" else f"Logic-grid solving by {name}"
    return AgentCard(
        name=name, description=f"{what}: {HELP}.", version="1",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, extensions=[AgentExtension(
            uri=COST_EXTENSION, description="What a message to this agent costs (USD).", required=False,
            params=price)]),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[AgentSkill(id="grids", name="Logic grids", description=HELP, tags=["puzzles"])])


class Solver(AgentExecutor):
    def __init__(self, price_usd: float, wrong_rate: float = 0.0, seed: int = 0) -> None:
        self.price_usd, self.wrong_rate, self.seed = price_usd, wrong_rate, seed

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:  # open a task so the reply can complete it
            history = [context.message] if context.message is not None else []
            await event_queue.enqueue_event(Task(id=context.task_id or "", context_id=context.context_id or "",
                                                 status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                                                 history=history))
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        answer = solve_text(get_message_text(context.message) if context.message else "", self.wrong_rate, self.seed)
        await updater.complete(updater.new_agent_message([new_text_part(answer)],
                                                         metadata={COST_KEY: self.price_usd}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("a solve is one quick step")


def build_app(url: str, price_usd: float, name: str = "csp", wrong_rate: float = 0.0, seed: int = 0) -> Starlette:
    card = agent_card(url, price_usd, name)
    handler = DefaultRequestHandler(agent_executor=Solver(price_usd, wrong_rate, seed), task_store=InMemoryTaskStore(),
                                    agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=9302)
    parser.add_argument("--price", type=float, default=0.001, help="USD per message")
    parser.add_argument("--name", default="csp", help="the name on its Agent Card")
    parser.add_argument("--wrong", type=float, default=0.0, help="share of puzzles answered wrongly (0 to 1)")
    parser.add_argument("--seed", type=int, default=0, help="which puzzles those are")
    args = parser.parse_args()
    if not 0 <= args.wrong <= 1:
        parser.error("--wrong must be between 0 and 1")
    app = build_app(f"http://127.0.0.1:{args.port}/", args.price, args.name, args.wrong, args.seed)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
