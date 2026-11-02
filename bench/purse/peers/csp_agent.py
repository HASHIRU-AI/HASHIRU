"""An outside agent for PURSE's grid jobs: it reads the last logic-grid puzzle in a message, solves it exactly and
answers with the JSON solution. Like the example calc agent, it needs no LLM, declares its price through HASHIRU's
cost extension and reports what each request cost.

    uv run python -m purse.peers.csp_agent --port 9302 --price 0.001
"""
import argparse
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


def solve_text(text: str) -> str:
    try:
        solutions, _ = solve(parse(text))
    except ValueError as exc:
        return f"error: {exc}; {HELP}"
    if len(solutions) != 1:
        return f"error: the puzzle has {'no' if not solutions else 'more than one'} solution"
    return f"<answer>{json.dumps(solutions[0], sort_keys=True)}</answer>"


def agent_card(url: str, price_usd: float) -> AgentCard:
    price = Struct()
    price.update({"currency": "USD", "per_message_usd": price_usd})
    return AgentCard(
        name="csp", description=f"Exact logic-grid solving: {HELP}.", version="1",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, extensions=[AgentExtension(
            uri=COST_EXTENSION, description="What a message to this agent costs (USD).", required=False,
            params=price)]),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[AgentSkill(id="grids", name="Logic grids", description=HELP, tags=["puzzles"])])


class Solver(AgentExecutor):
    def __init__(self, price_usd: float) -> None:
        self.price_usd = price_usd

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:  # open a task so the reply can complete it
            history = [context.message] if context.message is not None else []
            await event_queue.enqueue_event(Task(id=context.task_id or "", context_id=context.context_id or "",
                                                 status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                                                 history=history))
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        answer = solve_text(get_message_text(context.message) if context.message else "")
        await updater.complete(updater.new_agent_message([new_text_part(answer)],
                                                         metadata={COST_KEY: self.price_usd}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("a solve is one quick step")


def build_app(url: str, price_usd: float) -> Starlette:
    card = agent_card(url, price_usd)
    handler = DefaultRequestHandler(agent_executor=Solver(price_usd), task_store=InMemoryTaskStore(), agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=9302)
    parser.add_argument("--price", type=float, default=0.001, help="USD per message")
    args = parser.parse_args()
    uvicorn.run(build_app(f"http://127.0.0.1:{args.port}/", args.price), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
