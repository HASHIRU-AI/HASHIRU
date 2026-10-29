"""A stand-in remote A2A agent for the peer tests: it answers "echo: <text>", can declare a price through the
cost extension, can report what a request cost, and can require a bearer token."""
import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role, Task, TaskStatus
from google.protobuf.json_format import MessageToDict
from starlette.applications import Starlette
from starlette.responses import JSONResponse

from hashiru.peers import COST_KEY, Price, cost_extension


class Echo(AgentExecutor):
    def __init__(self, report: Any, delay: float, reply_size: int, state: Any) -> None:
        self.report, self.delay, self.reply_size, self.state = report, delay, reply_size, state
        self.seen: list[tuple[str, dict[str, Any]]] = []
        self.contexts: list[str] = []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        assert context.message is not None
        text = get_message_text(context.message)
        self.seen.append((text, MessageToDict(context.message.metadata)))
        self.contexts.append(context.context_id or "")
        await asyncio.sleep(self.delay)
        answer = "x" * self.reply_size if self.reply_size else f"echo: {text}"
        reply = new_text_message(answer, context_id=context.context_id, role=Role.ROLE_AGENT)
        if self.report is not None:
            reply.metadata.update({COST_KEY: self.report})
        if self.state is None:
            await event_queue.enqueue_event(reply)  # message mode, as many agents answer
            return
        await event_queue.enqueue_event(Task(id=context.task_id or "", context_id=context.context_id or "",
                                             status=TaskStatus(state=self.state, message=reply)))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def guarded(app: Any, token: str | None) -> Any:
    async def gate(scope: Any, receive: Any, send: Any) -> None:
        if token and scope["type"] == "http":
            sent = dict(scope.get("headers", [])).get(b"authorization", b"").decode()
            if sent != f"Bearer {token}":
                await JSONResponse({"error": "unauthorized"}, status_code=401)(scope, receive, send)
                return
        await app(scope, receive, send)
    return gate


@contextlib.asynccontextmanager
async def peer(port: int, price: Price | None = None, report: Any = None, token: str | None = None,
               delay: float = 0.0, declared_url: str | None = None, reply_size: int = 0,
               state: Any = None) -> AsyncIterator[tuple[str, Echo]]:
    """`declared_url`: the endpoint the card names (a hostile card points elsewhere). `state`: answer with a Task in
    that state instead of a Message."""
    url = f"http://127.0.0.1:{port}/"
    card = AgentCard(
        name="peer", description="echoes what it is sent", version="1",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, extensions=[cost_extension(price)] if price else []),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=declared_url or url,
                                             protocol_version="1.0")],
        skills=[AgentSkill(id="echo", name="echo", description="echo", tags=[])])
    executor = Echo(report, delay, reply_size, state)
    handler = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card)
    app = Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))
    srv = uvicorn.Server(uvicorn.Config(guarded(app, token), host="127.0.0.1", port=port, log_level="warning",
                                        timeout_graceful_shutdown=1))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    try:
        yield url, executor
    finally:
        srv.should_exit = True
        await task
