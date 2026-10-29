"""The CEO as a public A2A agent: its card at /.well-known/agent-card.json, JSON-RPC at /a2a.

Each A2A context is one conversation, so hires last between messages in a context. Progress arrives as `working`
status updates (one per activity line, the event in the message metadata); the reply completes the task and reports
its cost through the HASHIRU cost extension (`hashiru_cost_usd`).
"""
import contextlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from a2a.helpers import get_message_text, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
    TaskState,
)
from starlette.routing import Route

from hashiru.interop.activity import describe, public_event
from hashiru.interop.conversations import Busy
from hashiru.peers import COST_KEY, BoundedTaskStore, Price, begin, cost_extension, deadline_of, finish

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["A2A_PATH", "PublicExecutor", "a2a_routes", "public_card"]

A2A_PATH = "/a2a"
try:
    VERSION = version("hashiru")
except PackageNotFoundError:  # running from a source tree that was never installed
    VERSION = "2"
DESCRIPTION = ("HASHIRU's CEO. It splits a request into subtasks, hires manager and worker agents on local and cloud "
               "models within a dollar budget, delegates to them over A2A, and answers. Messages in one context "
               "continue one conversation; its agents stay hired between messages.")


def public_card(hub: "Hub") -> AgentCard:
    card = AgentCard(
        name="HASHIRU", description=DESCRIPTION, version=VERSION,
        documentation_url="https://github.com/HASHIRU-AI/HASHIRU",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        # A message can cost up to a conversation's whole budget (declaring 0 would read as "free" to a peer).
        capabilities=AgentCapabilities(streaming=True, extensions=[cost_extension(Price(hub.settings.budget_usd))]),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=f"{hub.settings.base_url}{A2A_PATH}",
                                             protocol_version="1.0")],
        skills=[AgentSkill(id="orchestrate", name="Orchestrate agents on a budget", description=DESCRIPTION,
                           tags=["multi-agent", "budget", "hiring"])])
    if hub.settings.api_key:
        card.security_schemes["bearer"].CopyFrom(
            SecurityScheme(http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="Bearer")))
        requirement = SecurityRequirement()
        requirement.schemes["bearer"].CopyFrom(StringList())
        card.security_requirements.append(requirement)
    return card


class PublicExecutor(AgentExecutor):
    """Answers each A2A message with one conversation turn, streaming the agents' work as status updates."""

    def __init__(self, hub: "Hub") -> None:
        self.hub = hub

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = get_message_text(context.message).strip() if context.message else ""
        updater = await begin(context, event_queue)
        if not text:
            await finish(context, event_queue, "failed: the message is empty", {COST_KEY: 0.0}, updater)
            return
        convs, key = self.hub.conversations, context.context_id or ""
        try:
            conv = convs.for_context(key) or await convs.open("a2a", context=key)
            turn = convs.ask(conv, text, deadline=deadline_of(self.hub, context))
        except Busy as exc:
            await updater.failed(updater.new_agent_message([new_text_part(f"failed: {exc}")]))
            return
        async with contextlib.aclosing(turn) as events:
            async for item in events:
                if item["kind"] == "answer":
                    await finish(context, event_queue, item["text"], {COST_KEY: item["cost_usd"]}, updater)
                elif line := describe(item):
                    await updater.update_status(TaskState.TASK_STATE_WORKING, message=updater.new_agent_message(
                        [new_text_part(line)], metadata={"hashiru_event": public_event(item)}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")


def a2a_routes(hub: "Hub") -> list[Route]:
    card = public_card(hub)
    handler = DefaultRequestHandler(agent_executor=PublicExecutor(hub), task_store=BoundedTaskStore(64),
                                    agent_card=card)
    return create_agent_card_routes(card) + create_jsonrpc_routes(handler, A2A_PATH)
