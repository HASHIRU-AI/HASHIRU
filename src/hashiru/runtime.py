"""Agent runtime: the LLM loop, MCP tool calls and A2A ask_agent.

Every action and message passes through call_tool or ask_agent, so this is the one place that writes
the trace and moves labels.
"""
import asyncio
import json
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

import httpx
import httpx2
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_message_text, new_text_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Role, SendMessageRequest
from a2a.utils.errors import A2AError
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from hashiru.ledger import AgentRecord
from hashiru.session import USER, Session
from hashiru.trace import SOURCES

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["CEO_PROMPT", "Executor", "ask_agent", "run_agent", "system_prompt"]

RULES = "\n".join([
    "Rules you must follow:",
    "1. Before every hire_agent call, call check_usage on your own pool and confirm the money and free memory "
    "cover the hire. If anyone may have hired since your last check, check again.",
    "2. Your own spending plus what you reserved for agents you hired must never exceed your pool.",
    "3. Only check your own pool.",
    "4. Prefer local models: they cost no money, only memory. Use cloud models when the task needs them.",
    "5. Fire agents once you no longer need them. Firing a local agent frees its memory.",
])

CEO_PROMPT = ("You are the CEO of HASHIRU. Break the user's request into subtasks, hire agents with hire_agent "
              "(managers use tools and can hire their own workers; workers answer once and have no tools), "
              "delegate with ask_agent, fire agents you no longer need, and answer the user.")

ASK_TOOL = {"type": "function", "function": {
    "name": "ask_agent",
    "description": "Send a message to an agent you hired and wait for its reply.",
    "parameters": {"type": "object",
                   "properties": {"name": {"type": "string"}, "text": {"type": "string"}},
                   "required": ["name", "text"]}}}


def system_prompt(rec: AgentRecord) -> str:
    """Workers get only their hirer's prompt; the CEO and managers also get their pool and the rules."""
    if rec.role == "worker":
        return rec.prompt
    return f"You are {rec.name}; your pool is '{rec.name}'.\n{RULES}\n\nYour job: {rec.prompt}"


def to_openai(tool: Any) -> dict[str, Any]:
    """MCP tool -> OpenAI-style tool for LiteLLM."""
    schema = tool.input_schema
    parameters = {"type": "object", "properties": schema.get("properties", {}), "required": schema.get("required", [])}
    if "$defs" in schema:  # nested types in a created tool's schema refer to these
        parameters["$defs"] = schema["$defs"]
    return {"type": "function", "function": {
        "name": tool.name, "description": tool.description or "", "parameters": parameters}}


def unpack(result: Any) -> dict[str, Any]:
    """MCP result -> dict. Dict-returning tools arrive as JSON text in content[0]."""
    text = result.content[0].text if result.content else ""
    if result.is_error:
        return {"error": text}
    out: dict[str, Any] = result.structured_content or (json.loads(text) if text else {})
    return out


def hop_timeout(hub: "Hub", session: Session, name: str) -> float:
    """Deeper hops get shorter timeouts. This does not bound a whole subtree: an agent may ask several times in
    sequence, and a callee that times out keeps running until it finishes, is fired, or its session closes."""
    rec = session.ledger.get(name)
    depth = rec.depth if rec else 0
    return hub.settings.a2a_timeout * (hub.settings.max_depth + 1 - depth)


def bearer(session: Session, me: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {session.tokens[me]}"}


async def a2a_send(hub: "Hub", session: Session, me: str, name: str, text: str) -> str:
    """One A2A request/response to the agent's endpoint, authenticated as `me`."""
    async with httpx.AsyncClient(timeout=hop_timeout(hub, session, name), headers=bearer(session, me)) as http:
        client = await create_client(hub.agent_url(session, name),
                                     client_config=ClientConfig(streaming=False, httpx_client=http))
        reply = ""
        request = SendMessageRequest(message=new_text_message(text, context_id=session.id, role=Role.ROLE_USER))
        async for event in client.send_message(request):
            if event.HasField("message"):
                reply = get_message_text(event.message)
        await client.close()
    return reply


async def ask_agent(hub: "Hub", session: Session, me: str, name: str, text: str) -> str:
    """Message an agent `me` hired over A2A, tracing both directions and moving labels (rules 2-4)."""
    target = session.ledger.get(name)
    if target is None or not session.may_ask(me, name):
        return f"error: {name!r} is not a live agent you hired"
    trace, task = session.trace, session.id
    me_rec = session.ledger.get(me)
    my_tier = me_rec.tier if me_rec else USER
    sent = trace.outgoing(me, task, text)
    trace.emit("msg", task=task, agent=me, tier=my_tier, to=name, to_tier=target.tier,
               text=text, labels=sent, reply=False)
    trace.receive(name, task, sent)
    try:
        reply = await a2a_send(hub, session, me, name, text)
    except (A2AError, httpx.HTTPError) as exc:
        return f"error: {name} failed: {exc}"
    back = trace.outgoing(name, task, reply)
    trace.emit("msg", task=task, agent=name, tier=target.tier, to=me, to_tier=my_tier,
               text=reply, labels=back, reply=True)
    trace.receive(me, task, back)
    return reply


async def call_tool(session: Session, mcp: Any, me: str, name: str, args: dict[str, Any],
                    turn: int | None = None, sources: Mapping[str, list[str]] = SOURCES,
                    created: Collection[str] = ()) -> dict[str, Any]:
    """Run one MCP tool for agent `me`, tracing it and applying label rules 1-3.

    `turn` is the seq of the LLM turn that issued the call: calls from one turn run concurrently,
    so completion order alone cannot say what the LLM had seen when it decided. Results of `created` (agent-made)
    tools carry their labels even when they are errors, because the tool wrote the error text.
    """
    trace, task = session.trace, session.id
    arg_labels = trace.outgoing(me, task, json.dumps(args))
    try:
        result = unpack(await mcp.call_tool(name, args))
    except (MCPError, httpx2.HTTPError) as exc:  # e.g. a revoked token (401) or an oversized request (413)
        result = {"error": f"{name} failed: {exc}"}
    got = sources.get(name, []) if "error" not in result or name in created else []
    trace.emit("tool", task=task, agent=me, tier=session.ledger.agents[me].tier, name=name, turn=turn,
               args=args, arg_labels=arg_labels, result=result, labels=got)
    trace.receive(me, task, got)
    if name == "hire_agent" and "hired" in result:
        trace.receive(result["hired"], task, arg_labels)  # rule 2: the hire prompt carries the hirer's labels
    return result


async def dispatch(hub: "Hub", session: Session, mcp: Any, me: str, call: Any,
                   turn: int | None = None) -> dict[str, Any]:
    """Route one LLM tool call to ask_agent (A2A) or an MCP tool."""
    try:
        args = json.loads(call.function.arguments or "{}")
    except json.JSONDecodeError:
        return {"error": "tool arguments were not valid JSON"}
    if not isinstance(args, dict):
        return {"error": "tool arguments must be a JSON object"}
    if call.function.name == "ask_agent":
        return {"reply": await ask_agent(hub, session, me, str(args.get("name", "")), str(args.get("text", "")))}
    return await call_tool(session, mcp, me, call.function.name, args, turn, hub.sources(), hub.toolsmith.names())


async def think(hub: "Hub", session: Session, rec: AgentRecord, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None) -> tuple[Any, list[Any], int]:
    """One LLM call: charge its cost to the agent and trace the thought. Returns (msg, calls, seq)."""
    msg, cost = await hub.llm.chat(rec.model, messages, tools)
    rec.spent += cost
    calls = msg.tool_calls or []
    event = session.trace.emit("llm", task=session.id, agent=rec.name, tier=rec.tier, model=rec.model,
                               text=msg.content or "", tool_calls=[c.function.name for c in calls], cost_usd=cost)
    return msg, calls, event["seq"]


def still_on(hub: "Hub", session: Session, rec: AgentRecord) -> bool:
    """False once the agent is fired or its session closed: its run stops before spending more."""
    return rec.live and hub.sessions.get(session.id) is session


async def run_agent(hub: "Hub", session: Session, name: str, text: str) -> str:
    """Workers answer with one LLM call. The CEO and managers loop over tools until they reply."""
    rec = session.ledger.agents[name]
    if not still_on(hub, session, rec):
        return "failed"
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt(rec)},
                                      {"role": "user", "content": text}]
    if rec.role == "worker":
        msg, _, _ = await think(hub, session, rec, messages, None)
        return msg.content or ""
    timeout = httpx2.Timeout(30, read=hub.settings.a2a_timeout)
    async with (httpx2.AsyncClient(headers=bearer(session, name), timeout=timeout) as http,
                streamable_http_client(f"{hub.settings.internal_url}/mcp", http_client=http) as (read, write, *_),
                ClientSession(read, write) as mcp):
        await mcp.initialize()
        for _ in range(hub.settings.max_turns):
            if not still_on(hub, session, rec):
                return "failed"
            # Re-read every turn: a tool created (or deleted) on the last turn is visible on this one.
            tools = [to_openai(t) for t in (await mcp.list_tools()).tools] + [ASK_TOOL]
            msg, calls, turn = await think(hub, session, rec, messages, tools)
            if not calls:
                return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in calls]})
            async with asyncio.TaskGroup() as group:  # if one call crashes, its siblings are cancelled too
                tasks = [group.create_task(dispatch(hub, session, mcp, name, c, turn)) for c in calls]
            messages += [{"role": "tool", "tool_call_id": c.id, "content": json.dumps(t.result())}
                         for c, t in zip(calls, tasks, strict=True)]
    return "failed"


class Executor(AgentExecutor):
    """A2A entry point. Every hired agent is served by one of these."""

    def __init__(self, hub: "Hub", session: Session, name: str) -> None:
        self.hub = hub
        self.session = session
        self.name = name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        text = get_message_text(context.message) if context.message else ""
        try:
            reply = await run_agent(self.hub, self.session, self.name, text)
        except Exception as exc:  # record any failure in the trace, then let A2A report it
            cause: BaseException = exc
            while isinstance(cause, BaseExceptionGroup) and len(cause.exceptions) == 1:
                cause = cause.exceptions[0]  # MCP's task group wraps the real error (e.g. WalletExceeded)
            self.session.trace.emit("error", task=self.session.id, agent=self.name,
                                    error=f"{type(cause).__name__}: {cause}")
            raise cause from None
        await event_queue.enqueue_event(
            new_text_message(reply, context_id=context.context_id, role=Role.ROLE_AGENT))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")
