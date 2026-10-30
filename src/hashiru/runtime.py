"""Agent runtime: the LLM loop, MCP tool calls and A2A ask_agent.

Every action and message passes through call_tool or ask_agent, so this is the one place that writes
the trace and moves labels.
"""
import asyncio
import json
import time
from collections.abc import Collection, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import httpx
import httpx2
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils.errors import A2AError
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import MCPError

from hashiru.ledger import AgentRecord
from hashiru.peers import DEADLINE_KEY, send, serve_request
from hashiru.privacy import may_hold, refuse, tool_refusal, where
from hashiru.session import USER, Session
from hashiru.trace import PRIVATE, SOURCES

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["CEO_PROMPT", "DEADLINE", "Executor", "ask_agent", "run_agent", "system_prompt"]

DEADLINE: ContextVar[float | None] = ContextVar("hashiru_deadline", default=None)  # Unix time, per agent run

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

# Sent once when a model ends its turn with nothing (seen with local thinking models whose whole answer went into
# their hidden reasoning): without it the caller gets an empty reply after all the agents' work.
EMPTY_REPLY = ("Your last reply was empty. Reply now with your answer to the message above, using what your tools "
               "and agents told you.")
NUDGE = [{"role": "assistant", "content": ""}, {"role": "user", "content": EMPTY_REPLY}]

ASK_TOOL = {"type": "function", "function": {
    "name": "ask_agent",
    "description": "Send a message to an agent you hired and wait for its reply.",
    "parameters": {"type": "object",
                   "properties": {"name": {"type": "string"}, "text": {"type": "string"}},
                   "required": ["name", "text"]}}}


def empty(rec: AgentRecord) -> str:
    """What an agent's caller gets when it answered with nothing twice: a failure, not an empty reply."""
    hint = " (a local model may have filled its context window: raise ollama_num_ctx)" if rec.tier == "local" else ""
    return f"failed: {rec.name} gave an empty answer{hint}"


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


def bearer(session: Session, me: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {session.tokens[me]}"}


def deadline_for(hub: "Hub") -> float:
    """The Unix time the current request must finish by: inherited inside an agent's run, fresh for the user."""
    return DEADLINE.get() or time.time() + hub.settings.task_timeout


async def a2a_send(hub: "Hub", session: Session, me: str, name: str, text: str) -> str:
    """One A2A request/response to the agent's endpoint, authenticated as `me`, carrying the request's deadline.

    The callee's run is bounded by the same deadline (Executor), so nobody keeps working after the reply that
    asked for it can no longer be used.
    """
    deadline = deadline_for(hub)
    remaining = deadline - time.time()
    if remaining <= 1:
        raise TimeoutError("the request's deadline has passed")
    reply, _, _ = await send(hub.agent_url(session, name), bearer(session, me), text, session.id,
                             remaining + 5, {DEADLINE_KEY: deadline})  # +5 s: the callee replies "out of time" first
    return reply


async def ask_agent(hub: "Hub", session: Session, me: str, name: str, text: str) -> str:
    """Message an agent `me` hired over A2A, tracing both directions and moving labels (rules 2-4). Private data
    goes neither to nor back to a cloud or remote agent (R3)."""
    target = session.ledger.get(name)
    if target is None or not session.may_ask(me, name):
        return f"error: {name!r} is not a live agent you hired"
    if not text.strip():
        return "error: the message is empty"
    trace, task = session.trace, session.id
    me_rec = session.ledger.get(me)
    my_tier = me_rec.tier if me_rec else USER
    sent = trace.outgoing(me, task, text)
    if PRIVATE in sent and not may_hold(target.tier):
        return "error: " + refuse(session, me, "message",
                                  f"{name} {where(target.tier)}, and this message carries private data")
    trace.emit("msg", task=task, agent=me, tier=my_tier, to=name, to_tier=target.tier,
               text=text, labels=sent, reply=False)
    trace.receive(name, task, sent)
    try:
        reply = await a2a_send(hub, session, me, name, text)
    except (A2AError, httpx.HTTPError, TimeoutError) as exc:
        return f"error: {name} failed: {exc}"
    back = trace.outgoing(name, task, reply)
    if PRIVATE in back and not may_hold(my_tier):
        return "error: " + refuse(session, me, "reply",
                                  f"{name}'s reply carries private data and {me} {where(my_tier)}")
    trace.emit("msg", task=task, agent=name, tier=target.tier, to=me, to_tier=my_tier,
               text=reply, labels=back, reply=True)
    trace.receive(me, task, back)
    return reply


async def call_tool(session: Session, mcp: Any, me: str, name: str, args: dict[str, Any],
                    turn: int | None = None, sources: Mapping[str, list[str]] = SOURCES,
                    created: Collection[str] = (), egress: Collection[str] = ()) -> dict[str, Any]:
    """Run one MCP tool for agent `me`, tracing it, applying label rules 1-3 and R3 (private stays local).

    `turn` is the seq of the LLM turn that issued the call: calls from one turn run concurrently,
    so completion order alone cannot say what the LLM had seen when it decided. Results of `created` (agent-made)
    tools carry their labels even when they are errors, because the tool wrote the error text. `egress` names the
    tools an agent holding private data may not call.
    """
    trace, task, rec = session.trace, session.id, session.ledger.agents[me]
    if refused := tool_refusal(session, rec, name, sources, egress):
        return {"error": refused}
    arg_labels = trace.outgoing(me, task, json.dumps(args))
    try:
        result = unpack(await mcp.call_tool(name, args))
    except (MCPError, httpx2.HTTPError) as exc:  # e.g. a revoked token (401) or an oversized request (413)
        result = {"error": f"{name} failed: {exc}"}
    extra = result.pop("_labels", [])  # labels a tool attaches to this particular result (e.g. memory)
    got = sources.get(name, []) if "error" not in result or name in created else []
    got = sorted(set(got) | {str(label) for label in extra}) if isinstance(extra, list) else got
    if PRIVATE in got and not may_hold(rec.tier):  # a label this result carries, e.g. a memory from a private chat
        result, got = {"error": refuse(session, me, "withheld", f"{name}'s result carries private data and {me} "
                                                                f"{where(rec.tier)}")}, []
    trace.emit("tool", task=task, agent=me, tier=rec.tier, name=name, turn=turn,
               args=args, arg_labels=arg_labels, result=result, labels=got)
    trace.receive(me, task, got)
    if name == "hire_agent" and "hired" in result:
        trace.receive(result["hired"], task, arg_labels)  # rule 2: the hire prompt carries the hirer's labels
    if name == "deliver_to_user" and "error" not in result:
        trace.receive(USER, task, arg_labels)  # the chat's history now holds it: a private delivery makes it private
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
    if not allowed(session.ledger.agents[me], call.function.name):
        return {"error": f"{call.function.name} is not one of your tools"}
    if call.function.name == "ask_agent":
        return {"reply": await ask_agent(hub, session, me, str(args.get("name", "")), str(args.get("text", "")))}
    egress = set() if hub.settings.private_egress == "allow" else hub.egress()  # an unknown value blocks
    return await call_tool(session, mcp, me, call.function.name, args, turn, hub.sources(), hub.external(), egress)


async def think(hub: "Hub", session: Session, rec: AgentRecord, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None) -> tuple[Any, list[Any], int]:
    """One LLM call: charge its cost to the agent and trace the thought. Returns (msg, calls, seq)."""
    call = asyncio.ensure_future(hub.llm.chat(rec.model, messages, tools))
    try:
        msg, cost = await asyncio.shield(call)
    except asyncio.CancelledError:
        # The provider bills a request it has received: when the call returns, charge and trace it anyway.
        call.add_done_callback(lambda done: charge(session, rec, done.result(), late=True)
                               if not done.cancelled() and done.exception() is None else None)
        raise
    return msg, msg.tool_calls or [], charge(session, rec, (msg, cost))


def charge(session: Session, rec: AgentRecord, answer: tuple[Any, float], late: bool = False) -> int:
    """Charge one LLM call to the agent and trace it. Returns the trace seq of the thought."""
    msg, cost = answer
    rec.spent += cost
    calls = msg.tool_calls or []
    event = session.trace.emit("llm", task=session.id, agent=rec.name, tier=rec.tier, model=rec.model,
                               text=msg.content or "", tool_calls=[c.function.name for c in calls], cost_usd=cost,
                               **({"after_stop": True} if late else {}))
    seq: int = event["seq"]
    return seq


def allowed(rec: AgentRecord, tool: str) -> bool:
    """A manager hired with a tools list may use only those; everyone else may use every tool."""
    return rec.tools is None or tool in rec.tools


def still_on(hub: "Hub", session: Session, rec: AgentRecord) -> bool:
    """False once the agent is fired or its session closed: its run stops before spending more."""
    return rec.live and hub.sessions.get(session.id) is session


def spent_out(hub: "Hub", session: Session, rec: AgentRecord) -> str | None:
    """Why the agent may not make another LLM call, under enforce_budget: a cloud agent needs money left."""
    if hub.settings.enforce_budget and rec.tier != "local" and session.ledger.left()[rec.name] <= 0:
        return f"failed: {rec.name}'s pool is spent"
    return None


async def run_agent(hub: "Hub", session: Session, name: str, text: str) -> str:
    """Workers answer with one LLM call. The CEO and managers loop over tools until they reply."""
    rec = session.ledger.agents[name]
    if not still_on(hub, session, rec):
        return "failed"
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt(rec)},
                                      {"role": "user", "content": text}]
    if rec.role == "worker":
        for attempt in range(2):
            if refused := spent_out(hub, session, rec):
                return refused
            msg, _, _ = await think(hub, session, rec, messages, None)
            if (msg.content or "").strip() or not still_on(hub, session, rec):
                return msg.content or ""
            if attempt:
                return empty(rec)
            messages += NUDGE
        return empty(rec)
    timeout = httpx2.Timeout(30, read=hub.settings.task_timeout)
    async with (httpx2.AsyncClient(headers=bearer(session, name), timeout=timeout) as http,
                streamable_http_client(f"{hub.settings.internal_url}/mcp", http_client=http) as (read, write, *_),
                ClientSession(read, write) as mcp):
        await mcp.initialize()
        nudged = False
        for _ in range(hub.settings.max_turns):
            if not still_on(hub, session, rec):
                return "failed"
            if refused := spent_out(hub, session, rec):
                return refused
            # Re-read every turn: a tool created (or deleted) on the last turn is visible on this one.
            tools = [to_openai(t) for t in (await mcp.list_tools()).tools if allowed(rec, t.name)]
            tools += [ASK_TOOL] if allowed(rec, "ask_agent") else []
            msg, calls, turn = await think(hub, session, rec, messages, tools)
            if not calls:
                if (msg.content or "").strip():
                    return msg.content or ""
                if nudged:
                    return empty(rec)
                nudged = True
                messages += NUDGE
                continue
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
    """A2A entry point. Every hired agent is served by one of these (through peers.serve_request)."""

    def __init__(self, hub: "Hub", session: Session, name: str) -> None:
        self.hub = hub
        self.session = session
        self.name = name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        async def work(text: str, deadline: float) -> str:
            token = DEADLINE.set(deadline)  # asks this run makes carry the same deadline
            try:
                return await run_agent(self.hub, self.session, self.name, text)
            finally:
                DEADLINE.reset(token)
        await serve_request(self.hub, self.session, self.name, context, event_queue, work)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")
