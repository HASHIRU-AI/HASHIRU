"""A2A plumbing shared by our agents and remote peers: sends that carry metadata both ways, the one path every
request is served through, and HASHIRU's cost extension for Agent Cards (see docs/a2a-cost-extension.md)."""
import asyncio
import collections
import json
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import httpx
import litellm
from a2a.client import ClientConfig, ClientFactory, create_client
from a2a.helpers import get_artifact_text, get_message_text, new_text_message, new_text_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, AgentExtension, AgentInterface, Role, SendMessageRequest, Task, TaskState, TaskStatus
from a2a.utils.errors import A2AError
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Struct

from hashiru.llm import WalletExceeded, is_local
from hashiru.session import Session

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["COST_EXTENSION", "COST_KEY", "DEADLINE_KEY", "PEER_PREFIX", "BoundedTaskStore", "PeerExecutor", "Price",
           "cost_extension", "fetch_card", "finish", "model_price", "pin", "price_of", "reply_of", "send",
           "serve_request", "usd"]

COST_EXTENSION = "https://github.com/HASHIRU-AI/HASHIRU/blob/main/docs/a2a-cost-extension.md"
COST_KEY = "hashiru_cost_usd"        # reply metadata: what the request cost, in USD
DEADLINE_KEY = "hashiru_deadline"    # request metadata: Unix time by which the reply is needed
PEER_PREFIX = "a2a/"                 # hire_agent model names for configured remote agents: a2a/<peer>
MAX_CARD_BYTES = 256_000
MAX_REPLY_CHARS = 20_000
CHARS_PER_TOKEN = 4                  # for estimating a token-priced peer's cost from text length
DONE = TaskState.TASK_STATE_COMPLETED
ENDED = {TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED,
         TaskState.TASK_STATE_REJECTED}

Work = Callable[[str, float], Awaitable[str]]  # (request text, deadline) -> reply text


@dataclass(frozen=True)
class Price:
    per_message_usd: float = 0.0
    per_input_token_usd: float = 0.0
    per_output_token_usd: float = 0.0

    def estimate(self, sent: str, received: str = "") -> float:
        """Per-message price plus token prices over the text lengths (about 4 characters per token)."""
        return (self.per_message_usd + self.per_input_token_usd * math.ceil(len(sent) / CHARS_PER_TOKEN)
                + self.per_output_token_usd * math.ceil(len(received) / CHARS_PER_TOKEN))


def usd(value: Any) -> float | None:
    """A finite, non-negative number, or None for anything else (inf, NaN, text, negative)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def cost_extension(price: Price) -> AgentExtension:
    params = Struct()
    params.update({"currency": "USD", **asdict(price)})
    return AgentExtension(uri=COST_EXTENSION, description="What a message to this agent costs (USD).",
                          required=False, params=params)


def model_price(model: str) -> Price:
    """What one of our own agents charges, for its Agent Card: LiteLLM's token prices, or nothing if local."""
    if is_local(model) or model.startswith(PEER_PREFIX):
        return Price()
    try:
        info = litellm.get_model_info(model)
    except Exception:  # LiteLLM raises a plain Exception for unmapped models
        return Price()
    return Price(0.0, float(info.get("input_cost_per_token") or 0), float(info.get("output_cost_per_token") or 0))


def price_of(card: AgentCard) -> Price | None:
    """The price an Agent Card declares through the cost extension, or None if it declares none (or nonsense)."""
    for extension in card.capabilities.extensions:
        if extension.uri == COST_EXTENSION:
            params = MessageToDict(extension.params)
            if params.get("currency", "USD") != "USD":
                return None
            parts = [usd(params.get(k, 0) or 0) for k in
                     ("per_message_usd", "per_input_token_usd", "per_output_token_usd")]
            if any(p is None for p in parts):
                return None
            return Price(*(float(p or 0.0) for p in parts))
    return None


async def fetch_card(url: str, headers: dict[str, str], timeout: float) -> AgentCard:
    """An agent's card from <url>/.well-known/agent-card.json, at most MAX_CARD_BYTES, no redirects."""
    async with (httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=False) as http,
                http.stream("GET", url.rstrip("/") + "/.well-known/agent-card.json") as resp):
        resp.raise_for_status()
        body = bytearray()
        async for chunk in resp.aiter_bytes():
            body += chunk
            if len(body) > MAX_CARD_BYTES:
                raise ValueError(f"the agent card is larger than {MAX_CARD_BYTES} bytes")
    card: AgentCard = ParseDict(json.loads(body), AgentCard(), ignore_unknown_fields=True)
    return card


def pin(card: AgentCard, url: str) -> AgentCard:
    """The card, talking only to the configured `url`: whatever endpoint the card itself names is ignored, so a peer
    cannot redirect our credentials or requests elsewhere."""
    bindings = {i.protocol_binding for i in card.supported_interfaces}
    if bindings and "JSONRPC" not in bindings:
        raise ValueError("the agent does not offer A2A over JSON-RPC")
    version = next((i.protocol_version for i in card.supported_interfaces if i.protocol_binding == "JSONRPC"), "")
    pinned = AgentCard()
    pinned.CopyFrom(card)
    del pinned.supported_interfaces[:]
    pinned.supported_interfaces.append(AgentInterface(protocol_binding="JSONRPC", url=url,
                                                      protocol_version=version or "1.0"))
    return pinned


def reply_of(event: Any) -> tuple[str, dict[str, Any], int] | None:
    """(text, metadata, task state) of the reply in one client event, whether a Message or a Task answered."""
    if event.HasField("message"):
        return get_message_text(event.message), MessageToDict(event.message.metadata), DONE
    if event.HasField("task"):
        task = event.task
        if task.status.HasField("message"):
            message = task.status.message
            return get_message_text(message), MessageToDict(message.metadata), task.status.state
        return "\n".join(get_artifact_text(a) for a in task.artifacts), {}, task.status.state
    if event.HasField("status_update") and event.status_update.status.HasField("message"):
        status = event.status_update.status
        return get_message_text(status.message), MessageToDict(status.message.metadata), status.state
    return None


async def send(url: str, headers: dict[str, str], text: str, context_id: str, timeout: float,
               metadata: dict[str, Any] | None = None,
               card: AgentCard | None = None) -> tuple[str, dict[str, Any], int]:
    """One A2A request/response: (reply text, reply metadata, final task state). With a (pinned) `card`, the card is
    not fetched again and requests go only where it points."""
    async with httpx.AsyncClient(timeout=timeout, headers=headers, follow_redirects=False) as http:
        config = ClientConfig(streaming=False, httpx_client=http)
        client = ClientFactory(config).create(card) if card is not None else \
            await create_client(url, client_config=config)
        message = new_text_message(text, context_id=context_id, role=Role.ROLE_USER)
        if metadata:
            message.metadata.update(metadata)
        answer: tuple[str, dict[str, Any], int] = ("", {}, TaskState.TASK_STATE_UNSPECIFIED)
        async for event in client.send_message(SendMessageRequest(message=message)):
            answer = reply_of(event) or answer
        await client.close()
    return answer


async def finish(context: RequestContext, event_queue: EventQueue, text: str, metadata: dict[str, Any]) -> None:
    """Answer a request in A2A task mode: the reply is the task's final (completed) status message.

    A bare Message reply ("message mode") never reaches a final task state, so the SDK keeps its two worker tasks
    for that request waiting forever: one pair leaked per message.
    """
    if context.current_task is None:  # built directly: the SDK's helper refuses a message with empty text
        history = [context.message] if context.message is not None else []
        await event_queue.enqueue_event(Task(id=context.task_id or "", context_id=context.context_id or "",
                                             status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED), history=history))
    updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
    await updater.complete(updater.new_agent_message([new_text_part(text or "(empty)")], metadata=metadata))


class BoundedTaskStore(InMemoryTaskStore):
    """Keeps only the most recent finished tasks: every request is a new task, and nothing reads old ones back."""

    def __init__(self, keep: int = 16) -> None:
        super().__init__()
        self._finished: collections.deque[str] = collections.deque()
        self._keep = keep

    async def save(self, task: Task, context: ServerCallContext) -> None:
        await super().save(task, context)
        if task.status.state in ENDED and task.id not in self._finished:
            self._finished.append(task.id)
            while len(self._finished) > self._keep:
                await super().delete(self._finished.popleft(), context)


def live(hub: "Hub", session: Session, name: str) -> bool:
    rec = session.ledger.get(name)
    return bool(rec and rec.live and hub.sessions.get(session.id) is session)


def spent_below(session: Session, name: str) -> float:
    """What an agent and everyone it hired (live or fired) have spent."""
    agents = session.ledger.agents
    return agents[name].spent + sum(spent_below(session, a.name) for a in agents.values() if a.parent == name)


def deadline_of(hub: "Hub", context: RequestContext) -> float:
    """The request's deadline: what the caller asked for, but never later than task_timeout from now."""
    latest = time.time() + hub.settings.task_timeout
    asked = usd(MessageToDict(context.message.metadata).get(DEADLINE_KEY)) if context.message else None
    return latest if asked is None else min(latest, asked)


async def serve_request(hub: "Hub", session: Session, name: str, context: RequestContext, event_queue: EventQueue,
                        work: Work) -> None:
    """The one way a hosted agent (ours or a proxied peer) answers a request.

    One request at a time per agent (concurrent asks queue, so nobody spends the same money twice), bounded by the
    request's deadline, cancellable by fire or close (the caller then gets "failed: stopped"), and finished in task
    mode with what the agent and its hires spent on it.
    """
    text = get_message_text(context.message) if context.message else ""
    deadline = deadline_of(hub, context)
    task = asyncio.current_task()
    hub.track(session, name, task)
    before = spent_below(session, name)
    try:
        async with asyncio.timeout(max(0.0, deadline - time.time())), hub.agent_lock(session, name):
            reply = await work(text, deadline)
    except TimeoutError:
        session.trace.emit("error", task=session.id, agent=name, error="out of time")
        reply = "failed: out of time"
    except asyncio.CancelledError:
        if task is None or live(hub, session, name):
            raise  # not our stop (e.g. the server is shutting down): let the cancellation through
        task.uncancel()  # fired, or its session closed: answer the request instead of leaving it hanging
        session.trace.emit("error", task=session.id, agent=name, error="stopped")
        reply = "failed: stopped (fired, or the session closed)"
    except Exception as exc:  # record any failure in the trace, then let A2A report it
        cause: BaseException = exc
        while isinstance(cause, BaseExceptionGroup) and len(cause.exceptions) == 1:
            cause = cause.exceptions[0]  # MCP's task group wraps the real error (e.g. WalletExceeded)
        session.trace.emit("error", task=session.id, agent=name, error=f"{type(cause).__name__}: {cause}")
        raise cause from None
    finally:
        hub.untrack(session, name, task)
    await finish(context, event_queue, reply, {COST_KEY: round(spent_below(session, name) - before, 8)})


class PeerExecutor(AgentExecutor):
    """Serves a hired remote agent on the hub: relays each message to the peer and charges the hirer's pool.

    The hub keeps the peer behind its own gate and trace, so the hirer talks to it like any other hire. The estimated
    price is reserved before sending (so concurrent asks cannot overdraw the pool); afterwards the charge is the
    larger of the estimate and the peer's reported cost, but never more than the pool held.
    """

    def __init__(self, hub: "Hub", session: Session, name: str) -> None:
        self.hub = hub
        self.session = session
        self.name = name

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await serve_request(self.hub, self.session, self.name, context, event_queue, self._relay)

    async def _relay(self, text: str, deadline: float) -> str:
        hub, session, rec = self.hub, self.session, self.session.ledger.agents[self.name]
        peer, card = hub.peer(rec.model), hub.peer_cards.get((session.id, rec.name))
        if not rec.live or peer is None or card is None:
            return "failed: this agent is no longer available"
        if rec.prompt:
            text = f"{rec.prompt}\n\n{text}"
        declared = price_of(card) or Price()
        price = Price(max(rec.price_usd, declared.per_message_usd), declared.per_input_token_usd,
                      declared.per_output_token_usd)
        estimate, left = price.estimate(text), session.ledger.left()[rec.name]
        if hub.settings.enforce_budget and left + 1e-9 < estimate:
            return f"failed: {rec.name}'s pool (${left:.6f}) cannot pay {peer.name}'s price of ${estimate:.6f}"
        try:
            hub.wallet.check()
        except WalletExceeded as exc:
            return f"failed: {exc}"
        rec.spent += estimate  # reserved before awaiting
        hub.wallet.spent += estimate
        try:
            reply, metadata, state = await send(peer.url, dict(peer.headers), text, f"{session.id}:{rec.name}",
                                                max(1.0, deadline - time.time()), {DEADLINE_KEY: deadline}, card)
        except BaseException as exc:
            rec.spent -= estimate  # nothing was delivered: give the reservation back
            hub.wallet.spent -= estimate
            if isinstance(exc, A2AError | httpx.HTTPError | ValueError):
                return f"failed: {peer.name} did not answer: {exc}"
            raise
        charged = max(estimate, price.estimate(text, reply))
        reported = usd(metadata.get(COST_KEY, 0) or 0)
        if reported is None:
            session.trace.emit("error", task=session.id, agent=rec.name,
                               error=f"{peer.name} reported an invalid cost {metadata.get(COST_KEY)!r}")
        else:
            charged = max(charged, reported)
        cap = max(estimate, left)  # never more than the pool held when the message was sent
        if charged > cap:
            session.trace.emit("error", task=session.id, agent=rec.name,
                               error=f"{peer.name} reported ${charged}; charged the pool's ${cap:.6f}")
            charged = cap
        rec.spent += charged - estimate
        hub.wallet.spent += charged - estimate
        if state != DONE:
            reply = f"failed: {peer.name} ended in state {TaskState.Name(state)}: {reply}"
        reply = reply[:MAX_REPLY_CHARS]
        session.trace.emit("llm", task=session.id, agent=rec.name, tier=rec.tier, model=rec.model, text=reply,
                           tool_calls=[], cost_usd=charged)
        session.trace.receive(rec.name, session.id, list(peer.labels))  # what a peer says is outside input
        return reply

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")
