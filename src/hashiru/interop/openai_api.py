"""An OpenAI-compatible Chat Completions API at /v1, so any OpenAI-style chat app can use HASHIRU as a model.

The model is "hashiru" and its answers come from the CEO. With stream=true, the agents' work arrives while it
happens as `reasoning_content` deltas (most chat apps show these as the model's thinking); what agents deliver
straight to the user and the CEO's reply arrive as `content`. Responses and chunks also carry a `hashiru` object
(the session id, each chunk's event, the cost) that other clients ignore.
"""
import asyncio
import contextlib
import json
import secrets
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING, Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from hashiru.interop.activity import describe, public_event
from hashiru.interop.conversations import Busy, Conversation, History, ShuttingDown, delivered

if TYPE_CHECKING:
    from hashiru.hub import Hub

__all__ = ["MODEL", "PRIVATE_HEADER", "SESSION_HEADER", "OpenAIRoutes", "error", "parse_messages", "wants_private"]

MODEL = "hashiru"
SESSION_HEADER = "x-hashiru-session"  # optional: the conversation a request continues (the chat page sends it)
PRIVATE_HEADER = "x-hashiru-private"  # optional, 1 or true: the conversation is private from this message on
MAX_BODY = 2_000_000


class Problem(Exception):
    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.status, self.code = status, code


def error(status: int, message: str, code: str | None = None) -> JSONResponse:
    kind = {401: "authentication_error", 403: "permission_error", 404: "not_found_error",
            429: "rate_limit_error"}.get(status, "invalid_request_error" if status < 500 else "server_error")
    return JSONResponse({"error": {"message": message, "type": kind, "param": None, "code": code}},
                        status_code=status, headers={"www-authenticate": "Bearer"} if status == 401 else None)


async def read_json(request: Request) -> Any:
    """The JSON body, at most MAX_BODY bytes. Only application/json: a cross-site form post cannot be one."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise Problem(415, "send the request as application/json")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY:
            raise Problem(413, f"the request is larger than {MAX_BODY} bytes")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise Problem(400, "the body is not valid JSON") from None


def text_of(content: Any) -> str:
    """A message's text: a string, or the text parts of a list of content parts (images and files are skipped)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content
                         if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


def wants_private(value: str) -> bool:
    """Whether a PRIVATE_HEADER value turns privacy on."""
    return value.strip().lower() in ("1", "true", "yes", "on")


def parse_messages(messages: Any) -> tuple[History, str, str]:
    """(the chat before the last user message, that message, system instructions) from OpenAI-style messages.

    Consecutive user messages are one turn (a turn that got no reply). Tool messages are skipped: HASHIRU's tools
    are its own.
    """
    if not isinstance(messages, list) or not messages:
        raise Problem(400, "messages must be a non-empty list")
    past: History = []
    pending: list[str] = []
    instructions: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise Problem(400, "each message must be an object")
        role, text = message.get("role"), text_of(message.get("content"))
        if role in ("system", "developer"):
            instructions.append(text)
        elif role == "user":
            pending.append(text)
        elif role == "assistant":
            past.append(("\n\n".join(pending), text))
            pending = []
    said = "\n\n".join(pending).strip()
    if not said:
        raise Problem(400, "the last message must be a user message with text")
    return past, said, "\n\n".join(i for i in instructions if i.strip())


class OpenAIRoutes:
    def __init__(self, hub: "Hub") -> None:
        self.hub = hub

    def routes(self) -> list[Route]:
        return [Route("/v1/models", self.models, methods=["GET"]),
                Route("/v1/chat/completions", self.complete, methods=["POST"]),
                Route("/v1/hashiru/info", self.info, methods=["GET"]),
                Route("/v1/hashiru/sessions/{session}", self.session, methods=["GET", "DELETE"])]

    async def models(self, request: Request) -> Response:
        return JSONResponse({"object": "list", "data": [
            {"id": MODEL, "object": "model", "created": 0, "owned_by": "hashiru"}]})

    async def info(self, request: Request) -> Response:
        """What a client can show about this hub: the CEO's model, what it may hire, and the budgets."""
        s = self.hub.settings
        return JSONResponse({"ceo_model": s.ceo_model, "models": self.hub.hireable(), "budget_usd": s.budget_usd,
                             "max_usd": s.max_usd, "tool_creation": s.tool_creation, "allow_hiring": s.allow_hiring,
                             "open_sessions": len(self.hub.conversations.open_), "max_sessions": s.max_sessions,
                             "private_by_default": s.private_by_default,
                             "private_mode": "obfuscate" if self.hub.obfuscating() else "block",
                             "private_ceo_model": self.hub.private_ceo_model()})

    async def session(self, request: Request) -> Response:
        """GET: the conversation's agents and pools. DELETE: close it (fires its agents, stops its work)."""
        convs = self.hub.conversations
        conv = convs.get(request.path_params["session"])
        if conv is None:
            return error(404, "no open conversation with that id", "session_not_found")
        if request.method == "DELETE":
            await convs.close(conv)
            return JSONResponse({"closed": conv.id})
        return JSONResponse(convs.state(conv))

    async def complete(self, request: Request) -> Response:
        try:
            body = await read_json(request)
            if not isinstance(body, dict):
                raise Problem(400, "the body must be a JSON object")
            past, said, instructions = parse_messages(body.get("messages"))
            conv = await self._conversation(past, request.headers.get(SESSION_HEADER, ""))
            turn = self.hub.conversations.ask(conv, said, instructions,
                                              private=wants_private(request.headers.get(PRIVATE_HEADER, "")))
        except Problem as p:
            return error(p.status, str(p), p.code)
        except ShuttingDown as exc:
            return error(503, str(exc), "shutting_down")
        except Busy as exc:
            return error(429, str(exc), "busy")
        reply_id, created = f"chatcmpl-{secrets.token_hex(12)}", int(time.time())
        headers = {SESSION_HEADER: conv.id}
        if body.get("stream"):
            usage = isinstance(body.get("stream_options"), dict) and bool(body["stream_options"].get("include_usage"))
            return StreamingResponse(self._stream(conv, turn, reply_id, created, usage), media_type="text/event-stream",
                                     headers=headers | {"cache-control": "no-cache", "x-accel-buffering": "no"})
        # Starlette notices a client leaving only for streams, so watch for it here: a turn nobody waits for stops.
        work = asyncio.ensure_future(self._collect(turn))
        gone = asyncio.ensure_future(self._disconnected(request))
        try:
            await asyncio.wait({work, gone}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            gone.cancel()
            if not work.done():
                work.cancel()  # the turn's cleanup closes the conversation
        if work.cancelled():
            return error(499, "the client closed the request")
        try:
            lines, answer = work.result()
        except Exception as exc:  # the turn failed outside the agents (e.g. the wallet cap)
            return error(500, f"{type(exc).__name__}: {exc}")
        message = {"role": "assistant", "content": answer["text"]} | (
            {"reasoning_content": "\n".join(lines)} if lines else {})
        return JSONResponse({
            "id": reply_id, "object": "chat.completion", "created": created, "model": MODEL,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop", "logprobs": None}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "hashiru": {"session": conv.id, "cost_usd": answer["cost_usd"]}}, headers=headers)

    @staticmethod
    async def _collect(turn: AsyncGenerator[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
        lines: list[str] = []
        answer: dict[str, Any] = {}
        async with contextlib.aclosing(turn) as events:
            async for item in events:
                if item["kind"] == "answer":
                    answer = item
                elif line := describe(item):
                    lines.append(line)
        return lines, answer

    @staticmethod
    async def _disconnected(request: Request) -> None:
        """Returns when the client goes away (the body has been read, so nothing else arrives before that)."""
        while (await request.receive())["type"] != "http.disconnect":
            pass

    async def _conversation(self, past: History, named: str) -> Conversation:
        """The conversation this chat continues, or a new one that is shown the chat so far."""
        convs = self.hub.conversations
        conv = convs.find(past, named)
        if conv is not None and conv.busy:
            raise Problem(409, "this conversation is still answering the last message", "conversation_busy")
        return conv or await convs.open("chat", past)

    async def _stream(self, conv: Conversation, turn: AsyncGenerator[dict[str, Any]], reply_id: str, created: int,
                      usage: bool) -> AsyncIterator[str]:
        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> str:
            return "data: " + json.dumps({
                "id": reply_id, "object": "chat.completion.chunk", "created": created, "model": MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": None}],
                "hashiru": {"session": conv.id, **extra}}) + "\n\n"

        yield chunk({"role": "assistant", "content": ""})
        parts = 0
        try:
            async with contextlib.aclosing(turn) as events:
                async for item in events:
                    if item["kind"] == "idle":
                        yield ": keepalive\n\n"
                    elif item["kind"] == "answer":
                        yield chunk({"content": ("\n\n" if parts else "") + item["reply"]})
                        yield chunk({}, "stop", cost_usd=item["cost_usd"])
                    elif text := delivered(item):
                        yield chunk({"content": ("\n\n" if parts else "") + text}, event=public_event(item))
                        parts += 1
                    elif line := describe(item):
                        yield chunk({"reasoning_content": line + "\n"}, event=public_event(item))
                    else:
                        yield chunk({}, event=public_event(item))
        except Exception as exc:  # the turn failed outside the agents (e.g. the wallet cap): tell the client
            yield "data: " + json.dumps({"error": {"message": f"{type(exc).__name__}: {exc}",
                                                   "type": "server_error", "param": None, "code": None}}) + "\n\n"
        if usage:
            yield "data: " + json.dumps({
                "id": reply_id, "object": "chat.completion.chunk", "created": created, "model": MODEL, "choices": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}) + "\n\n"
        yield "data: [DONE]\n\n"
