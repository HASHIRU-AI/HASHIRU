"""Command line.

  hashiru ask "Summarize ..." [key=value ...]   one request to a fresh session's CEO
  hashiru chat [key=value ...]                  talk to the CEO turn by turn (Ctrl-D, Ctrl-C or /quit ends)
  hashiru serve [key=value ...]                 the chat page, OpenAI-compatible API, A2A agent and MCP server

key=value pairs are Hydra overrides of hashiru/conf/config.yaml, e.g. budget_usd=1.0.
Keys for cloud models come from the environment: run with `uv run --env-file .env hashiru ...`.
"""
import argparse
import asyncio
import contextlib
import logging
import signal
import sys
from typing import Any

from hydra.errors import HydraException

from hashiru.approvals import Answer, Decision, Request
from hashiru.config import Settings, load_settings
from hashiru.hub import Hub
from hashiru.interop.conversations import with_history
from hashiru.privacy import holds_private
from hashiru.runtime import ask_agent
from hashiru.session import USER, Session

__all__ = ["main"]


def deliveries(events: list[dict[str, Any]]) -> list[str]:
    """deliver_to_user goes straight to the user, not through the CEO's reply."""
    return [f"[{e['agent']} -> you] {e['args'].get('text', '')}" for e in events
            if e["kind"] == "tool" and e["name"] == "deliver_to_user"]


def terminal(session: Session) -> Answer:
    """Approvals asked on the terminal, one at a time. End of input denies."""
    lock = asyncio.Lock()

    async def answer(request: Request) -> Decision:
        async with lock:
            print(f"\n[approval] {request.what}", flush=True)
            try:
                said = await asyncio.to_thread(input, "approve? (approve, or say why not) ")
            except EOFError:
                said = ""
        return Decision.of(said, session.trace.outgoing(USER, session.id, said))
    return answer


def attend(session: Session) -> None:
    """Ask the person at the terminal when work needs approval; without one (a script), such work is denied."""
    if sys.stdin.isatty():
        session.approvals.answer = terminal(session)


async def ask(settings: Settings, text: str) -> None:
    hub = Hub(settings)
    async with hub.serving():
        session = hub.new_session("ask")
        attend(session)
        try:
            why = await hub.make_private(session) if settings.private_by_default else None
            reply = f"error: {why}" if why else await ask_agent(hub, session, USER, "ceo", hub.recall(session, text))
        finally:
            await hub.close_session(session)
    print("\n".join([*deliveries(session.trace.events), f"[ceo -> you] {reply}", f"trace: {session.trace.path}"]))


async def chat(settings: Settings) -> None:
    """The whole chat is one session and one trace; hires last between turns."""
    hub = Hub(settings)
    async with hub.serving():
        session = hub.new_session("chat")
        attend(session)
        if settings.private_by_default and (why := await hub.make_private(session)):
            await hub.close_session(session)
            raise SystemExit(f"error: {why}")
        print(f"Chatting with the CEO ({session.ledger.agents['ceo'].model}). Ctrl-D, Ctrl-C or /quit to end.\n"
              f"trace: {session.trace.path}")
        history: list[tuple[str, str]] = []
        try:
            while True:
                uvicorn_handler = signal.signal(signal.SIGINT, signal.default_int_handler)  # uvicorn swallows ^C
                try:
                    text = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                finally:
                    signal.signal(signal.SIGINT, uvicorn_handler)
                if text == "/quit":
                    break
                if not text:
                    continue
                if holds_private(session, USER) and (why := await hub.make_private(session)):
                    print(f"[ceo -> you] error: {why}")  # a private delivery made the chat private, and it cannot be
                    break
                seen = len(session.trace.events)
                reply = await ask_agent(hub, session, USER, "ceo",
                                        with_history(session, history, hub.recall(session, text)))
                turn = session.trace.events[seen:]
                cost = sum(e.get("cost_usd", 0) for e in turn if e["kind"] == "llm")
                hires = [e["result"]["hired"] for e in turn
                         if e["kind"] == "tool" and e["name"] == "hire_agent" and "hired" in e["result"]]
                print("\n".join([*deliveries(turn), f"[ceo -> you] {reply}"]))
                print(f"(this turn: ${cost:.4f}" + (f", hired {', '.join(hires)})" if hires else ")"))
                if reply.startswith("error: "):
                    break  # a timed-out CEO run may still be going; a new turn would race it in the same trace
                history.append((text, reply))
        finally:
            await hub.close_session(session)
    print(f"trace: {session.trace.path}")


async def serve(settings: Settings) -> None:
    """Serve until Ctrl-C or SIGTERM, then close every conversation (firing agents, unloading models)."""
    hub = Hub(settings)
    async with hub.serving():
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):  # replaces uvicorn's handlers, which would re-raise ^C on exit
            with contextlib.suppress(NotImplementedError):  # Windows: uvicorn's handlers stay
                loop.add_signal_handler(sig, stop.set)
        url = settings.base_url
        print(f"HASHIRU is serving (Ctrl-C stops it). CEO: {settings.ceo_model}, ${settings.budget_usd} per chat.\n"
              f"  chat page       {url}/\n"
              f"  OpenAI API      {url}/v1   (model \"hashiru\")\n"
              f"  A2A agent       {url}/.well-known/agent-card.json\n"
              f"  MCP server      {url}/public/mcp\n"
              + ("  clients send the api_key as a bearer token" if settings.api_key
                 else "  no api_key: only this machine can connect"), flush=True)
        stopped = asyncio.ensure_future(stop.wait())
        await asyncio.wait([stopped, asyncio.ensure_future(hub.until_stopped())],
                           return_when=asyncio.FIRST_COMPLETED)
        stopped.cancel()
        print("stopping: closing conversations", flush=True)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.WARNING, force=True)
    # Known-benign SDK noise: MCP's OpenTelemetry spans end in a different task context than they began
    # (logged, never raised; no exporter is configured), and the A2A queue warns after each reply is sent.
    logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)
    logging.getLogger("a2a.server.events.event_queue_v2").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(prog="hashiru", description=(__doc__ or "").splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ask_cmd = sub.add_parser("ask", help="send one request to the CEO")
    ask_cmd.add_argument("text")
    ask_cmd.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. budget_usd=1.0")
    ask_cmd.add_argument("--config", help="YAML config file (default: $HASHIRU_CONFIG)")
    chat_cmd = sub.add_parser("chat", help="talk to the CEO turn by turn")
    chat_cmd.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. budget_usd=1.0")
    chat_cmd.add_argument("--config", help="YAML config file (default: $HASHIRU_CONFIG)")
    serve_cmd = sub.add_parser("serve", help="serve the chat page and the public APIs")
    serve_cmd.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. port=9300")
    serve_cmd.add_argument("--config", help="YAML config file (default: $HASHIRU_CONFIG)")
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.overrides, args.config)
    except (ValueError, OSError, HydraException) as exc:  # a bad override or config file: one line, no traceback
        sys.exit(f"hashiru: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
    try:
        commands = {"ask": lambda: ask(settings, args.text), "chat": lambda: chat(settings),
                    "serve": lambda: serve(settings)}
        asyncio.run(commands[args.command]())
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"hashiru: {exc}")
