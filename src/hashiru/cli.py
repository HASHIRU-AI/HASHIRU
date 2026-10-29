"""Command line.

  hashiru ask "Summarize ..." [key=value ...]   one request to a fresh session's CEO
  hashiru chat [key=value ...]                  talk to the CEO turn by turn (Ctrl-D, Ctrl-C or /quit ends)

key=value pairs are Hydra overrides of hashiru/conf/config.yaml, e.g. budget_usd=1.0.
Keys for cloud models come from the environment: run with `uv run --env-file .env hashiru ...`.
"""
import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from hydra.errors import HydraException

from hashiru.config import Settings, load_settings
from hashiru.hub import Hub
from hashiru.runtime import ask_agent
from hashiru.session import USER, Session

__all__ = ["main"]


def deliveries(events: list[dict[str, Any]]) -> list[str]:
    """deliver_to_user goes straight to the user, not through the CEO's reply."""
    return [f"[{e['agent']} -> you] {e['args'].get('text', '')}" for e in events
            if e["kind"] == "tool" and e["name"] == "deliver_to_user"]


def with_history(session: Session, history: list[tuple[str, str]], text: str) -> str:
    """The CEO keeps no memory between messages, so each one carries the chat so far and who is still hired."""
    if not history:
        return text
    past = "\n".join(f"User: {said}\nYou: {reply}" for said, reply in history)
    live = [f"{a.name} ({a.role}, {a.tier} {a.model})" for a in session.live_hires("ceo")]
    return (f"Our conversation so far:\n{past}\nAgents you hired that are still live: {', '.join(live) or 'none'}."
            f"\n\nThe user's new message: {text}")


async def ask(settings: Settings, text: str) -> None:
    hub = Hub(settings)
    async with hub.serving():
        session = hub.new_session("ask")
        try:
            reply = await ask_agent(hub, session, USER, "ceo", hub.recall(session, text))
        finally:
            await hub.close_session(session)
    print("\n".join([*deliveries(session.trace.events), f"[ceo -> you] {reply}", f"trace: {session.trace.path}"]))


async def chat(settings: Settings) -> None:
    """The whole chat is one session and one trace; hires last between turns."""
    hub = Hub(settings)
    async with hub.serving():
        session = hub.new_session("chat")
        print(f"Chatting with the CEO ({settings.ceo_model}). Ctrl-D, Ctrl-C or /quit to end.\n"
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
    args = parser.parse_args(argv)
    try:
        settings = load_settings(args.overrides, args.config)
    except (ValueError, OSError, HydraException) as exc:  # a bad override or config file: one line, no traceback
        sys.exit(f"hashiru: {str(exc).splitlines()[0] if str(exc) else type(exc).__name__}")
    try:
        asyncio.run(ask(settings, args.text) if args.command == "ask" else chat(settings))
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"hashiru: {exc}")
