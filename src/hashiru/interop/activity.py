"""Trace events as the user sees them while the agents work: one line each, plus a small public form.

The lines are what OpenAI-style chat apps show as the model's reasoning, what A2A clients get as progress
messages and what MCP clients get as progress notifications.
"""
import json
from typing import Any

from hashiru.session import USER

__all__ = ["describe", "public_event"]

CLIP = 240
RULE_NAMES = {"R3": "private stays local"}
PUBLIC_KEYS = ("seq", "ts", "kind", "agent", "to", "name", "tier", "model", "cost_usd", "labels")


def clip(text: Any, limit: int = CLIP) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def money(value: Any) -> str:
    """Dollars as people read them: $0.50, $0.0042, $0.00."""
    try:
        usd = float(value or 0)
    except (TypeError, ValueError):
        return str(value)
    return f"${usd:.2f}" if usd == 0 or abs(usd) >= 0.1 else f"${usd:.4f}"


def _args(args: dict[str, Any]) -> str:
    return clip(", ".join(f"{k}={json.dumps(v, default=str)}" for k, v in args.items()), 120)


def _tool(e: dict[str, Any]) -> str | None:
    agent, name, args, result = e["agent"], e["name"], e.get("args", {}), e.get("result", {})
    if "error" in result:
        return f"{agent}: {name} failed: {clip(result['error'])}"
    if name == "hire_agent":
        tools = f", tools {', '.join(args['tools'])}" if args.get("tools") else ""
        return (f"{agent} hired {result.get('hired')} as a {args.get('role')} on {args.get('model')} "
                f"with {money(args.get('budget_usd'))}{tools}")
    if name == "fire_agent":
        return f"{agent} fired {', '.join(result.get('fired', [])) or 'nobody'}"
    if name == "check_usage":
        return (f"{agent} checked {result.get('pool')}'s pool: {money(result.get('api_usd'))} left, "
                f"{result.get('free_mb')} MB free")
    if name == "deliver_to_user":
        return f"{agent} sent you a message"
    return f"{agent} called {name}({_args(args)})"


def describe(e: dict[str, Any]) -> str | None:
    """One line for an event worth showing, or None (the user's own message, the final reply, bookkeeping)."""
    kind, agent = e["kind"], e.get("agent", "")
    if kind == "msg":
        if USER in (agent, e.get("to")):
            return None
        return f"{agent} → {e['to']}: {clip(e.get('text', ''))}"
    if kind == "tool":
        return _tool(e)
    if kind == "llm":
        text = (e.get("text") or "").strip()
        return f"{agent}: {clip(text)}" if text and e.get("tool_calls") else None
    if kind == "error":
        return f"{agent}: {clip(e.get('error', ''))}"
    if kind == "recall":
        return f"added {len(e.get('results', []))} saved memories to your message"
    if kind == "escalate":
        return f"{agent} moved from {e.get('from_model')} to {e.get('to_model')} after: {clip(e.get('why', ''))}"
    if kind == "policy":
        rule = e.get("rule", "")
        name = RULE_NAMES.get(rule)
        return f"{rule} ({name}): {clip(e.get('why', ''))}" if name else f"{rule}: {clip(e.get('why', ''))}"
    return None


def public_event(e: dict[str, Any]) -> dict[str, Any]:
    """The part of an event a client needs to draw it: no full arguments or results."""
    out: dict[str, Any] = {k: e[k] for k in PUBLIC_KEYS if k in e}
    if e["kind"] == "llm":
        out["tool_calls"] = list(e.get("tool_calls", []))
    if e["kind"] == "tool" and isinstance(e.get("result"), dict) and "error" in e["result"]:
        out["failed"] = True  # e.g. a hire that did not happen: not counted as one
    line = describe(e)
    if line:
        out["line"] = line
    return out
