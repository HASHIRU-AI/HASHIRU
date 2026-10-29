# /// script
# requires-python = ">=3.13"
# dependencies = ["a2a-sdk[http-server]==1.2.2", "uvicorn==0.54.0"]
# ///
"""A remote A2A agent that HASHIRU managers can hire: exact arithmetic, which small models often get wrong.

It needs no LLM. It answers the last arithmetic expression in each message (+ - * / // % and whole-number
powers **, with parentheses) in exact rational arithmetic, and says what it read: "(17.5 / 100) * 2340 = 409.5",
or "1 / 3 ≈ 0.333333333333" when a decimal would not end. It declares a price on its Agent Card through HASHIRU's
cost extension (docs/a2a-cost-extension.md) and reports what each request cost, so the hub can charge the hire.

    uv run examples/calc_a2a_agent.py --port 9300 --price 0.001

Then list it in a hub's config (examples/hashiru.yaml does) and managers can hire it as a worker, a2a/calc:

    a2a_peers:
      calc: {url: http://127.0.0.1:9300/}
"""
import argparse
import ast
import math
import operator
import re
from collections.abc import Callable
from decimal import Decimal, Inexact, localcontext
from fractions import Fraction

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

COST_EXTENSION = "https://github.com/HASHIRU-AI/HASHIRU/blob/main/docs/a2a-cost-extension.md"
COST_KEY = "hashiru_cost_usd"  # reply metadata: what the request cost
HELP = "send one arithmetic expression, such as (17.5 / 100) * 2340"
NUMBER = r"(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
EXPRESSION = re.compile(rf"(?:{NUMBER}|[+\-*/%()\s])+")
DATE = re.compile(r"\d{4}-\d\d-\d\d")  # 2026-12-25 is a date, not a subtraction
MAX_TEXT = 4_000          # it reads only the end of a longer message
MAX_CHARS = 200           # longest expression it will read
MAX_EXPONENT = 1_000      # largest exponent in a literal such as 1e300
MAX_BITS = 4_096          # largest power it will compute, by the bits of the base's numerator or denominator
MAX_SHOWN_BITS = 12_000   # largest whole number it will write out (about 3,600 digits)
EXACT_DIGITS, DIGITS = 40, 12  # a decimal that ends within 40 digits is shown exactly, others to 12

OPS: dict[type[ast.operator], Callable[[Fraction, Fraction], Fraction | int]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}


class Unsupported(ArithmeticError):
    """Arithmetic this agent does not do."""


def literal(source: str) -> Fraction:
    """A number as written: 0.1 is one tenth, not the nearest binary float."""
    exponent = source.lower().partition("e")[2]
    if exponent and abs(int(exponent)) > MAX_EXPONENT:
        raise OverflowError
    return Fraction(source)


def power(base: Fraction, exponent: Fraction) -> Fraction:
    if exponent.denominator != 1:
        raise Unsupported("only whole-number powers")
    # The result's numerator and denominator grow to `exponent` times the base's, whatever the base's size.
    size = max(base.numerator.bit_length(), base.denominator.bit_length())
    if base not in (0, 1, -1) and abs(exponent) * size > MAX_BITS:
        raise OverflowError
    return base ** int(exponent)


def value(node: ast.expr, source: str) -> Fraction:
    """Evaluate a parsed expression of numbers and arithmetic operators; ValueError for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return literal(ast.get_source_segment(source, node) or "")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        operand = value(node.operand, source)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return power(value(node.left, source), value(node.right, source))
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return Fraction(OPS[type(node.op)](value(node.left, source), value(node.right, source)))
    raise ValueError("not arithmetic")


def show(number: Fraction) -> str:
    """'= <exact value>', or '≈ <12 significant digits>' for a decimal that does not end."""
    if number.denominator == 1:
        if number.numerator.bit_length() > MAX_SHOWN_BITS:
            raise OverflowError
        return f"= {number.numerator}"
    for digits, relation in ((EXACT_DIGITS, "="), (DIGITS, "≈")):
        with localcontext(prec=digits) as context:  # Decimal: no float overflow or underflow
            context.clear_flags()
            shown = Decimal(number.numerator) / Decimal(number.denominator)
            if relation == "≈" or not context.flags[Inexact]:
                return f"{relation} {shown:g}"
    raise AssertionError("unreachable")


def glued(text: str, match: re.Match[str]) -> bool:
    """Whether a candidate touches a word, as in ISO-8601 or x2+1."""
    found, start, end = match.group(), match.start(), match.end()
    return bool((found[0].strip() and start > 0 and text[start - 1].isalnum())
                or (found[-1].strip() and end < len(text) and text[end].isalnum()))


def calculate(text: str) -> str:
    """'<expression> = <value>' for the last arithmetic expression in `text`, or a line starting with "error:"."""
    text = text[-MAX_TEXT:]
    for match in reversed(list(EXPRESSION.finditer(text))):
        expression = " ".join(match.group().split()).rstrip(".")  # a sentence's full stop is not a decimal point
        if (len(expression) > MAX_CHARS or not any(c.isdigit() for c in expression) or glued(text, match)
                or DATE.search(expression)):
            continue
        try:
            tree = ast.parse(expression, mode="eval").body
            if isinstance(tree, ast.Constant):
                continue  # a lone number is not a question
            return f"{expression} {show(value(tree, expression))}"
        except (SyntaxError, ValueError):
            continue
        except ZeroDivisionError:
            return "error: division by zero"
        except OverflowError:
            return "error: the result would be too large"
        except Unsupported as exc:
            return f"error: {exc}"
    return f"error: {HELP}"


def agent_card(url: str, price_usd: float) -> AgentCard:
    price = Struct()
    price.update({"currency": "USD", "per_message_usd": price_usd})
    return AgentCard(
        name="calc", description=f"Exact arithmetic: {HELP}.", version="1",
        default_input_modes=["text/plain"], default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False, extensions=[AgentExtension(
            uri=COST_EXTENSION, description="What a message to this agent costs (USD).", required=False,
            params=price)]),
        supported_interfaces=[AgentInterface(protocol_binding="JSONRPC", url=url, protocol_version="1.0")],
        skills=[AgentSkill(id="arithmetic", name="Exact arithmetic", description=HELP, tags=["math"],
                           examples=["(17.5 / 100) * 2340"])])


class Calculator(AgentExecutor):
    def __init__(self, price_usd: float) -> None:
        self.price_usd = price_usd

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.current_task is None:  # open a task so the reply can complete it, rather than leave it pending
            history = [context.message] if context.message is not None else []
            await event_queue.enqueue_event(Task(id=context.task_id or "", context_id=context.context_id or "",
                                                 status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                                                 history=history))
        updater = TaskUpdater(event_queue, context.task_id or "", context.context_id or "")
        answer = calculate(get_message_text(context.message) if context.message else "")
        await updater.complete(updater.new_agent_message([new_text_part(answer)],
                                                         metadata={COST_KEY: self.price_usd}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("every answer is immediate")


def build_app(url: str, price_usd: float) -> Starlette:
    """The agent's card at /.well-known/agent-card.json and its JSON-RPC endpoint at /, for `url`."""
    card = agent_card(url, price_usd)
    # InMemoryTaskStore keeps every task; an agent meant to run for months would bound or persist it.
    handler = DefaultRequestHandler(agent_executor=Calculator(price_usd), task_store=InMemoryTaskStore(),
                                    agent_card=card)
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, "/"))


def usd(text: str) -> float:
    price = float(text)
    if not math.isfinite(price) or price < 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a price: use a finite number >= 0")
    return price


def main() -> None:
    parser = argparse.ArgumentParser(description="An A2A agent that does exact arithmetic for a price.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--price", type=usd, default=0.001, help="USD per message (default 0.001)")
    args = parser.parse_args()
    uvicorn.run(build_app(f"http://{args.host}:{args.port}/", args.price), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
