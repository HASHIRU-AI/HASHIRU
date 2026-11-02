"""F1: exact integer arithmetic."""
import ast
import operator
import random
import re

import pytest

from purse.families.arithmetic import ARITHMETIC, DIGITS, OPERATORS

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def evaluate(node: ast.AST) -> int:
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant):
        return int(node.value)
    assert isinstance(node, ast.BinOp)
    return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))


def expression(task: str) -> str:
    return task.split("\n\n", 1)[1]


def test_same_seed_same_job_and_answer_is_exact():
    first = ARITHMETIC.generate(3, "mid", random.Random(7))
    assert first == ARITHMETIC.generate(3, "mid", random.Random(7))
    task, answer, _ = first
    assert task.startswith("Compute the exact value of this expression:")
    assert evaluate(ast.parse(expression(task), mode="eval")) == int(answer)


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_levels_set_size(level):
    task, _, features = ARITHMETIC.generate(level, "high", random.Random(level))
    expr = expression(task)
    assert len(re.findall(r"[-+*]", expr)) == OPERATORS[level]
    assert {len(n) for n in re.findall(r"\d+", expr)} == {DIGITS[level]}
    assert features == {"operators": OPERATORS[level], "digits": DIGITS[level]}


def test_hidden_bins_change_carries():
    def big_digits(hidden: str) -> float:
        tasks = [ARITHMETIC.generate(3, hidden, random.Random(s))[0] for s in range(30)]
        return sum(sum(c >= "5" for c in re.sub(r"\D", "", expression(t))) for t in tasks) / 30
    assert big_digits("high") > big_digits("mid") > big_digits("low")


def test_correct_ignores_formatting():
    for response in ("<answer>1,234</answer>", "1234.", "`1234`", "<answer> 1 234 </answer>", "<answer>1234</answer>"):
        assert ARITHMETIC.correct("1234", response), response
    for response in ("1235", "12340", "<answer>-1234</answer>", "about 1234", ""):
        assert not ARITHMETIC.correct("1234", response), response
    assert ARITHMETIC.correct("-56", "<answer>-56</answer>")
