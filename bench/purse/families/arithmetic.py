"""F1: evaluate a fully parenthesized integer expression exactly. The level sets its size; the hidden bin sets how
many multiplications it has and how big its digits are (so how many carries and borrows)."""
import random

from purse.jobs import WIDE, InfeasibleCell, register_family, same_integer

__all__ = ["ARITHMETIC", "DIGITS", "OPERATORS", "Arithmetic"]

OPERATORS = {1: 2, 2: 4, 3: 8, 4: 14, 5: 22}
DIGITS = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8}
TIMES_SHARE = {"low": 0.2, "mid": 0.35, "high": 0.5}  # the rest: + 70%, - 30%
WIDE_TIMES_SHARE = {"low": 0.1, "mid": 0.35, "high": 0.7}  # --wide: more multiplications in the high bin
WIDE_DIGITS = {"low": -1, "mid": 0, "high": 2}  # --wide: digits per number, beside the level's
DIGIT_SET = {"low": "01234", "mid": "0123456789", "high": "56789"}
ATTEMPTS = 100


class Arithmetic:
    name = "arithmetic"
    checkable = False

    def generate(self, level: int, hidden: str, rng: random.Random) -> tuple[str, str, dict[str, int]]:
        digits = max(1, DIGITS[level] + WIDE_DIGITS[hidden]) if WIDE.get() else DIGITS[level]
        cap = 10 ** (3 * digits)
        for _ in range(ATTEMPTS):
            text, value, biggest = self._tree(OPERATORS[level], digits, hidden, rng, cap)
            if biggest < cap:
                return (f"Compute the exact value of this expression:\n\n{text[1:-1]}", str(value),
                        {"operators": OPERATORS[level], "digits": digits})
        raise InfeasibleCell(f"arithmetic level {level} {hidden}: no expression within 10**{3 * digits}")

    def _tree(self, ops: int, digits: int, hidden: str, rng: random.Random, cap: int) -> tuple[str, int, int]:
        """(text, value, largest absolute intermediate) of a random tree with `ops` operators. A product that would
        reach `cap` becomes a sum, so deep trees of big numbers stay feasible."""
        if ops == 0:
            chars = DIGIT_SET[hidden]
            first = rng.choice(chars.replace("0", "") or chars)
            text = first + "".join(rng.choice(chars) for _ in range(digits - 1))
            return text, int(text), int(text)
        left = rng.randint(0, ops - 1)
        lt, lv, lb = self._tree(left, digits, hidden, rng, cap)
        rt, rv, rb = self._tree(ops - 1 - left, digits, hidden, rng, cap)
        roll, times = rng.random(), (WIDE_TIMES_SHARE if WIDE.get() else TIMES_SHARE)[hidden]
        if roll < times and abs(lv * rv) < cap:
            op, value = "*", lv * rv
        elif roll < times + (1 - times) * 0.7:
            op, value = "+", lv + rv
        else:
            op, value = "-", lv - rv
        return f"({lt} {op} {rt})", value, max(lb, rb, abs(value))

    def correct(self, answer: str, response: str) -> bool:
        return same_integer(answer, response)


ARITHMETIC = register_family(Arithmetic())
