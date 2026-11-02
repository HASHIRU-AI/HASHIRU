"""F4: apply k string operations in order. The level sets k and the start string's length; the hidden bin sets the
alphabet (letters; letters and digits; plus punctuation and spaces) and how many operations depend on positions."""
import math
import random
import string

from purse.jobs import WIDE, extract_answer, register_family

__all__ = ["LENGTH", "OPERATIONS", "STRINGS", "Strings", "caesar", "double_vowels", "drop_every", "reverse",
           "rotate_left", "swapcase"]

OPERATIONS = {1: 2, 2: 3, 3: 5, 4: 7, 5: 10}
LENGTH = {1: 8, 2: 12, 3: 16, 4: 24, 5: 32}
ALPHABET = {"low": string.ascii_letters, "mid": string.ascii_letters + string.digits,
            "high": string.ascii_letters + string.digits + "!?.,;:-#@&* "}  # no quotes or backticks: they frame answers
POSITIONAL = ("rotate_left", "drop_every")
ATTEMPTS = 100

Op = tuple[str, int]


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def reverse(s: str) -> str:
    return s[::-1]


def rotate_left(s: str, r: int) -> str:
    r %= max(len(s), 1)
    return s[r:] + s[:r]


def caesar(s: str, k: int) -> str:
    def shift(c: str) -> str:
        base = "a" if c.islower() else "A"
        return chr((ord(c) - ord(base) + k) % 26 + ord(base)) if c.isascii() and c.isalpha() else c
    return "".join(shift(c) for c in s)


def swapcase(s: str) -> str:
    return s.swapcase()


def drop_every(s: str, j: int) -> str:
    return "".join(c for i, c in enumerate(s, 1) if i % j)


def double_vowels(s: str) -> str:
    return "".join(c * 2 if c in "aeiouAEIOU" else c for c in s)


APPLY = {"reverse": lambda s, a: reverse(s), "rotate_left": rotate_left, "caesar": caesar,
         "swapcase": lambda s, a: swapcase(s), "drop_every": drop_every, "double_vowels": lambda s, a: double_vowels(s)}
DESCRIBE = {"reverse": "Reverse the string.",
            "rotate_left": "Rotate the string left by {a} characters (move the first {a} characters to the end).",
            "caesar": "Shift every letter forward {a} places in the alphabet, wrapping from z to a and keeping its "
                      "case; leave other characters as they are.",
            "swapcase": "Swap the case of every letter.",
            "drop_every": "Delete every {o} character (the {o}, the {o2}, and so on, counting from 1).",
            "double_vowels": "Double every vowel (a, e, i, o, u, in either case)."}


class Strings:
    name = "strings"
    checkable = False

    def generate(self, level: int, hidden: str, rng: random.Random) -> tuple[str, str, dict[str, int]]:
        k, length = OPERATIONS[level], LENGTH[level]
        if WIDE.get():  # --wide: the bin also sets the start string's length (half, as is, double)
            length = {"low": max(4, length // 2), "mid": length, "high": 2 * length}[hidden]
        for _ in range(ATTEMPTS):
            start = "".join(rng.choice(ALPHABET[hidden]) for _ in range(length))
            s, ops = start, []
            for name in self._names(k, hidden, rng):
                arg = self._arg(name, s, hidden, rng)
                s = APPLY[name](s, arg)
                ops.append((name, arg))
            if s and s == s.strip() and start == start.strip():
                steps = "\n".join(f"{i}. {DESCRIBE[n].format(a=a, o=ordinal(a), o2=ordinal(2 * a))}"
                                  for i, (n, a) in enumerate(ops, 1))
                task = f"Start with the string `{start}` and apply these operations in order:\n{steps}\n" \
                       "What is the final string?"
                return task, s, {"operations": k, "length": length}
        raise ValueError(f"strings level {level} {hidden}: no pipeline with a nonempty answer")  # not seen in tests

    def _names(self, k: int, hidden: str, rng: random.Random) -> list[str]:
        wide = WIDE.get()  # --wide: the low bin loses the Caesar shift; all but one high-bin step is positional
        if hidden == "low":
            simple = ("reverse", "swapcase") if wide else ("reverse", "caesar", "swapcase")
            return [rng.choice(simple) for _ in range(k)]
        names = [rng.choice(tuple(APPLY)) for _ in range(k)]
        if hidden == "high":
            m = k - 1 if wide else math.ceil(k / 2)
            names[:m] = [rng.choice(POSITIONAL) for _ in range(m)]
            rng.shuffle(names)
        return names

    def _arg(self, name: str, s: str, hidden: str, rng: random.Random) -> int:
        if name == "rotate_left":
            return rng.randint(1, max(1, len(s) - 1))
        if name == "caesar":
            return rng.randint(1, 13) if hidden == "low" else rng.randint(14, 25) if hidden == "high" else rng.randint(
                1, 25)
        if name == "drop_every":
            return rng.randint(3, 5)
        return 0

    def correct(self, answer: str, response: str) -> bool:
        return extract_answer(response) == answer


STRINGS = register_family(Strings())
