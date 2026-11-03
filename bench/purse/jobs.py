"""A job: one task with an exact answer, from one family of generators, at a visible level and a hidden difficulty."""
import random
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol

__all__ = ["ANSWER_NOTE", "BINS", "FAMILIES", "LEVELS", "WIDE", "WORKER_PROMPT", "Family", "InfeasibleCell", "Job",
           "extract_answer", "family", "level_bin", "make_job", "register_family", "same_integer"]

BINS = ("low", "mid", "high")
LEVELS = (1, 2, 3, 4, 5)
ANSWER_NOTE = "Put only your final answer inside <answer></answer>."
WORKER_PROMPT = "Solve the task exactly. " + ANSWER_NOTE

TAGGED = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
FENCE = re.compile(r"^```[\w-]*\n?(.*?)\n?```$", re.S)
NOISE = re.compile(r"[\s,_]")

# Wider hidden bins (purse pool --wide): within a level, the hidden bin moves difficulty further, so that knowing it is
# worth more (V5). Off, the generators make exactly the first round's pool.
WIDE: ContextVar[bool] = ContextVar("purse_wide_bins", default=False)


@dataclass(frozen=True)
class Job:
    """`id` is the job's slot in a project (j1 to j8), empty in the pool; `uid` names it in the pool and the table."""

    id: str
    uid: str
    family: str
    level: int
    hidden: str
    value: int
    prompt: str
    answer: str
    features: tuple[tuple[str, int], ...]


class Family(Protocol):
    name: str
    checkable: bool  # a cheap check (grid_check) tells a wrong answer from a right one

    def generate(self, level: int, hidden: str, rng: random.Random) -> tuple[str, str, dict[str, int]]:
        """The task text, its exact answer and the visible features, for a visible level and a hidden bin."""
        ...

    def correct(self, answer: str, response: str) -> bool:
        """Whether a worker's whole response gives `answer`."""
        ...


class InfeasibleCell(ValueError):
    """The generator cannot make a job of this family, level and hidden bin."""


FAMILIES: dict[str, Family] = {}


def register_family(f: Family) -> Family:
    FAMILIES[f.name] = f
    return f


def family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"no job family named {name!r}; known: {sorted(FAMILIES)}") from None


def extract_answer(response: str) -> str:
    """The last <answer>...</answer> in a response, else all of it, without a code fence, backticks or quotes."""
    found = TAGGED.findall(response)
    text = (found[-1] if found else response).strip()
    if fenced := FENCE.match(text):
        text = fenced.group(1).strip()
    for left, right in (("`", "`"), ('"', '"'), ("'", "'")):
        if len(text) >= 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text


def level_bin(level: int) -> str:
    """The hidden bin a visible level suggests: 1 and 2 low, 3 mid, 4 and 5 high."""
    return "low" if level <= 2 else "mid" if level == 3 else "high"


def make_job(family_name: str, level: int, hidden: str, seed: int, uid: str, value: int = 1) -> Job:
    task, answer, features = family(family_name).generate(level, hidden, random.Random(seed))
    return Job(id="", uid=uid, family=family_name, level=level, hidden=hidden, value=value,
               prompt=f"{task}\n\n{ANSWER_NOTE}", answer=answer, features=tuple(sorted(features.items())))


def same_integer(answer: str, response: str) -> bool:
    """The response's answer is the integer `answer`, ignoring spaces, thousands separators and a final period. In
    "expression = value" (how the calc agent replies) the value is what counts."""
    text = NOISE.sub("", extract_answer(response).rsplit("=", 1)[-1]).removesuffix(".")
    return re.fullmatch(r"-?\d+", text) is not None and _integer(text) == _integer(answer)


def _integer(text: str) -> str:
    """An integer's canonical digits, compared as text: int() refuses more than 4300 digits, and a model that loops
    can write far more."""
    digits = text.strip().lstrip("-").lstrip("0") or "0"
    return digits if digits == "0" or not text.strip().startswith("-") else "-" + digits
