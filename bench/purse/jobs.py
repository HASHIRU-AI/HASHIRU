"""A job: one task with an exact answer, from one family of generators, at a visible level and a hidden difficulty."""
import random
import re
from dataclasses import dataclass
from typing import Protocol

__all__ = ["ANSWER_NOTE", "BINS", "FAMILIES", "LEVELS", "WORKER_PROMPT", "Family", "InfeasibleCell", "Job",
           "extract_answer", "family", "make_job", "register_family"]

BINS = ("low", "mid", "high")
LEVELS = (1, 2, 3, 4, 5)
ANSWER_NOTE = "Put only your final answer inside <answer></answer>."
WORKER_PROMPT = "Solve the task exactly. " + ANSWER_NOTE

TAGGED = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
FENCE = re.compile(r"^```[\w-]*\n?(.*?)\n?```$", re.S)


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


def make_job(family_name: str, level: int, hidden: str, seed: int, uid: str, value: int = 1) -> Job:
    task, answer, features = family(family_name).generate(level, hidden, random.Random(seed))
    return Job(id="", uid=uid, family=family_name, level=level, hidden=hidden, value=value,
               prompt=f"{task}\n\n{ANSWER_NOTE}", answer=answer, features=tuple(sorted(features.items())))
