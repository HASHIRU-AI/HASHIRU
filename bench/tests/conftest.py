"""Shared fakes for PURSE's tests: no API keys, no Ollama, no tokens spent."""
import random
from typing import Any

import pytest

from purse.jobs import FAMILIES, extract_answer, register_family


class FakeFamily:
    """A family whose answer is the random number in its task text."""

    name = "fake"
    checkable = False

    def generate(self, level: int, hidden: str, rng: random.Random) -> tuple[str, str, dict[str, int]]:
        n = rng.randint(1, 9) if level > 1 else 1
        return f"Say {n}.", str(n), {"size": level}

    def correct(self, answer: str, response: str) -> bool:
        return extract_answer(response) == answer


@pytest.fixture
def fake_family() -> Any:
    family = FAMILIES.get("fake") or register_family(FakeFamily())
    yield family
