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


class FakeOllama:
    """Records loads and unloads; every local model loads as 100 MB, and sizes() reports `sizes`."""

    def __init__(self, sizes: dict[str, int] | None = None) -> None:
        self.loaded: list[str] = []
        self.unloaded: list[str] = []
        self._sizes = sizes or {}

    async def load(self, model: str) -> int:
        self.loaded.append(model)
        return 100

    async def unload(self, model: str) -> None:
        self.unloaded.append(model)

    async def sizes(self) -> dict[str, int]:
        return dict(self._sizes)
