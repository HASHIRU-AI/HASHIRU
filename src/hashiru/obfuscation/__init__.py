"""Obfuscation: how private data reaches cloud models when private_mode is "obfuscate" (the default).

Text that carries `private` and goes to a cloud or remote agent is masked with the session's vault: sensitive values
become placeholders (⟦PERSON_1⟧) or generalizations ("a six-figure amount"). Text that comes back to a local agent,
a local tool or the user has its placeholders restored. Detection that fails makes the caller refuse (R3).
"""
from typing import TYPE_CHECKING, Any

from hashiru.obfuscation.detect import DETECT_PROMPT, Detector
from hashiru.obfuscation.vault import PLACEHOLDER, Span, Vault

if TYPE_CHECKING:
    from hashiru.session import Session

__all__ = ["DETECT_PROMPT", "PLACEHOLDER", "Detector", "Obfuscator", "Span", "Vault"]


class Obfuscator:
    def __init__(self, detector: Detector) -> None:
        self.detector = detector

    async def mask(self, session: "Session", text: str) -> str | None:
        """`text` as a cloud model may see it, or None if its sensitive values could not be found."""
        spans = await self.detector.detect(text)
        return None if spans is None else session.vault.mask(text, spans)

    async def mask_value(self, session: "Session", value: Any) -> Any | None:
        """A JSON value (e.g. a tool result) with every string masked, or None if detection failed."""
        found: dict[str, list[Span]] = {}
        for text in strings(value):
            if text not in found:
                spans = await self.detector.detect(text)
                if spans is None:
                    return None
                found[text] = spans
        return session.vault.mask_value(value, lambda text: found.get(text, []))

    @staticmethod
    def unmask(session: "Session", text: str) -> tuple[str, bool]:
        return session.vault.unmask(text)

    @staticmethod
    def unmask_value(session: "Session", value: Any) -> tuple[Any, bool]:
        return session.vault.unmask_value(value)


def strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for v in value for s in strings(v)]
    if isinstance(value, dict):
        return [s for v in value.values() for s in strings(v)]
    return []
