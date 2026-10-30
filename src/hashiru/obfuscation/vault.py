"""A session's vault: reversible placeholders for sensitive values, and the one-way generalizations.

mask() replaces each detected value with a placeholder such as ⟦PERSON_1⟧, the same one every time the value comes
up in the session, or with its generalization ("a six-figure amount") when the detector gave one. unmask() puts the
values back, tolerating the bracket changes models make ([PERSON_1], ⟦ PERSON_1 ⟧). Only the vault, which stays on
this machine, knows the values.
"""
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["PLACEHOLDER", "Span", "Vault"]

PLACEHOLDER = re.compile(r"[⟦\[]{1,2}\s*([A-Z][A-Z_]*?)_(\d+)\s*[⟧\]]{1,2}")
# Presidio's entity names, shortened for placeholders. Anything else keeps its own name.
LABELS = {"EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE", "CREDIT_CARD": "CARD", "IP_ADDRESS": "IP",
          "IBAN_CODE": "IBAN", "US_BANK_NUMBER": "ACCOUNT", "CRYPTO": "WALLET", "DATE_TIME": "DATE",
          "US_SSN": "ID", "US_ITIN": "ID", "US_PASSPORT": "ID", "US_DRIVER_LICENSE": "ID", "MEDICAL_LICENSE": "ID",
          "UK_NHS": "ID", "LOCATION": "PLACE", "ORGANIZATION": "ORG", "CONTACT": "CONTACT"}


@dataclass(frozen=True)
class Span:
    """A sensitive value at text[start:end]. `general`, if set, replaces it instead of a placeholder."""

    start: int
    end: int
    kind: str
    general: str = ""


def label(kind: str) -> str:
    out = LABELS.get(kind, kind).upper()
    return re.sub(r"[^A-Z_]", "_", out).strip("_") or "VALUE"


class Vault:
    def __init__(self) -> None:
        self._placeholders: dict[str, str] = {}  # value -> "PERSON_1"
        self._values: dict[str, str] = {}        # "PERSON_1" -> value
        self._counts: dict[str, int] = {}

    def placeholder(self, kind: str, value: str) -> str:
        key = self._placeholders.get(value)
        if key is None:
            name = label(kind)
            self._counts[name] = self._counts.get(name, 0) + 1
            key = f"{name}_{self._counts[name]}"
            self._placeholders[value], self._values[key] = key, value
        return f"⟦{key}⟧"

    def mask(self, text: str, spans: list[Span]) -> str:
        """`text` with every span replaced. Overlapping spans keep the longer one."""
        chosen: list[Span] = []
        for s in sorted(spans, key=lambda s: (s.start - s.end, s.start)):  # longest first
            if s.end > s.start and all(s.end <= c.start or s.start >= c.end for c in chosen):
                chosen.append(s)
        out, at = [], 0
        for s in sorted(chosen, key=lambda s: s.start):
            out += [text[at:s.start], s.general or self.placeholder(s.kind, text[s.start:s.end])]
            at = s.end
        return "".join([*out, text[at:]])

    def unmask(self, text: str) -> tuple[str, bool]:
        """`text` with known placeholders restored, and whether any was."""
        restored = False

        def put_back(match: re.Match[str]) -> str:
            nonlocal restored
            value = self._values.get(f"{match[1]}_{match[2]}")
            if value is None:
                return match[0]
            restored = True
            return value
        return PLACEHOLDER.sub(put_back, text), restored

    def mask_value(self, value: Any, spans_for: Callable[[str], list[Span]]) -> Any:
        """A JSON value with every string masked (keys stay as they are)."""
        if isinstance(value, str):
            return self.mask(value, spans_for(value))
        if isinstance(value, list):
            return [self.mask_value(v, spans_for) for v in value]
        if isinstance(value, dict):
            return {k: self.mask_value(v, spans_for) for k, v in value.items()}
        return value

    def unmask_value(self, value: Any) -> tuple[Any, bool]:
        if isinstance(value, str):
            return self.unmask(value)
        if isinstance(value, list):
            pairs = [self.unmask_value(v) for v in value]
            return [v for v, _ in pairs], any(r for _, r in pairs)
        if isinstance(value, dict):
            items = {k: self.unmask_value(v) for k, v in value.items()}
            return {k: v for k, (v, _) in items.items()}, any(r for _, r in items.values())
        return value, False
