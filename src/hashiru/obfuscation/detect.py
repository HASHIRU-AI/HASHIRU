"""Finding sensitive values in private text: Presidio's pattern recognizers (contact details, card, account and ID
numbers, IPs, URLs, dates) plus recognizers for secrets, and optionally a local model for what patterns cannot see
(names, places, employers, health and money details), which also proposes generalizations for values whose meaning
matters. If the local model cannot answer, detect() returns None and the caller refuses instead (fail closed).
"""
import asyncio
import functools
import json
import logging
from collections import OrderedDict
from typing import Any

from hashiru.obfuscation.vault import Span

__all__ = ["DETECT_PROMPT", "Detector"]

log = logging.getLogger(__name__)

DETECT_PROMPT = (
    "You mark private details in text for a privacy filter, so they can be hidden before the text goes to a cloud "
    "model. Reply with only a JSON list of objects {\"text\": the exact substring, \"type\": one of PERSON, PLACE, "
    "ORG, ID, CONTACT, MONEY, DATE, AGE, HEALTH, OTHER, \"general\": a vague description, only for MONEY, DATE, AGE "
    "and HEALTH (e.g. \"a six-figure amount\", \"in the late 1980s\", \"in their forties\", \"a chronic "
    "condition\")}. Mark names of private people, home and work places, employers, account and ID numbers, contact "
    "details, amounts, personal dates, ages and health details. Do not mark public figures, well-known places or "
    "ordinary words. Reply [] if there is nothing to mark.")
GENERAL = {"MONEY", "DATE", "AGE", "HEALTH"}  # attributes a placeholder would empty of meaning
SECRETS = (  # (name, regex) for credentials Presidio has no recognizer for
    ("openai_style_key", r"\bsk-[A-Za-z0-9_-]{16,}"),
    ("github_token", r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("hugging_face_token", r"\bhf_[A-Za-z0-9]{30,}"),
    ("slack_token", r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    ("private_key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"),
)
SCORE = 0.3   # Presidio's lowest score kept: masking a false positive costs little
CACHE = 512   # texts whose spans are remembered (history is resent every turn)


@functools.cache
def analyzer() -> Any:
    """Presidio's analyzer on a blank spaCy pipeline: its pattern recognizers, no downloaded NER model. Built once."""
    import spacy
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_analyzer.nlp_engine import SpacyNlpEngine

    class BlankEngine(SpacyNlpEngine):  # type: ignore[misc]
        def load(self) -> None:
            self.nlp = {"en": spacy.blank("en")}  # type: ignore[assignment]  # Presidio types it None until load()

    engine = BlankEngine(models=[{"lang_code": "en", "model_name": "blank"}])
    engine.load()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=engine)
    registry.add_recognizer(PatternRecognizer(supported_entity="SECRET", name="SecretRecognizer",
                                              patterns=[Pattern(name, regex, 0.9) for name, regex in SECRETS]))
    return AnalyzerEngine(nlp_engine=engine, registry=registry, supported_languages=["en"])


class Detector:
    """`llm` and `model` (a local model) add the model pass; without them only patterns run."""

    def __init__(self, llm: Any, model: str) -> None:
        self.llm, self.model = llm, model
        self._cache: OrderedDict[str, list[Span]] = OrderedDict()

    async def detect(self, text: str) -> list[Span] | None:
        if (hit := self._cache.get(text)) is not None:
            self._cache.move_to_end(text)
            return hit
        spans = await asyncio.to_thread(patterns, text)
        if self.llm is not None and self.model:
            found = await self._model_pass(text)
            if found is None:
                return None
            spans += found
        self._cache[text] = spans
        while len(self._cache) > CACHE:
            self._cache.popitem(last=False)
        return spans

    async def _model_pass(self, text: str) -> list[Span] | None:
        messages = [{"role": "system", "content": DETECT_PROMPT}, {"role": "user", "content": text}]
        try:
            msg, _ = await self.llm.chat(self.model, messages, None)
        except Exception as exc:  # any failure (Ollama down, a timeout, a provider error) must fail closed
            log.warning("the local detector failed: %s", exc)
            return None
        return parse(msg.content or "", text)


def patterns(text: str) -> list[Span]:
    return [Span(r.start, r.end, r.entity_type) for r in analyzer().analyze(text=text, language="en",
                                                                            score_threshold=SCORE)]


def parse(reply: str, text: str) -> list[Span] | None:
    """Spans from the model's JSON list, for substrings that really occur in `text`; None if it is not a list."""
    start, end = reply.find("["), reply.rfind("]")
    try:
        items = json.loads(reply[start:end + 1]) if 0 <= start < end else None
    except json.JSONDecodeError:
        items = None
    if not isinstance(items, list):
        return None
    spans = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip():
            continue
        value, kind = item["text"], str(item.get("type", "OTHER")).upper()
        general = str(item.get("general") or "") if kind in GENERAL else ""
        at = text.find(value)
        while at >= 0:  # every occurrence: the model names a value once
            spans.append(Span(at, at + len(value), kind, general))
            at = text.find(value, at + len(value))
    return spans
