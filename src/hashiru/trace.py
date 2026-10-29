"""Trace writer (one JSONL file per session) and the four label rules.

Labels are recorded but not enforced yet; enforcement comes with the policy layer.
"""
import json
import time
from pathlib import Path
from typing import Any

__all__ = ["CLEAN_REPLIES", "PRIVATE", "SOURCES", "UNTRUSTED", "Trace"]

PRIVATE, UNTRUSTED = "private", "untrusted"
SOURCES = {"read_file": [PRIVATE], "memory": [PRIVATE], "get_website": [UNTRUSTED]}  # rule 1
CLEAN_REPLIES = {"done", "failed"}                                                      # rule 4


class Trace:
    """The ordered events of one session. The first event is 'start' with the session's settings."""

    def __init__(self, path: Path | None, **start: Any) -> None:
        self.events: list[dict[str, Any]] = []
        self._labels: dict[tuple[str, str], set[str]] = {}
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")
        self.emit("start", **start)

    def emit(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Append one event. seq is the total order of everything that happened."""
        event = {"seq": len(self.events), "ts": round(time.time(), 3), "kind": kind, **fields}
        self.events.append(event)
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        return event

    def labels(self, agent: str, task: str) -> set[str]:
        """Rule 2: everything the agent received this task. Sticky until the task ends."""
        return self._labels.setdefault((agent, task), set())

    def receive(self, agent: str, task: str, new: list[str]) -> None:
        """Add labels to an agent and record a 'label' event whenever its set grows."""
        have = self.labels(agent, task)
        if set(new) - have:
            have.update(new)
            self.emit("label", task=task, agent=agent, labels=sorted(have))

    def outgoing(self, agent: str, task: str, text: str) -> list[str]:
        """Rule 3: what an agent sends carries its labels. Rule 4: an exact 'done'/'failed' carries none."""
        return [] if text.strip() in CLEAN_REPLIES else sorted(self.labels(agent, task))
