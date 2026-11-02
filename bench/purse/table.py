"""The outcome table: every recorded attempt of a job on an option, kept as JSON lines so an interrupted calibration
resumes where it stopped. Cell statistics are what allocators are told; per-job rates are the oracles' truth."""
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from purse.jobs import BINS, Job, level_bin
from purse.market import Option

__all__ = ["CellStat", "OutcomeTable", "Record", "Stats", "cell_stats", "visible_stats"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Record:
    uid: str
    option: str
    repeat: int
    answer: str
    correct: bool
    cost_usd: float
    seconds: float
    refused: str = ""

    @property
    def solved(self) -> bool:
        """Right, and actually run: a refused attempt is a failure whatever its answer field says."""
        return self.correct and not self.refused


@dataclass(frozen=True)
class CellStat:
    rate: float
    cost_usd: float
    n: int


Stats = dict[tuple[str, int, str | None, str], CellStat]  # (family, level, hidden bin or None, option)


class OutcomeTable:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._by: dict[tuple[str, str], list[Record]] = defaultdict(list)
        self._done: set[tuple[str, str, int]] = set()
        if path is not None and path.exists():
            for n, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    self._keep(Record(**json.loads(line)))
                except (json.JSONDecodeError, TypeError) as exc:
                    log.warning("%s line %d skipped (%s): a write cut short?", path, n, exc)

    def _keep(self, r: Record) -> None:
        self._by[(r.uid, r.option)].append(r)
        self._done.add((r.uid, r.option, r.repeat))

    def add(self, r: Record) -> None:
        self._keep(r)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+") as f:
                f.seek(0, 2)
                if f.tell():  # start on a new line even after a line cut short
                    f.seek(f.tell() - 1)
                    if f.read(1) != "\n":
                        f.write("\n")
                f.write(json.dumps(asdict(r)) + "\n")
                f.flush()

    def has(self, uid: str, option: str, repeat: int) -> bool:
        return (uid, option, repeat) in self._done

    def options(self) -> set[str]:
        return {option for (_, option) in self._by}

    def records(self, uid: str, option: str) -> list[Record]:
        return sorted(self._by.get((uid, option), []), key=lambda r: r.repeat)

    def rate(self, uid: str, option: str) -> float | None:
        recs = self._by.get((uid, option))
        return sum(r.solved for r in recs) / len(recs) if recs else None

    def cost(self, uid: str, option: str) -> float | None:
        recs = self._by.get((uid, option))
        return sum(r.cost_usd for r in recs) / len(recs) if recs else None


def cell_stats(table: OutcomeTable, jobs: Iterable[Job], hidden: bool, market: Sequence[Option] = ()) -> Stats:
    """Success rate and mean cost per (family, level, hidden bin if `hidden`, option), over every recorded attempt.
    An outside agent in `market` costs its current price; only its correctness comes from the table."""
    prices = {o.name: o.per_message_usd for o in market if o.kind == "agent"}
    acc: dict[tuple[str, int, str | None, str], list[tuple[bool, float]]] = defaultdict(list)
    options = table.options()
    for job in jobs:
        for option in options:
            for r in table.records(job.uid, option):
                key = (job.family, job.level, job.hidden if hidden else None, option)
                acc[key].append((r.solved, prices.get(option, r.cost_usd)))
    return {k: CellStat(sum(c for c, _ in v) / len(v), sum(x for _, x in v) / len(v), len(v)) for k, v in acc.items()}


def visible_stats(hidden: Stats, rho: float) -> Stats:
    """What an allocator measures per (family, level, option) on projects drawn with `rho`: the hidden bins' cells
    weighted by P(bin | level) = rho [bin is the level's bin] + (1 - rho) / 3, renormalized over the bins present."""
    groups: dict[tuple[str, int, str], list[tuple[float, CellStat]]] = defaultdict(list)
    for (fam, level, b, option), s in hidden.items():
        if b is not None:
            groups[(fam, level, option)].append((rho * (b == level_bin(level)) + (1 - rho) / len(BINS), s))
    out: Stats = {}
    for (fam, level, option), cells in groups.items():
        total = sum(w for w, _ in cells)
        if total > 0:
            out[(fam, level, None, option)] = CellStat(sum(w * s.rate for w, s in cells) / total,
                                                        sum(w * s.cost_usd for w, s in cells) / total,
                                                        sum(s.n for _, s in cells))
    return out
