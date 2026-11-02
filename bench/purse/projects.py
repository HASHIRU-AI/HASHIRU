"""The calibration pool, projects drawn from it (two jobs per family, with the observability knob rho), the text an
allocator reads, and scoring."""
import json
import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import purse.families  # noqa: F401  (registers the four families)
from purse.jobs import BINS, LEVELS, InfeasibleCell, Job, family, level_bin, make_job
from purse.market import Option
from purse.table import Stats

__all__ = ["ALL", "Project", "build_pool", "draw_hidden", "fold_of", "level_bin", "make_project", "parse_answers",
           "project_text", "score", "with_budget"]

log = logging.getLogger(__name__)

ALL = ("arithmetic", "grids", "aggregation", "strings")
VALUES = (1, 2, 4)
RETRIES = 3  # fresh seeds for a job its generator could not make, before the cell counts as infeasible
REDRAWS = 50  # new (level, bin) draws for a slot whose cell has no unused job left
FENCE = re.compile(r"```[\w-]*\n?(.*?)```", re.S)
TRAILING_COMMA = re.compile(r",\s*([}\]])")


def draw_hidden(level: int, rho: float, rng: random.Random) -> str:
    """With probability rho the hidden bin is the one the visible level suggests; otherwise any bin."""
    return level_bin(level) if rng.random() < rho else rng.choice(BINS)


def build_pool(seed: int, per_cell: int = 8, families: Sequence[str] = ALL) -> tuple[list[Job], list[str]]:
    """Every (family, level, bin) cell with `per_cell` jobs, and the cells skipped because a job could not be made."""
    jobs, skipped, index = [], [], 0
    for name in families:
        for level in LEVELS:
            for hidden in BINS:
                cell, made = f"{name}-l{level}-{hidden}", []
                for i in range(per_cell):
                    index += 1
                    for k in range(RETRIES):
                        try:
                            made.append(make_job(name, level, hidden, seed * 1_000_003 + index + 7919 * k,
                                                 f"{cell}-{i:03d}"))
                            break
                        except InfeasibleCell:
                            continue
                    else:
                        break
                if len(made) == per_cell:
                    jobs += made
                else:
                    log.warning("skipping %s: its generator could not make %d jobs", cell, per_cell)
                    skipped.append(cell)
    return jobs, skipped


def fold_of(job: Job) -> int:
    return int(job.uid.rsplit("-", 1)[1]) % 2


@dataclass(frozen=True)
class Project:
    id: str
    jobs: tuple[Job, ...]
    rho: float
    budget_usd: float = 0.0
    memory_mb: int = 12288
    deadline_s: float = 1800.0


def make_project(pool: Sequence[Job], fold: int, rho: float, seed: int, used: set[str]) -> Project:
    """Two jobs per family, each from a cell drawn by visible level and rho, unused so far and in `fold`. A slot whose
    cell is used up draws again (up to REDRAWS times)."""
    rng = random.Random(seed)
    free: dict[tuple[str, int, str], list[Job]] = {}
    for j in pool:
        if fold_of(j) == fold and j.uid not in used:
            free.setdefault((j.family, j.level, j.hidden), []).append(j)
    jobs: list[Job] = []
    for name in ALL:
        for _ in range(2):
            for _ in range(REDRAWS):
                level = rng.choice(LEVELS)
                cell = free.get((name, level, draw_hidden(level, rho, rng)))
                if cell:
                    picked = cell.pop(rng.randrange(len(cell)))
                    break
            else:
                raise ValueError(f"the pool has no unused {name} jobs left in fold {fold}")
            jobs.append(replace(picked, id=f"j{len(jobs) + 1}", value=rng.choice(VALUES)))
            used.add(picked.uid)
    return Project(f"p{seed}-f{fold}-r{rho:g}", tuple(jobs), rho)


def with_budget(p: Project, budget_usd: float) -> Project:
    return replace(p, budget_usd=budget_usd)


def _describe(o: Option) -> str:
    if o.kind == "local":
        return f"- {o.name}: a local model, $0 (it takes memory while hired)"
    if o.kind == "cloud":
        return (f"- {o.name}: a cloud model, ${o.input_usd_per_m:.2f} / ${o.output_usd_per_m:.2f} per million input / "
                "output tokens")
    only = f", {' and '.join(o.families)} jobs only" if o.families else ""
    return f"- {o.name}: an outside agent, ${o.per_message_usd:g} per message{only}"


def project_text(p: Project, options: Sequence[Option], stats: Stats | None) -> str:
    lines = [f"You have ${p.budget_usd:.4f} for this project (it also pays for your own model calls) and "
             f"{p.memory_mb} MB of memory for local models. Each job is worth its value in points; answer as many as "
             "you can.", ""]
    for j in p.jobs:
        lines += [f"### {j.id} ({j.family}, level {j.level}, value {j.value})", j.prompt, ""]
    lines += ["Who you can hire:", *(_describe(o) for o in options), ""]
    if stats:
        lines += ["Success rate and mean cost per attempt, measured on similar jobs:", "",
                  "| family | level | option | success | mean cost |", "|---|---|---|---|---|"]
        lines += [f"| {f} | {lvl} | {o} | {s.rate:.0%} | ${s.cost_usd:.4f} |"
                  for (f, lvl, _, o), s in sorted(stats.items())]
        lines.append("")
    lines.append('Answer with one JSON object mapping each job id to its answer, e.g. {"j1": "..."}.')
    return "\n".join(lines)


def _objects(text: str) -> list[str]:
    """Every balanced {...} span in `text`, outermost only, in order."""
    spans, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c == "{":
            start = i if depth == 0 else start
            depth += 1
        elif c == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start:i + 1])
    return spans


def parse_answers(reply: str) -> dict[str, str]:
    """The job answers in the last JSON object of `reply` that has any; ids are read case-insensitively. Never
    raises: what cannot be read is unanswered."""
    text = "\n".join(FENCE.findall(reply)) or reply
    for span in reversed(_objects(text)):
        for candidate in (span, TRAILING_COMMA.sub(r"\1", span)):
            try:
                found = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(found, dict):
                answers = {str(k).strip().lower(): v if isinstance(v, str) else json.dumps(v)
                           for k, v in found.items() if re.fullmatch(r"j\d+", str(k).strip().lower())}
                if answers:
                    return answers
    return {}


def score(p: Project, answers: Mapping[str, str]) -> float:
    """Value-weighted share of jobs answered correctly."""
    total = sum(j.value for j in p.jobs)
    won = sum(j.value for j in p.jobs if family(j.family).correct(j.answer, answers.get(j.id, "")))
    return won / total if total else 0.0
