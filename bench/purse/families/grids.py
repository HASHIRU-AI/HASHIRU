"""F2: logic grids. n houses in a row, m attributes, clues with exactly one solution. The level sets the size; the
hidden bin is how many case splits a propagating solver needs (0 low, 1 mid, 2 or more high). The text format is
read back by `parse`, which the csp agent and the grid_check tool use."""
import json
import random
import re
from dataclasses import dataclass

from purse.jobs import InfeasibleCell, extract_answer, register_family

__all__ = ["GRIDS", "SIZE", "Clue", "Grids", "Puzzle", "bin_of", "draw", "parse", "render", "satisfies", "solve"]

SIZE = {1: (3, 2), 2: (3, 3), 3: (4, 3), 4: (4, 4), 5: (5, 4)}
ATTRIBUTES = (("color", ("red", "green", "blue", "white", "yellow")),
              ("pet", ("cat", "dog", "bird", "fish", "horse")),
              ("drink", ("tea", "coffee", "milk", "juice", "water")),
              ("job", ("baker", "doctor", "pilot", "teacher", "writer")))
KINDS = ("same", "not_same", "at", "left_of", "next_to", "right_after")
MAX_ATTEMPTS = 1000  # small grids rarely need 2+ splits: level 2 high succeeds in about 1 of 200 draws
MAX_CLUES = 80

Var = tuple[str, str]  # (attribute, value)
Domains = dict[Var, frozenset[int]]


@dataclass(frozen=True)
class Clue:
    kind: str
    a: Var
    b: Var | None = None
    position: int | None = None


@dataclass(frozen=True)
class Puzzle:
    n: int
    attributes: tuple[tuple[str, tuple[str, ...]], ...]
    clues: tuple[Clue, ...]


# ---- text ----

def _house(v: Var) -> str:
    return f"the house with {v[0]} {v[1]}"


TEMPLATES = {"same": "{A} has {b0} {b1}.", "not_same": "{A} does not have {b0} {b1}.", "at": "{A} is house {k}.",
             "left_of": "{A} is somewhere to the left of {B}.", "next_to": "{A} is next to {B}.",
             "right_after": "{A} is directly to the right of {B}."}
V = r"the house with (\w+) (\w+)"
PATTERNS = [("same", re.compile(rf"^{V} has (\w+) (\w+)\.$")),
            ("not_same", re.compile(rf"^{V} does not have (\w+) (\w+)\.$")),
            ("at", re.compile(rf"^{V} is house (\d+)\.$")),
            ("left_of", re.compile(rf"^{V} is somewhere to the left of {V}\.$")),
            ("next_to", re.compile(rf"^{V} is next to {V}\.$")),
            ("right_after", re.compile(rf"^{V} is directly to the right of {V}\.$"))]


def render(p: Puzzle) -> str:
    lines = [f"There are {p.n} houses in a row, numbered 1 to {p.n} from left to right. Each house has a different "
             "value of each attribute:"]
    lines += [f"- {name}: {', '.join(values)}" for name, values in p.attributes]
    lines += ["", "Clues:"]
    for i, c in enumerate(p.clues, 1):
        b = c.b or ("", "")
        text = TEMPLATES[c.kind].format(A=_house(c.a), B=_house(b), b0=b[0], b1=b[1], k=c.position)
        lines.append(f"{i}. {text[0].upper()}{text[1:]}")
    example = ", ".join(f'"{name}": "..."' for name, _ in p.attributes)
    lines += ["", f'Give each house\'s attributes as JSON, e.g. {{"1": {{{example}}}, ...}}.']
    return "\n".join(lines)


def parse(text: str) -> Puzzle:
    """The last puzzle in `text` (a message may quote other things before it)."""
    starts = [m.start() for m in re.finditer(r"There are \d+ houses in a row", text)]
    if not starts:
        raise ValueError("no grid puzzle in the text")
    body = text[starts[-1]:]
    n = int(re.match(r"There are (\d+) houses", body).group(1))  # type: ignore[union-attr]
    attributes = tuple((m.group(1), tuple(v.strip() for v in m.group(2).split(",")))
                       for m in re.finditer(r"^- (\w+): (.+)$", body, re.M))
    clues = []
    for m in re.finditer(r"^\d+\. (.+)$", body, re.M):
        line = m.group(1)
        line = line[0].lower() + line[1:]
        for kind, pattern in PATTERNS:
            if g := pattern.match(line):
                a = (g.group(1), g.group(2))
                if kind == "at":
                    clues.append(Clue(kind, a, None, int(g.group(3))))
                else:
                    clues.append(Clue(kind, a, (g.group(3), g.group(4))))
                break
        else:
            raise ValueError(f"not a clue: {line}")
    return Puzzle(n, attributes, tuple(clues))


# ---- solving ----

def _holds(c: Clue, pa: int, pb: int) -> bool:
    return {"same": pa == pb, "not_same": pa != pb, "left_of": pa < pb, "next_to": abs(pa - pb) == 1,
            "right_after": pa == pb + 1}[c.kind]


def _narrow(c: Clue, d: Domains) -> Domains | None:
    if c.kind == "at":
        new = {c.a: d[c.a] & {c.position}}
    else:
        assert c.b is not None
        da, db = d[c.a], d[c.b]
        new = {c.a: frozenset(x for x in da if any(_holds(c, x, y) for y in db)),
               c.b: frozenset(y for y in db if any(_holds(c, x, y) for x in da))}
    if any(not v for v in new.values()):
        return None
    return d | new


def _propagate(p: Puzzle, d: Domains) -> Domains | None:
    changed = True
    while changed:
        before = d
        for c in p.clues:
            if (narrowed := _narrow(c, d)) is None:
                return None
            d = narrowed
        for name, values in p.attributes:
            for v in values:  # a placed value leaves its house to the others
                if len(d[(name, v)]) == 1:
                    for w in values:
                        if w != v:
                            d = d | {(name, w): d[(name, w)] - d[(name, v)]}
            for house in range(1, p.n + 1):  # a house only one value can take gets it
                can = [v for v in values if house in d[(name, v)]]
                if not can:
                    return None
                if len(can) == 1:
                    d = d | {(name, can[0]): frozenset({house})}
        if any(not v for v in d.values()):
            return None
        changed = d != before
    return d


def _search(p: Puzzle, d: Domains) -> tuple[list[Domains], int]:
    narrowed = _propagate(p, d)
    if narrowed is None:
        return [], 0
    if all(len(v) == 1 for v in narrowed.values()):
        return [narrowed], 0
    var = min((k for k, v in narrowed.items() if len(v) > 1), key=lambda k: (len(narrowed[k]), k))
    found: list[Domains] = []
    splits = 1
    for value in sorted(narrowed[var]):
        sols, s = _search(p, narrowed | {var: frozenset({value})})
        if sols and not found:
            splits += s  # count splits on the path to the first solution only
        found += sols
        if len(found) >= 2:
            break
    return found, splits


def _assignment(p: Puzzle, d: Domains) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {str(h): {} for h in range(1, p.n + 1)}
    for (name, value), houses in d.items():
        out[str(next(iter(houses)))][name] = value
    return out


def solve(p: Puzzle) -> tuple[list[dict[str, dict[str, str]]], int]:
    """Up to two solutions, and the number of case splits on the way to the first."""
    start = {(name, v): frozenset(range(1, p.n + 1)) for name, values in p.attributes for v in values}
    found, splits = _search(p, start)
    return [_assignment(p, d) for d in found], splits


def satisfies(p: Puzzle, assignment: dict[str, dict[str, str]]) -> bool:
    try:
        where = {(name, assignment[str(h)][name].strip().lower()): h
                 for h in range(1, p.n + 1) for name, _ in p.attributes}
    except (KeyError, AttributeError, TypeError):
        return False
    if any(sorted(v for (n, v), _ in where.items() if n == name) != sorted(values) for name, values in p.attributes):
        return False
    return all(where[c.a] == c.position if c.kind == "at" else _holds(c, where[c.a], where[c.b])  # type: ignore[index]
               for c in p.clues)


# ---- generating ----

def bin_of(splits: int) -> str:
    return "low" if splits == 0 else "mid" if splits == 1 else "high"


def _true_clue(p: Puzzle, where: dict[Var, int], rng: random.Random) -> Clue:
    variables = sorted(where)
    while True:
        kind = rng.choice(KINDS)
        a = rng.choice(variables)
        if kind == "at":
            return Clue(kind, a, None, where[a])
        b = rng.choice([v for v in variables if v[0] != a[0]] if kind in ("same", "not_same") else variables)
        if b != a and _holds(Clue(kind, a, b), where[a], where[b]):
            return Clue(kind, a, b)


def draw(level: int, rng: random.Random) -> Puzzle:
    """A puzzle with one solution and no clue it can do without."""
    n, m = SIZE[level]
    attributes = tuple((name, values[:n]) for name, values in ATTRIBUTES[:m])
    where = {(name, v): h + 1 for name, values in attributes for h, v in enumerate(rng.sample(values, n))}
    p = Puzzle(n, attributes, ())
    while len(solve(p)[0]) != 1:
        if len(p.clues) >= MAX_CLUES:
            p = Puzzle(n, attributes, ())
        p = Puzzle(n, attributes, (*p.clues, _true_clue(p, where, rng)))
    for clue in rng.sample(p.clues, len(p.clues)):
        fewer = Puzzle(n, attributes, tuple(c for c in p.clues if c != clue))
        if len(solve(fewer)[0]) == 1:
            p = fewer
    return p


class Grids:
    name = "grids"
    checkable = True

    def generate(self, level: int, hidden: str, rng: random.Random) -> tuple[str, str, dict[str, int]]:
        for _ in range(MAX_ATTEMPTS):
            p = draw(level, rng)
            solutions, splits = solve(p)
            if bin_of(splits) == hidden:
                features = {"positions": p.n, "attributes": len(p.attributes), "clues": len(p.clues)}
                return render(p), json.dumps(solutions[0], sort_keys=True), features
        raise InfeasibleCell(f"grids level {level} {hidden}: no puzzle in {MAX_ATTEMPTS} tries")

    def correct(self, answer: str, response: str) -> bool:
        text = extract_answer(response)
        start, end = text.find("{"), text.rfind("}")
        try:
            given = json.loads(text[start:end + 1]) if start >= 0 else None
        except json.JSONDecodeError:
            return False
        if not isinstance(given, dict):
            return False

        def norm(x: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
            return {str(k).strip(): {str(a).strip().lower(): str(v).strip().lower() for a, v in row.items()}
                    for k, row in x.items() if isinstance(row, dict)}
        return norm(given) == norm(json.loads(answer))


GRIDS = register_family(Grids())
