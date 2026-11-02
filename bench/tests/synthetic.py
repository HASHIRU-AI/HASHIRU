"""A synthetic pool and outcome table built to pass PURSE's validity checks, with a switch that breaks exactly one."""
from purse.jobs import BINS, LEVELS
from purse.market import MARKET, SMALL
from purse.projects import ALL
from purse.table import OutcomeTable, Record
from tests.helpers import job

SHIFT = {"low": -1.5, "mid": 0.0, "high": 1.5}  # how much the hidden bin moves a job's real difficulty
REPEATS = 20


def rate(option: str, family: str, level: int, hidden: str, broken: str | None) -> float:
    e = level + (0.0 if broken == "V5" else SHIFT[hidden])
    if broken == "V1":
        return 0.5
    if option.startswith("a2a/"):  # unlike the real calc and csp, these fail hard jobs, so hidden bins matter here too
        return 0.0 if broken == "V6" else 1.0 if e <= 3.5 else 0.1
    if option == "meta/muse-spark-1.3":
        return 1.0 if broken == "V4" else 0.95 if e <= 4.5 else 0.6 if e <= 5.5 else 0.2
    if option == "gpt-6-luna" or (broken == "V3" and option == SMALL):
        return 0.9 if e <= 3.5 else 0.2
    return 0.95 if e <= 1.5 else 0.05  # the small tier


def cost(option: str, broken: str | None) -> float:
    if broken == "V1":
        return 0.0
    return {SMALL: 0.0003, "gpt-6-luna": 0.002,
            "meta/muse-spark-1.3": 0.00001 if broken == "V2" else 0.004}.get(option, 0.0)


def answer(family: str, n: int, broken: str | None) -> str:
    """A right answer in the family's own format; under V7 every job of a family has the same one."""
    n = 1 if broken == "V7" else n
    return {"grids": f'{{"1": {{"color": "c{n}"}}}}', "strings": f"s{n}"}.get(family, str(n))


def build(broken: str | None = None, per_cell: int = 8) -> tuple[list, OutcomeTable]:
    pool = [job(f"{f}-l{lvl}-{h}-{i:03d}", family=f, level=lvl, hidden=h, answer=answer(f, n, broken))
            for n, (f, lvl, h, i) in enumerate((f, lvl, h, i) for f in ALL for lvl in LEVELS for h in BINS
                                               for i in range(per_cell))]
    table = OutcomeTable(None)
    for j in pool:
        for o in MARKET:
            if o.families and j.family not in o.families:
                continue
            right = round(rate(o.name, j.family, j.level, j.hidden, broken) * REPEATS)
            for r in range(REPEATS):
                table.add(Record(j.uid, o.name, r, j.answer if r < right else "wrong", r < right, cost(o.name, broken),
                                 1.0))
    return pool, table
