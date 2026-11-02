# PURSE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build PURSE, the benchmark that measures how agents split a money pool across hired models and agents,
with oracles, solver baselines, validity checks and a HASHIRU adapter, all tested with fake models at $0.

**Architecture:** A separate package `bench/purse` that imports `hashiru` (never the reverse). Jobs come from four
seeded generators with exact verifiers. Projects draw calibrated jobs. Allocators run against a `Market`: a
`ReplayMarket` over the recorded outcome table, or a `HubMarket` that hires through a live HASHIRU hub as the CEO.
Two small HASHIRU changes: a `web_tools` switch and a `memory_mb` cap.

**Tech Stack:** Python 3.13, uv, pytest, the stdlib (`random`, `json`, `fractions`, `statistics`); `hashiru`, and
through it a2a-sdk, mcp, omegaconf and httpx. No new dependencies.

**Spec:** `plan/2026-11-01-allocation-benchmark.md` (read it first; section numbers below refer to it).

## Global Constraints

- No new dependencies. All randomness from `random.Random(seed)`; never `hash()` (salted per process).
- `src/hashiru` never imports `purse`. Files stay under 400 lines; frozen dataclasses for data; type hints; `__all__`
  in each `__init__.py`; logging, not `print`, except CLI output.
- Tests make no real model call and no network call outside localhost. Every CLI command that can reach a real model
  refuses unless `--spend` is given, and Claude still asks the user before running one (memory: no real model runs).
- Bench test files are named `bench/tests/test_purse_*.py` (pytest needs unique basenames across `tests/` and
  `bench/tests/`, neither of which is a package).
- Values fixed by the spec: K = 8 jobs per project, two per family; values drawn from {1, 2, 4}; ρ ∈ {1, 0.5, 0};
  τ ∈ {0.25, 0.5, 1}; M ∈ {12288, 24576} MB, default 12288; knapsack margin 0.9; `grid_check` at most 3 calls per
  puzzle; calibration pool 4 families × 5 levels × 3 bins × 8 jobs = 480; repeats 2 for local models and
  `gpt-6-luna`, 1 for `meta/muse-spark-1.3`; $0.25 reserved per calibration hire; 2 folds.
- Market names: `ollama_chat/qwen3.5:4b`, `ollama_chat/gemma4:12b` (local); `gpt-6-luna`, `meta/muse-spark-1.3`
  (cloud); `a2a/calc` (arithmetic only), `a2a/csp` (grids only). The CEO is `gpt-6-luna`.
- Every worker gets the system prompt `WORKER_PROMPT = "Solve the task exactly. Put only your final answer inside
  <answer></answer>."` and the job's canonical prompt as its message.
- Docs and comments: no em dashes.
- Outputs go under `bench/out/` (gitignored).

## Review Focus

1. A worker's answer with formatting noise (no `<answer>` tags, commas in numbers, code fences, quotes, a trailing
   period): the verifier normalizes it and does not mark a right answer wrong. Tests in Tasks 2 to 5.
2. The CEO's final reply has malformed JSON, JSON in a code fence, prose around it, or ids like `J3`: answers are
   still read, a missing id counts as unanswered, and nothing crashes. Test in Task 7.
3. A recorded cost bigger than what is left in the pool: charged in full (as the hub does), after which cloud and
   agent attempts are refused. Left never goes negative before an attempt is allowed. Test in Task 8.
4. A (level, bin) cell the generator cannot produce: it raises `InfeasibleCell`, and the pool builder reports and
   skips the cell instead of mislabeling a job. Tests in Tasks 3 and 7.
5. Calibration stops midway (Ctrl-C, a provider error, a local model refused for memory): the JSONL table keeps every
   finished record, a rerun skips them, refusals are recorded with their reason, and `max_usd` stops spending. Test
   in Task 15.

---

### Task 1: Package skeleton, jobs and the family registry

**Files:**
- Modify: `pyproject.toml` (pytest `testpaths = ["tests", "bench/tests"]`, `pythonpath = ["bench"]`; ruff `include`
  adds `"bench/**/*.py"`; mypy `files` adds `"bench/purse"`), `.gitignore` (adds `bench/out/`)
- Create: `bench/purse/__init__.py`, `bench/purse/jobs.py`, `bench/purse/families/__init__.py`,
  `bench/tests/conftest.py`
- Test: `bench/tests/test_purse_jobs.py`

**Interfaces:**
- Produces, in `purse.jobs`:
  - `BINS = ("low", "mid", "high")`, `LEVELS = (1, 2, 3, 4, 5)`, `WORKER_PROMPT` (above),
    `ANSWER_NOTE = "Put only your final answer inside <answer></answer>."`
  - `@dataclass(frozen=True) class Job: id: str; uid: str; family: str; level: int; hidden: str; value: int;
    prompt: str; answer: str; features: tuple[tuple[str, int], ...]`. `id` is the project slot (`j1`..`j8`) and is
    `""` in the pool; `uid` is `"{family}-l{level}-{hidden}-{i:03d}"`.
  - `class Family(Protocol): name: str; checkable: bool; def generate(self, level: int, hidden: str,
    rng: random.Random) -> tuple[str, str, dict[str, int]]` (task text, answer, visible features);
    `def correct(self, answer: str, response: str) -> bool`.
  - `class InfeasibleCell(ValueError)`; `FAMILIES: dict[str, Family]`; `register_family(f: Family) -> Family`;
    `family(name: str) -> Family`.
  - `extract_answer(response: str) -> str`: the text of the last `<answer>...</answer>`, else the whole response,
    stripped of whitespace, surrounding backticks, quotes and a code fence.
  - `make_job(family_name: str, level: int, hidden: str, seed: int, uid: str, value: int = 1) -> Job`: prompt is the
    task text + `"\n\n" + ANSWER_NOTE`.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_answer_takes_the_last_tagged_answer():
    assert extract_answer("x <answer>1</answer> y <answer> 42 </answer>") == "42"
    assert extract_answer("```\n42\n```") == "42" and extract_answer("`abc`") == "abc"
    assert extract_answer("no tags") == "no tags"

def test_registry_rejects_unknown_families_by_name():
    with pytest.raises(KeyError, match="nope"):
        family("nope")

def test_a_job_is_immutable_and_its_prompt_asks_for_tags(fake_family):  # conftest registers a 1-line family
    job = make_job("fake", 1, "low", seed=1, uid="fake-l1-low-000")
    assert job.prompt.endswith(ANSWER_NOTE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        job.value = 4  # type: ignore[misc]
```

- [ ] **Step 2: Run** `uv run pytest bench/tests/test_purse_jobs.py -q`. Expected: FAIL (`ModuleNotFoundError: purse`).
- [ ] **Step 3: Implement** `purse/jobs.py` and the skeleton; `families/__init__.py` imports each family module (added
  by Tasks 2 to 5) so importing `purse.families` registers all four.
- [ ] **Step 4: Run** the test file, then `uv run pytest -q`, `uv run ruff check .`, `uv run mypy`. Expected: all pass;
  the existing 441 tests still pass.
- [ ] **Step 5: Commit** `feat(bench): PURSE skeleton, jobs and the family registry`, together with the three plan
  files (`plan/2026-11-01-*.md`, `plan/whitepaper/`).

### Task 2: F1 arithmetic

**Files:** Create `bench/purse/families/arithmetic.py`; Test `bench/tests/test_purse_arithmetic.py`

**Interfaces:** Produces `ARITHMETIC: Family` registered as `"arithmetic"`, `checkable = False`.

Parameters (spec §5): operators `n = {1: 2, 2: 4, 3: 8, 4: 14, 5: 22}`; operand digits
`D = {1: 2, 2: 3, 3: 4, 4: 6, 5: 8}`. Hidden bins: share of `*` among operators `{low: 0.2, mid: 0.35, high: 0.5}`,
the rest split `+` 70% and `-` 30%; operand digits drawn from `0-4` (low), `0-9` (mid), `5-9` (high), leading digit
nonzero. A random binary tree, fully parenthesized; redraw (up to 100 times) until every intermediate value has
`abs(x) < 10 ** (3 * D)`. Answer `str(int)`. Task text: `"Compute the exact value of this expression:\n\n{expr}"`.
Features: `{"operators": n, "digits": D}`.

- [ ] **Step 1: Failing tests**: `test_same_seed_same_job_and_answer_is_exact` (two `generate` calls with
  `Random(7)` are equal; `ast`-evaluating the expression equals `int(answer)`); `test_levels_set_size`
  (operator count of the expression == `n[level]`); `test_hidden_bins_change_carries` (over 30 seeds at level 3, mean
  count of digits ≥ 5 is higher for `high` than `low`); `test_correct_ignores_formatting`
  (`correct("1234", r)` for `r` in `"<answer>1,234</answer>"`, `"1234."`, `"`1234`"`; not for `"1235"`, `"12340"`).
- [ ] **Step 2: Run** them. Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(bench): arithmetic jobs`.

### Task 3: F2 logic grids, with the shared parser

**Files:** Create `bench/purse/families/grids.py`; Test `bench/tests/test_purse_grids.py`

**Interfaces:** Produces `GRIDS: Family` (`"grids"`, `checkable = True`) and, for Task 12:
`@dataclass(frozen=True) class Puzzle: n: int; attributes: tuple[tuple[str, tuple[str, ...]], ...];
clues: tuple[Clue, ...]`; `@dataclass(frozen=True) class Clue: kind: str; a: tuple[str, str];
b: tuple[str, str] | None; position: int | None`; `render(p: Puzzle) -> str`; `parse(text: str) -> Puzzle`;
`solve(p: Puzzle) -> tuple[list[dict[str, str]], int]` (all solutions up to 2, and the split count of the first);
`satisfies(p: Puzzle, assignment: dict[str, dict[str, str]]) -> bool`.

Parameters: `(n, m) = {1: (3, 2), 2: (3, 3), 3: (4, 3), 4: (4, 4), 5: (5, 4)}`; attributes in order `color`, `pet`,
`drink`, `job`, each with 5 values (first `n` used). Clue kinds: `same`, `not_same`, `at`, `left_of`, `next_to`,
`right_after`, each rendered from one fixed English template that `parse` reads back. Generation: draw a solution;
add true clues until `solve` finds exactly 1 solution; remove clues in random order while still unique; measure
splits; bins `low` = 0 splits, `mid` = 1, `high` ≥ 2; up to 300 attempts, else `InfeasibleCell`. Answer: JSON
`{"1": {"color": "red", ...}, ...}` with sorted keys. Features `{"positions": n, "attributes": m, "clues": len}`.

The split count, which the tests do not determine:

```python
def search(domains):            # domains: (position, attribute) -> set of values
    domains = propagate(domains)  # apply every clue and all-different until nothing changes; None if a domain empties
    if domains is None: return [], 0
    if all(len(d) == 1 for d in domains.values()): return [domains], 0
    var = min((k for k, d in domains.items() if len(d) > 1), key=lambda k: (len(domains[k]), k))
    found, splits = [], 1         # this node branches
    for value in sorted(domains[var]):
        sols, s = search(domains | {var: {value}})
        if sols and not found: splits += s   # count splits on the path to the first solution only
        found += sols
        if len(found) >= 2: break
    return found, splits
```

- [ ] **Step 1: Failing tests**: `test_render_parse_round_trip` (for 20 seeds per level, `parse(render(p)) == p`);
  `test_generated_puzzles_have_one_solution_and_the_bin_they_claim`; `test_satisfies_accepts_only_the_solution`
  (flip two values: False); `test_infeasible_cell_raises` (a `generate` with `max_attempts=1` patched to always miss
  raises `InfeasibleCell`); `test_correct_reads_json_in_tags_or_fences_case_insensitively`.
- [ ] **Steps 2 to 4:** run (FAIL), implement, run (PASS).
- [ ] **Step 5: Commit** `feat(bench): logic grid jobs and their solver`.

### Task 4: F3 aggregation

**Files:** Create `bench/purse/families/aggregation.py`; Test `bench/tests/test_purse_aggregation.py`

**Interfaces:** `AGGREGATION: Family` (`"aggregation"`, `checkable = False`).

Parameters: records `N = {1: 40, 2: 120, 3: 400, 4: 1000, 5: 2500}`. A record line:
`"#{id} | {YYYY-MM-DD} | customer {First Last} | region {region} | items {k} | total ${x.xx} | status {status}"`.
Question by bin: `low` one field (`region == R`); `mid` two fields (`region == R and status == S`); `high` two
fields and `total above $T`, with near-miss distractors (regions `North` against `North-East` and `Northwest`; totals
exactly `$T`, which do not count; status `shipped` against `shipped-late`). The number of matches is between 3 and
`max(3, N // 5)`. Answer: the count. Task text: `"Here are {N} orders, one per line.\n\n{lines}\n\nHow many orders
{condition}? Answer with the count."` Features `{"records": N}`.

- [ ] **Step 1: Failing tests**: `test_answer_counts_matches` (recount from the lines with an independent parser in
  the test); `test_high_bin_contains_near_misses` (each kind of distractor appears at least once at level 3 high);
  `test_level_5_is_long` (`len(prompt) > 150_000` characters); `test_correct_ignores_formatting`.
- [ ] **Steps 2 to 5** as in Task 2. Commit `feat(bench): aggregation jobs`.

### Task 5: F4 string pipelines

**Files:** Create `bench/purse/families/strings.py`; Test `bench/tests/test_purse_strings.py`

**Interfaces:** `STRINGS: Family` (`"strings"`, `checkable = False`).

Parameters: operations `k = {1: 2, 2: 3, 3: 5, 4: 7, 5: 10}`; input length `{1: 8, 2: 12, 3: 16, 4: 24, 5: 32}`.
Operations: `reverse`; `rotate_left r`; `caesar s` (letters only, keeps case); `swapcase`; `drop_every j` (delete
every j-th character, counting from 1); `double_vowels`. Bins: `low` letters only and operations from `reverse`,
`caesar` (s ≤ 13), `swapcase`; `mid` letters and digits, any operation; `high` letters, digits, punctuation and
spaces, at least half the operations `rotate_left` or `drop_every`, and every `caesar` with s > 13. Answer: the exact
final string. Task text: ``"Start with the string `{s}` and apply these operations in order:\n{numbered ops}\nWhat
is the final string?"`` Features `{"operations": k, "length": L}`.

- [ ] **Step 1: Failing tests**: `test_each_operation` (one hand-checked case per operation);
  `test_answer_applies_the_listed_operations_in_order`; `test_correct_strips_tags_and_backticks_but_not_inner_spaces`
  (`"a b"` is not `"ab"`).
- [ ] **Steps 2 to 5.** Commit `feat(bench): string pipeline jobs`.

### Task 6: Market, outcome table and replay

**Files:** Create `bench/purse/market.py`, `bench/purse/table.py`, `bench/purse/replay.py`; Test
`bench/tests/test_purse_table.py`

**Interfaces:**
- `market.py`: `@dataclass(frozen=True) class Option: name: str; kind: Literal["local", "cloud", "agent"];
  per_message_usd: float = 0.0; families: tuple[str, ...] = ()` (empty: serves every family); `MARKET` (the six
  options from Global Constraints); `CEO_MODEL = "gpt-6-luna"`; `serves(o: Option, job: Job) -> bool`;
  `priced(market: Sequence[Option], agent_price_usd: float) -> tuple[Option, ...]`;
  `@dataclass(frozen=True) class Outcome: answer: str; correct: bool; cost_usd: float; seconds: float;
  refused: str = ""`; `class Market(Protocol): async def attempt(self, job: Job, option: str) -> Outcome`.
- `table.py`: `@dataclass(frozen=True) class Record: uid: str; option: str; repeat: int; answer: str; correct: bool;
  cost_usd: float; seconds: float; refused: str = ""`; `class OutcomeTable: __init__(self, path: Path | None);
  add(r: Record) -> None` (appends a JSON line and flushes); `has(uid, option, repeat) -> bool`;
  `records(uid, option) -> list[Record]`; `rate(uid, option) -> float | None`; `cost(uid, option) -> float | None`
  (means over non-refused records; refused counts as a failure at its recorded cost);
  `@dataclass(frozen=True) class CellStat: rate: float; cost_usd: float; n: int`;
  `Stats = dict[tuple[str, int, str | None, str], CellStat]` keyed (family, level, hidden or None, option);
  `cell_stats(table: OutcomeTable, jobs: Iterable[Job], hidden: bool) -> Stats`.
- `replay.py`: `class ReplayMarket: __init__(self, table: OutcomeTable, draw: int = 0)`; `attempt` returns the
  record `records[draw % len(records)]` as an `Outcome`, or `Outcome("", False, 0.0, 0.0, refused="no record")`.

- [ ] **Step 1: Failing tests**: `test_table_survives_a_reload_and_skips_done_records` (write 3, reopen, `has` is
  true); `test_cell_stats_with_and_without_hidden_bins`; `test_replay_draws_are_deterministic_and_rotate`.
- [ ] **Steps 2 to 5.** Commit `feat(bench): market, outcome table and replay`.

### Task 7: The pool and projects

**Files:** Create `bench/purse/projects.py`; Test `bench/tests/test_purse_projects.py`

**Interfaces:**
- Consumes `make_job`, `FAMILIES`, `InfeasibleCell` (Task 1), `Option` and `Stats` (Task 6) for `project_text`.
- Produces:
  - `level_bin(level: int) -> str` (1 and 2 low, 3 mid, 4 and 5 high);
    `draw_hidden(level: int, rho: float, rng: random.Random) -> str`;
  - `build_pool(seed: int, per_cell: int = 8, families: Sequence[str] = ALL) -> tuple[list[Job], list[str]]`
    (jobs, and the infeasible cells skipped, as uids' cell prefixes); job seed = `seed * 1_000_003 + index`;
  - `fold_of(job: Job) -> int` (the uid's index modulo 2);
  - `@dataclass(frozen=True) class Project: id: str; jobs: tuple[Job, ...]; rho: float; budget_usd: float = 0.0;
    memory_mb: int = 12288; deadline_s: float = 1800.0`;
  - `make_project(pool: Sequence[Job], fold: int, rho: float, seed: int, used: set[str]) -> Project` (two jobs per
    family; each slot draws a visible level uniformly, then `draw_hidden`, then an unused job of that cell in that
    fold; values from {1, 2, 4}; ids `j1`..`j8` in order; adds the uids to `used`);
  - `with_budget(p: Project, budget_usd: float) -> Project`;
  - `project_text(p: Project, options: Sequence[Option], stats: Stats | None) -> str` (each job's id, family, level,
    value and task text; the pool; each option with its prices; the stats table when given; and the closing line
    `"Answer with one JSON object mapping each job id to its answer, e.g. {\"j1\": \"...\"}."`);
  - `parse_answers(reply: str) -> dict[str, str]`; `score(p: Project, answers: Mapping[str, str]) -> float`.

- [ ] **Step 1: Failing tests**: `test_draw_hidden_agrees_with_the_visible_level_at_rate_rho` (10,000 draws:
  agreement ≈ ρ + (1 − ρ)/3 within 0.02, for ρ in 0, 0.5, 1); `test_projects_never_reuse_a_job_and_stay_in_their_fold`;
  `test_pool_skips_infeasible_cells_and_reports_them` (a family raising `InfeasibleCell` for one cell);
  `test_parse_answers_reads_fenced_and_messy_json` (fenced, with prose around it, id `J3` read as `j3`, trailing
  comma tolerated or the id counted as missing, never an exception); `test_score_weights_by_value`.
- [ ] **Steps 2 to 5.** Commit `feat(bench): pool, projects, project text and scoring`.

### Task 8: Solver allocators and the executor

**Files:** Create `bench/purse/solvers.py`; Test `bench/tests/test_purse_solvers.py`

**Interfaces:**
- Produces: `Allocation = dict[str, str]` (job id to option name, or `SKIP = "skip"`);
  `class Pool: __init__(self, budget_usd: float); left: float; allows(o: Option) -> bool` (local always; cloud while
  `left > 0`; agent while `left >= o.per_message_usd`); `charge(usd: float) -> None`;
  `@dataclass(frozen=True) class Attempt: job_id: str; option: str; outcome: Outcome`;
  `@dataclass(frozen=True) class RunResult: answers: dict[str, str]; spent_usd: float;
  attempts: tuple[Attempt, ...]; allocator_usd: float = 0.0`;
  `knapsack(items: Sequence[tuple[str, Sequence[tuple[str, float, float]]]], budget_usd: float) -> Allocation`
  (each item is a job id with its (option, gain, cost) choices; skipping is always allowed);
  `plan_knapsack(p: Project, stats: Stats, market: Sequence[Option], hidden: bool = False, margin: float = 0.9) ->
  Allocation`; `fixed(p, option: str) -> Allocation`; `hybrid(p) -> Allocation` (levels 1 and 2 qwen, 3 and 4 Luna,
  5 Spark); `uniform(p, stats, market) -> Allocation` (each job the best option whose mean cost fits `B / K`);
  `random_allocation(p, market, seed) -> Allocation`;
  `async run_upfront(p: Project, alloc: Allocation, market: Market, options: Sequence[Option]) -> RunResult` (jobs in
  descending value, then id; a job runs when its option is not SKIP and `Pool.allows` it; its outcome's cost is
  charged in full);
  `async cascade(p, stats, market: Market, options, theta: float) -> RunResult`;
  `async fit_theta(projects, stats, market, options, grid: Sequence[float] = (0.3, 0.5, 0.7, 0.9)) -> float`.

The knapsack, in units of $0.0001 with costs rounded up:

```python
best = {0: (0.0, {})}                        # spent units -> (gain, allocation)
for job_id, choices in items:
    nxt = {}
    for spent, (gain, alloc) in best.items():
        for option, g, c in [(SKIP, 0.0, 0.0), *choices]:
            s = spent + math.ceil(c / UNIT)
            if s * UNIT <= budget_usd + 1e-12:
                cand = (gain + g, alloc | {job_id: option})
                if s not in nxt or cand[0] > nxt[s][0] + 1e-12: nxt[s] = cand
    best = nxt
return max(best.items(), key=lambda kv: (kv[1][0], -kv[0]))[1][1]   # most gain, then least spent
```

The cascade: for each job in descending value per cost, options that serve it with cell rate ≥ θ, cheapest first.
A checkable job (grids) tries them in turn until `correct` passes on the answer (at most 3 tries, the `grid_check`
limit). Any other job asks the two cheapest; if their extracted answers agree it keeps that answer, else it asks the
next option and keeps its answer. Every attempt needs `Pool.allows`.

- [ ] **Step 1: Failing tests**: `test_knapsack_matches_brute_force` (100 random instances, 4 jobs, 3 options);
  `test_executor_charges_an_overrunning_call_in_full_then_refuses_cloud` (Review Focus 3: pool $0.01, a recorded
  $0.03 call, then the next cloud job is refused and local ones still run); `test_cascade_escalates_on_disagreement`;
  `test_cascade_stops_at_three_checks`; `test_fit_theta_picks_the_best_on_the_other_fold`.
- [ ] **Steps 2 to 5.** Commit `feat(bench): knapsack, cascade and fixed allocators`.

### Task 9: Oracles and validity checks

**Files:** Create `bench/purse/oracles.py`, `bench/purse/checks.py`; Test `bench/tests/test_purse_checks.py`

**Interfaces:**
- `oracles.py`: `truth(table, job, option) -> tuple[float, float]` (per-job rate and cost);
  `expected_score(p, alloc, table) -> float`; `expected_cost(p, alloc, table) -> float`;
  `oracle_full(p, table, market, budget_usd) -> Allocation` (knapsack on per-job truth, no margin);
  `c_full(p, table, market) -> float` (per job the option with the highest rate, ties to the cheapest; summed cost);
  `oracle_cells(p, stats, market, budget_usd, hidden: bool) -> Allocation`;
  `voi(p, table, stats_same_fold, market, budget_usd) -> float` (expected score of `oracle_cells(hidden=True)` minus
  that of `hidden=False`, both scored by `expected_score`); `sweep_prices(table, pool, market) -> tuple[float, ...]`
  and `oracle_buys(p, table, market, budget_usd) -> set[str]` (ledger pre-flight ruling 2).
- `checks.py`: `@dataclass(frozen=True) class Check: name: str; value: float; threshold: str; passed: bool`;
  `run_checks(pool: Sequence[Job], table: OutcomeTable, market: Sequence[Option], seed: int = 0) -> list[Check]`,
  which builds its own projects (30 per ρ, folds alternating) and returns V1 to V7 with the spec's §12 thresholds.
  V7's heuristic answers each job with the most common answer of its family in the other fold.

- [ ] **Step 1: Failing tests**, on synthetic tables written in the test: `test_oracle_full_bounds_every_allocator`
  (it scores at least the knapsack, cascade replay and every fixed policy on 20 projects);
  `test_voi_is_zero_when_hidden_bins_carry_no_information` (a table where rates ignore the bin);
  `test_checks_pass_on_a_table_built_to_pass_and_fail_on_one_built_to_fail` (each of V1 to V7 flips when its
  condition is broken on purpose).
- [ ] **Steps 2 to 5.** Commit `feat(bench): oracles and validity checks`.

### Task 10: HASHIRU, a switch for the web tools

**Files:** Modify `src/hashiru/config.py` (`web_tools: bool = True`), `src/hashiru/tools.py:59` (offer
`get_website` and `arxiv_search` only when on), `src/hashiru/conf/config.yaml` (the setting, with a comment),
`README.md` (*Tools*: one sentence); Test `tests/test_tools.py`

- [ ] **Step 1: Failing test** `test_web_tools_can_be_switched_off`: with `web_tools=False`, the hub's MCP tool list
  has neither name and still has `check_usage`; with the default, both are listed.
- [ ] **Steps 2 to 4:** run (FAIL), implement, run the whole suite.
- [ ] **Step 5: Commit** `feat(tools): web_tools switch, so a benchmark can take web lookups away`.

### Task 11: HASHIRU, a memory cap

**Files:** Modify `src/hashiru/config.py` (`memory_mb: int = 0`, 0 meaning no cap), `src/hashiru/hub.py` (new
`Hub.free_memory() -> int`; `fits` uses it), `src/hashiru/approvals.py:203`, `src/hashiru/planning.py:274` and
`src/hashiru/tools.py:246` (use `hub.free_memory()`), `src/hashiru/conf/config.yaml`, `README.md` (*Budgets*: one
sentence); Test `tests/test_hub.py`

`Hub.free_memory()` returns the measured `free_mb()` or, with a cap, `min(free_mb(), memory_mb - held)`, where
`held` is the sum of `gpu_mb` over distinct local models that live agents use, in every session.

- [ ] **Step 1: Failing tests**: `test_memory_cap_refuses_a_second_model_that_would_pass_it` (cap 120 MB, the fake
  models take 100 MB each: the second local hire on another model is refused, a second hire on the same model is
  not); `test_no_cap_keeps_measured_memory` (monkeypatched `free_mb`).
- [ ] **Steps 2 to 4.** Existing tests that monkeypatch `free_mb` keep passing (move their patch target to
  `hashiru.hub` only if they break).
- [ ] **Step 5: Commit** `feat(hub): memory_mb caps the memory local hires may take`.

### Task 12: The `csp` agent and the `grid_check` server

**Files:** Create `bench/purse/peers/__init__.py`, `bench/purse/peers/csp_agent.py`,
`bench/purse/peers/grid_check.py`; Test `bench/tests/test_purse_peers.py`

**Interfaces:** `csp_agent.py` mirrors `examples/calc_a2a_agent.py`: `build_app(price_usd: float) -> Starlette` and
`main()` (`--port`, `--price`); it parses the last puzzle in a message with `grids.parse`, solves it and answers with
`<answer>{json}</answer>`, declaring its price through the cost extension and reporting `hashiru_cost_usd`.
`grid_check.py`: an MCP stdio server (`mcp.server.fastmcp`) with one tool `grid_check(puzzle: str, answer: str) ->
str` returning `"valid"`, `"invalid"` or `"limit reached"` (3 calls per puzzle, keyed by its SHA-256).

- [ ] **Step 1: Failing tests**: `test_csp_agent_solves_a_generated_grid` and
  `test_calc_agent_answers_the_canonical_arithmetic_prompt` (the example agent, on a `make_job("arithmetic", 3, ...)`
  prompt, returns the job's answer: the prompt must keep the expression as its last arithmetic);
  `test_csp_agent_is_hired_and_paid_like_any_peer` (as `tests/test_examples.py` does for `calc`);
  `test_grid_check_limits_calls_per_puzzle`.
- [ ] **Steps 2 to 5.** Commit `feat(bench): csp agent and grid_check tool`.

### Task 13: The HASHIRU adapter

**Files:** Create `bench/purse/hashiru_run.py`; Test `bench/tests/test_purse_hashiru.py`

**Interfaces:**
- Consumes `Hub`, `Settings`, `A2APeer`, `McpServer` (`hashiru.config`), `runtime.perform`, `runtime.ask_agent`,
  `runtime.bearer`, `planning.begin`, `planning.end`, `planning.Plan`, `hashiru.session.USER`, and Tasks 6 to 8.
- Produces:
  - `bench_settings(p: Project, market: Sequence[Option], *, out_dir: Path, max_usd: float, port: int = 9250,
    max_escalations: int = 2, peer_urls: Mapping[str, str] | None = None, ceo_model: str = CEO_MODEL) -> Settings`:
    `models` = the
    local and cloud options, `a2a_peers` from `peer_urls` with `price_usd_per_message` from the option,
    `mcp_servers` = `grid_check` over stdio (`sys.executable -m purse.peers.grid_check`, `PYTHONPATH=<repo>/bench` in
    its env, `labels=("trusted",)`, `egress=False`), `budget_usd=p.budget_usd`, `memory_mb=p.memory_mb`,
    `task_timeout=p.deadline_s`, `web_tools=False`, approvals 0, `trace_dir` and `data_dir` under `out_dir`.
  - `class HubMarket` (an async context manager): `__init__(self, hub: Hub, session: Session, reserve_usd: float)`;
    opens an MCP client session as `ceo` the way `runtime.run_agent` does; `attempt(job, option) -> Outcome` hires a
    worker `w{n}` with `WORKER_PROMPT`, the option as `model` and `reserve_usd`, asks it `job.prompt`, fires it, and
    returns the reply, `correct`, the hire's spend from the ledger, the seconds taken, and `refused` with the hub's
    error when the hire fails.
  - `@dataclass(frozen=True) class LiveRun: answers: dict[str, str]; spent_usd: float; ceo_usd: float; trace: str;
    reply: str`; `async run_hashiru(hub: Hub, p: Project, text: str) -> LiveRun` (an act turn from `USER` to the
    CEO; spend from the ledger; the session closed afterwards).
  - `async propose_split(hub: Hub, p: Project, text: str) -> tuple[Allocation, float]` (a plan turn: `begin(...,
    "plan")`, `ask_agent`, `end`; the CEO's spend); `plan_to_allocation(plan: Plan, p: Project) -> Allocation`
    (a hire named after a job id, case-insensitive, gives that job its model; a job without one goes to
    `CEO_MODEL`, which the CEO answers itself). The text for this condition adds: "Plan one worker per job you
    delegate, named after the job id."
  - `variant_text(base: str, variant: str) -> str` for the HASHIRU variants: `"names-models"` adds "Name a model
    for every hire; do not give a difficulty." (v1's way); `"split"` adds the line above; `""` leaves the text as it
    is. The uninformed condition is `project_text(..., stats=None)`; a stronger CEO is `bench_settings(...,
    ceo_model=...)`.

- [ ] **Step 1: Failing tests**, with a scripted fake LLM and `FakeOllama` (in `bench/tests/conftest.py`):
  `test_hub_market_hires_asks_fires_and_reports_cost`; `test_hub_market_reports_a_refused_hire` (a memory cap the
  local model does not fit); `test_run_hashiru_reads_the_ceos_json_answer`;
  `test_plan_to_allocation_maps_hires_to_jobs_and_the_rest_to_the_ceo`; `test_bench_settings_turn_off_web_and_approvals`.
- [ ] **Steps 2 to 5.** Commit `feat(bench): run allocators through a HASHIRU hub`.

### Task 14: What a live run did, from its trace

**Files:** Create `bench/purse/traces.py`, `bench/purse/metrics.py`; Test `bench/tests/test_purse_traces.py`

**Interfaces:**
- `traces.py`: `read(path: Path) -> list[dict[str, Any]]`; `@dataclass(frozen=True) class Hire: name: str; boss:
  str; model: str; budget_usd: float; difficulty: str; spent_usd: float; fired: bool`; `hires(events) -> list[Hire]`
  (successful `hire_agent` tool events, `llm` costs and `fire_agent` events); `escalations(events) -> list[tuple[str,
  str, str]]` (agent, from, to); `jobs_of(events, p: Project) -> dict[str, list[str]]` (job id to the hires that
  were sent its task text or its id, in order); `live_allocation(events, p) -> dict[str, list[str]]` (job id to the
  models tried, escalations included); `declared(events, p) -> dict[str, str]` (job id to declared difficulty);
  `prompt_fidelity(events, p) -> float`; `slack(events) -> float` (reserved minus spent, over fired hires).
- `metrics.py`: `bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float]`;
  `paired_ci(a, b, n=2000, seed=0) -> tuple[float, float]`; `efficiency(score, floor, oracle) -> float`;
  `spearman(x, y) -> float`; `confusion(declared: Mapping[str, str], hidden: Mapping[str, str]) ->
  dict[tuple[str, str], int]` (declared easy, medium, hard against hidden low, mid, high);
  `buy_share(attempts: Sequence[Attempt], market) -> float`.
- Replaying HASHIRU's decisions: `async replay_live(p: Project, tried: dict[str, list[str]], market: Market) ->
  RunResult` in `traces.py`: the last model tried per job, scored on the table.

- [ ] **Step 1: Failing tests**, on a trace written by a fake-model run of Task 13's adapter (CEO hires `w1` on qwen
  for `j1` with difficulty easy, it fails, it escalates to Luna, CEO fires it): `test_hires_spend_and_refunds`;
  `test_live_allocation_follows_escalations`; `test_declared_difficulty_per_job`; `test_prompt_fidelity`;
  `test_spearman_and_bootstrap_on_known_values` (hand-computed cases).
- [ ] **Steps 2 to 5.** Commit `feat(bench): read allocations and spend from HASHIRU traces`.

### Task 15: The `purse` command

**Files:** Create `bench/purse/cli.py`, `bench/purse/__main__.py`; Test `bench/tests/test_purse_cli.py`

**Interfaces:** `main(argv: Sequence[str] | None = None) -> int`, run as `uv run python -m purse <command>`:
- `pool --seed S --out FILE`: writes the 480 jobs (JSONL) and prints infeasible cells.
- `calibrate --pool FILE --table FILE --options A,B --repeats R [--levels 1,3,5] [--per-cell N] --max-usd X
  --spend`: one `HubMarket` attempt per (job, option, repeat) not yet in the table, `max_escalations=0`,
  $0.25 reserved per hire, `max_usd=X` as the hub's wallet. Without `--spend` it prints the planned call count and
  exits 2. (`--levels 1,3,5 --per-cell 5` is the pilot.)
- `check --pool FILE --table FILE`: prints V1 to V7 and exits 1 if any fails.
- `run --allocator NAME --pool FILE --table FILE --rho R --tau T --projects N --seeds S [--price P]
  [--memory-mb M] [--ceo MODEL] [--uninformed] [--live] --out DIR [--spend]`: NAME is `hashiru`,
  `hashiru-names-models`, `llm-split`, `knapsack`, `cascade`, `fixed:<option>`, `hybrid`, `uniform`, `random`,
  `oracle`. Replayed allocators need no `--spend`; `hashiru*`, `llm-split` and any `--live` run do. `--live` runs a
  replayed allocator through `HubMarket` instead (check V8: compare with its replay, with `metrics.paired_ci`).
  Writes one JSON line per (project, seed) with the score, spend, allocator spend and allocation, and `run.json` in
  DIR with `environment()`.
- `environment() -> dict[str, str]`: the git SHA, the `uv.lock` SHA-256, Python and platform, the Ollama version
  (`/api/version`, or `"unreachable"`), the price map's date (`model_catalog.json`'s mtime, or `"bundled"`), and
  the command line.

- [ ] **Step 1: Failing tests** (fake markets, no hub where possible): `test_live_commands_refuse_without_spend`;
  `test_calibrate_resumes_and_records_refusals` (Review Focus 5: a fake market that raises on the 3rd call; rerun
  skips the first 2; a refusal is a record with its reason); `test_check_exits_nonzero_on_a_failing_table`;
  `test_run_replays_knapsack_without_spend`; `test_run_writes_the_environment` (every key present, no secret: the
  values contain no `sk-` and no `LLM_`).
- [ ] **Steps 2 to 5.** Commit `feat(bench): the purse command`.

### Task 16: Close out

- [ ] **Step 1:** `uv run pytest -q`, `uv run ruff check .`, `uv run mypy`: all pass. Two mutation checks per core
  module (knapsack, `Pool.allows`, `draw_hidden`, `parse_answers`, the split count): each mutation must fail a test.
- [ ] **Step 2:** `bench/README.md`: what PURSE measures, the commands, and that live commands spend money.
- [ ] **Step 3:** Mark §17 steps 2 to 4 done in the spec and Phase D's items in the whitepaper plan; commit
  `docs(bench): PURSE readme and plan status`.
- [ ] **Step 4:** Ask the user before the pilot (`calibrate --levels 1,3,5 --per-cell 5`), with its call count and
  cost estimate.
