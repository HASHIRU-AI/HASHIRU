# PURSE

PURSE (working name) measures how agents split a money pool across the models and agents they hire. A project is 8
jobs with exact answers and one budget. Each job declares a level, but a hidden part of its difficulty is only learned
by trying, and a dial (rho) sets how often the level tells the truth. The market has local models (they cost memory),
cloud models (they cost dollars) and outside agents (a price per message). Because every job is calibrated on every
option, the best possible split is known for every project and budget, so "spent well" has a reference.

The design is in [`plan/2026-11-01-allocation-benchmark.md`](../plan/2026-11-01-allocation-benchmark.md). This
directory is not part of the `hashiru` package: `purse` imports HASHIRU, never the reverse.

## What is here

| Module | What it does |
|---|---|
| `purse/families/` | The four job generators, each with an exact checker: arithmetic, logic grids, record counting in long documents, string pipelines |
| `purse/projects.py` | The calibration pool, projects drawn from it with rho, the text an allocator reads, scoring |
| `purse/market.py`, `table.py`, `replay.py` | The options and their prices; the outcome table (JSON lines, resumable); a market that answers from it |
| `purse/solvers.py` | Allocators that are not LLMs: the knapsack (up front), the cascade (during the task), fixed policies |
| `purse/oracles.py`, `checks.py` | The best possible splits, the value of information, the agents' price sweep, and validity checks V1 to V7 |
| `purse/hashiru_run.py`, `traces.py` | HASHIRU as an allocator, the up-front LLM split in plan mode, and what a live run did, read from its trace |
| `purse/peers/` | The `csp` agent (solves grids, priced over A2A) and the `grid_check` MCP tool; arithmetic is bought from `examples/calc_a2a_agent.py` |
| `purse/cli.py`, `runner.py` | The `purse` command |

## Commands

Run from the repository root with `PYTHONPATH=bench` (the hub reads `model_catalog.json` there for current
prices; `purse` is not installed as a package).

```bash
PYTHONPATH=bench uv run python -m purse pool --out bench/out/pool.jsonl                     # 480 jobs, about 12 s, $0
PYTHONPATH=bench uv run python -m purse check --pool bench/out/pool.jsonl --table bench/out/table.jsonl   # V1 to V7, $0
PYTHONPATH=bench uv run python -m purse run --allocator knapsack --pool ... --table ... --rho 0.5 --tau 0.5 --out bench/out/knapsack
```

`run` replays `knapsack`, `cascade`, `oracle`, `hybrid`, `uniform`, `random` and `fixed:<option>` on the outcome
table at no cost. By default replay uses the first hub's pool rule (a call is charged in full, even past the pool);
`--bounded` uses v2's (a call is bounded by its pool before it is made, and a call cut at its pool answers nothing).

Round 2 (2026-11-02) added:

- `pool --wide` makes wider hidden bins, so that knowing a job's bin is worth more (V5); without it, `pool` makes the
  first round's jobs exactly.
- `run --allocator hashiru --route-on-rates` lets hires name a job's kind (`arithmetic level 4`) and route on the
  success rates an informed allocator is told; `--by-reference` holds long sections back from the CEO and attaches
  what a message names. Each is a switch, so each can be ablated.
- The small tier is `gpt-5-nano` ($0.05 / $0.40 per million tokens), cheaper per token than `gpt-6-luna`. Muse
  Glimmer 30B on NIM and Gemma 4 31B on OpenRouter stay priced in `prices.json` but are unused.
- `a2a/gridfast` is a cheap, unreliable seller beside `csp`: run `python -m purse.peers.csp_agent --port 9303
  --price 0.00025 --name gridfast --wrong 0.4` and calibrate it like any option.
- The project text gives the budget, the market, the statistics and the answer format before the jobs, so that
  `--by-reference`, which holds back long sections (a section runs to the next heading), never holds them back.
- Live runs share one provider's per-minute token limit (2M for `gpt-6-luna`): round 2 ran at most five processes at
  once, and a rate-limited call waits and is sent again instead of failing the agent.

These call real models and refuse unless `--spend` is given:

- `calibrate` sends jobs to options through a HASHIRU hub (`--levels 1,3,5 --per-cell 5` is the pilot);
- `run --allocator hashiru` (and `hashiru-names-models`, `llm-split`) runs the CEO live;
- `run --live` runs a replayed allocator through a live hub, to check that replay matches live (V8).

Each takes `--max-usd`, the hub's cap on real spend. Outside agents must be running and named with
`--peer calc=http://127.0.0.1:9301/ --peer csp=http://127.0.0.1:9302/`:

```bash
uv run examples/calc_a2a_agent.py --port 9301 --price 0.001
PYTHONPATH=bench uv run python -m purse.peers.csp_agent --port 9302 --price 0.001
```

Outputs go to `bench/out/` (gitignored): `results.jsonl` with one line per project and seed, and `run.json` with the
environment (git commit, `uv.lock` hash, Python, Ollama version, the price map's date, the command).

## Tests

`uv run pytest -q bench/tests` runs PURSE's tests with scripted models: no keys, no Ollama, no money.
