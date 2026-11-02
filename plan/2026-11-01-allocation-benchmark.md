# Design: PURSE, a benchmark for how agents spend (working name)

Status: design for the user's approval; nothing built. Date: 2026-11-01. Replaces the GAIA proposal and HASHIRUBench in
the whitepaper plan (`plan/2026-11-01-whitepaper-v2.md`, §5 and W7). The user asked for "a new hard benchmark that will
guarantee it tests what you want" and skipped the questions; every call below is mine unless it says otherwise.

## 1. Why

The paper's claim (§3 of the whitepaper plan) is a measurement: how LLM agents divide money they hold when they hire
other agents during a task, and how that compares with an optimizer at equal dollars. Existing suites (GAIA, v1's
HASHIRUBench) cannot carry that claim: the best split of the money is unknown, so "spent well" has no reference;
nothing controls how much of a job's difficulty is visible before trying, so adaptivity cannot be separated from luck;
and their answers may be on the web or in training data.

"Guarantee it tests what you want" means construct validity, not a win for HASHIRU: the score must move with the
allocation, the best allocation must be computable, and some conditions must be built so that adapting during the task
cannot help (the controls). §12 lists the checks that hold the benchmark to this before any agent runs.

## 2. Goals, non-goals, anti-goals

**Goals.**
- G1. The split of the money decides the score: a good allocator beats every fixed split by a clear margin.
- G2. A known optimum: for every project and budget, the oracle allocation and its expected score are computable.
- G3. Controlled observability: a knob sets how much of each job's difficulty is visible before trying, with a setting
  where it is all visible (the control).
- G4. Equal information: every allocator compared in the 2x2 gets the same project text, prices and statistics.
- G5. Hard: the strongest model in the market fails a real share of the hardest jobs (headroom, no ceiling).
- G6. Fresh: procedurally generated from seeds, exact answers, nothing to look up.
- G7. Cheap: calibration and the paper's runs fit the $50 cloud proposal (W4), with a pilot first.
- G8. Usable by others: any system that can hire models within a budget plugs in through one interface.

**Non-goals.** Decomposition (each project's jobs are given; we measure allocation, not planning); privacy and security;
open-ended answers that need a judge; web research; code execution of model output.

**Anti-goals** (failures even if it runs):
- Built so HASHIRU wins (only hidden-difficulty projects, or baselines starved of information).
- Allocation does not matter (all jobs need the same tier, or the budget never binds).
- A shortcut beats the models (a regex, a lookup, a web calculator, guessable answers).
- A verifier that can be gamed, or a score that rewards style.

## 3. Prior art it must not duplicate

| Benchmark | What is allocated, by whom | What PURSE adds |
|---|---|---|
| R³-Bench (2608.16033, Aug 2026) | One model splits its own output tokens or actions across a six-problem suite; difficulty hidden; an offline oracle as a multiple-choice knapsack over budget levels from single-problem response curves; equal split and two schedulers as fixed policies | Money spent by hiring other models and agents at different prices, not one model's own thinking; make or buy; memory as a second currency; LLM against solver allocators, up front and during the task |
| EcoAgent-Bench (2608.05519, Aug 2026) | One agent picks priced actions (lookup, search, a stronger model tier, human escalation) per task; seven agents and four oracle scripted controls | A pool shared across a project's jobs (a knapsack, not per-task choices); hires instead of actions; an observability knob |
| AllocBench (2607.23332, Jul 2026) | Whether an agent invests a fixed budget in reusable tools | Allocation across hired models and agents |
| BAGEN (2606.00198) | Whether agents estimate their remaining spend; finds over-optimism | Allocation quality against an oracle, not self-estimates |
| EnterpriseArena (2603.23638) | Simulated capital across a firm's units | Real model calls, real prices, exact answers |
| RouterBench (2403.12031) | A router picks one model per query from precomputed outputs; an oracle router | A shared budget across jobs, hidden difficulty, hires in a tree, outside agents |
| CostBench (ACL 2026) | One tool-use agent plans a cost-optimal sequence of tools and replans when tools fail or prices change | Uncertain success of hired models, a pool shared across jobs, hidden difficulty |
| TokenCliff (EMNLP 2026), EconEvals (2503.18825) | Episode-level token budgets; economic decisions in simulated settings | Money spent on hires with real prices and exact answers |
| ZEBRA (2605.20485) | A solver splits dollars across a fixed pipeline before execution; its LLM-direct baseline loses to a knapsack | Both of ZEBRA's allocators appear as cells of the 2x2, next to agents allocating during the task |

Credit in the paper: the oracle as a multiple-choice knapsack (R³-Bench), evaluation on precomputed outcomes
(RouterBench). All of these are to be verified at their venue of record before they enter `refs.bib`.

## 4. A project

A project is K jobs (K = 8 by default) with one budget. Each job has:
- a family (§5) and a visible level 1 to 5 (the published size parameters);
- a hidden difficulty (§6), not published;
- a value v ∈ {1, 2, 4} points, visible and drawn independently of difficulty, so value, difficulty and cost all trade;
- a canonical prompt and an exact verifier.

The project text lists every job with its id, family, visible level and value, and asks for one JSON object mapping
each id to its answer. Score = Σ v·correct / Σ v. Each project has a budget B (the allocator's whole pool, its own model
calls included), a memory cap M for local models, and one deadline T. Projects mix families (two jobs per family) and
levels.

## 5. Families (procedural, seeded, exact verifiers)

| Family | Job | Visible level sets | Hidden difficulty sets | Can a cheap check tell a wrong answer? | Outside agent to buy from |
|---|---|---|---|---|---|
| F1 Arithmetic | Evaluate a nested integer expression exactly | Number of operations, digit lengths | Carries and borrows, intermediate magnitudes, operator mix | No | `calc` (the existing example agent, exact) |
| F2 Logic grids | Solve an n-by-m grid puzzle with a unique solution | n, m, number of clues | Deduction depth: propagation rounds and case splits a solver needs | Yes: `grid_check` (all clues hold or not) | `csp` (a constraint solver behind A2A) |
| F3 Aggregation | Count or sum the records in a long document that match a condition | Document length, record count | Matches, near-miss distractors, conditions that span fields | No | None |
| F4 String pipelines | Apply k string operations in order to an input | k, input length | Interacting operations (order matters, edge characters) | No | None |

Why these four: two can be bought and two only made; one has a cheap check (verification is cheaper than solving, as
with unit tests) and three do not; F3 separates local from cloud by context length (a 16K-token local window cannot hold
a long document, and a bigger window costs memory). All answers are integers, a grid assignment or an exact string, so
guessing scores near 0. Levels are tuned (§12, V3 and V4) so that level 5 is hard for the strongest model.

Cost varies with difficulty: reasoning models spend more output tokens on harder jobs, so what a job costs is also
uncertain before trying. The calibration records it.

## 6. Hidden difficulty and the observability knob

Each job draws its visible level ℓ_v uniformly from 1 to 5. With probability ρ its hidden level ℓ_h equals ℓ_v;
otherwise ℓ_h is drawn independently (three bins: low, mid, high). The generator sets the visible parameters from ℓ_v and
the hidden ones from ℓ_h, so the visible level predicts difficulty partly, and ρ sets how much.

- ρ = 1: what the text declares is what the job is. An up-front allocator with the statistics has all the information;
  adapting during the task can only help through the models' own randomness. This is the control.
- ρ = 0.5 and ρ = 0: difficulty is learned by trying.

The value of information VOI(ρ, B) = oracle score with the hidden levels − oracle score with only the visible ones,
computed from the outcome table (§8). It bounds what any adaptive allocator can gain over the best up-front split, and
it is reported for every cell, so a result for "during the task" can always be read against what was possible.

An LLM allocator reads the job's text and may estimate hidden difficulty from it (count the carries, notice the
distractors); the solver sees only the declared features. That is part of what is measured (do LLM allocators read
difficulty better than features do?), and the paper says so.

## 7. The market

| Option | Kind | Price | Memory |
|---|---|---|---|
| `ollama_chat/qwen3.5:4b` | local | $0 | about 3.5 GB at a 16K context |
| `ollama_chat/gemma4:12b` | local (pulled on this machine) | $0 | to measure; with M = 12 GB the two local models cannot both stay loaded with long contexts |
| `gpt-6-luna` | cloud | $0.10 / $0.50 per M tokens | none |
| `meta/muse-spark-1.3` | cloud | $1.25 / $4.25 per M tokens | none |
| `a2a/calc`, `a2a/csp` | outside agents, cost extension | p per message, swept | none |

Outside agents are exact programs, so their reliability is known; a later variant may make one fail with a hidden
probability. `grid_check` is an MCP tool, free, limited to 3 calls per job, answering only valid or invalid. Workers
have no tools; web tools are off for every agent (§13).

Rational demand is what oracle-full buys at each price (local models cost $0, so a per-job break-even price against
making would be 0 for any job a local model ever solves). Prices are c_ref × (0.25, 1, 4, 16), where c_ref is the
median cost of `gpt-6-luna` on arithmetic and grid jobs; the default is c_ref. Each allocator's demand curve (the
share of those jobs it buys at each price) is read against the oracle's. An outside agent's cost in replay is its
current price; only its correctness comes from the table.

## 8. Calibration: the outcome table

Every generated job is sent, with its canonical prompt, as one worker call to each model in the market: answer,
correct or not, dollars, latency and memory. (Tokens and the response's model name are not recorded: HASHIRU's
`llm` trace event carries only `cost_usd`, which is what the oracles and the pilot's nonzero-cost check need.)

- Grid: 4 families × 5 visible levels × 3 hidden bins = 60 cells, 8 jobs per cell = 480 jobs.
- Repeats: 2 per job for the local models and `gpt-6-luna`, 1 for `muse-spark-1.3` (cost).
- Statistics without leakage: jobs are split into two folds. A project drawn from one fold gets its statistics (success
  rate and mean cost per family, visible level and model) from the other fold.

The table serves three purposes:
1. The statistics allocators receive in the informed condition.
2. The oracles (§9) and the validity checks (§12), at $0 once measured.
3. Replay: an allocator that sends canonical prompts can be scored on recorded outcomes instead of new calls (as
   RouterBench does). The up-front cells, the fixed teams and the cascade run this way. HASHIRU runs live, and its
   decisions (which model for which job, in which order, with which escalations) are also replayed on the table.
   This separates how well it allocated from how well its CEO rewrote the prompts.

The calibration's curves (success and cost against level, per model) are themselves a figure for the paper.

## 9. Allocators compared

All allocators run in the informed condition. They get the same project text, prices and fold statistics, the same
budget, the same market and the same worker prompt.

| | Up front (fixed before any job runs) | During the task |
|---|---|---|
| **LLM** (`gpt-6-luna`) | **LLM split**: the CEO writes a plan in HASHIRU's plan mode, one hire per job it delegates, named after the job. The plan becomes a fixed allocation that runs unchanged, with no escalation, through the same executor as the knapsack, so the two up-front cells differ only in who allocated (ZEBRA's LLM-direct) | **HASHIRU**: the CEO hires, asks, fires, rehires and escalates while results arrive. Managers may hire too (depth 2) |
| **Solver** | **Knapsack**: a multiple-choice knapsack over (job, model or agent) that maximizes expected value under the budget, from the fold statistics (ZEBRA- or BAMAS-style) | **Cascade**: per job, in value-per-cost order, the cheapest option whose expected success clears a threshold; on a failed `grid_check`, or when two cheap answers disagree, move up a tier, until the pool is spent. Thresholds fit on the other fold |

Around the 2x2:
- **Oracles**: oracle-full (knows each job's recorded outcomes) bounds every allocator; oracle-visible is the knapsack
  with true per-cell rates, which bounds up-front allocators.
- **Fixed policies**: a single agent (the CEO's model answers everything, no hires); every job on one model
  (`qwen3.5:4b`, `gpt-6-luna`, or `muse-spark-1.3` until the money runs out); a fixed hybrid by visible level, in
  AgentCARD's style (local for levels 1 and 2, Luna for 3 and 4, Spark for 5); uniform split; random.
- **HASHIRU variants**: the CEO names models, as in v1 (no declared difficulty); a stronger CEO (`muse-spark-1.3`, the
  allocator ablation); the uninformed condition (prices and model names only, no statistics), which is what a real
  deployment has.

## 10. Conditions

| Factor | Levels | Default |
|---|---|---|
| Observability ρ | 1 (control), 0.5, 0 | 0.5 |
| Budget tightness τ = B / C_full, where C_full is what oracle-full spends at its best score | 0.25, 0.5, 1 (money does not bind: a control). No τ = 0: a cloud CEO pays for its own turns from B, so it could not think; local-only is a fixed policy instead | 0.5 |
| Memory cap M | 12 GB, 24 GB | 12 GB |
| Outside agent price p | c_ref × (0.25, 1, 4, 16) (§7) | c_ref |
| Information | informed, uninformed | informed |

## 11. Metrics

- **Outcome**: score (value-weighted accuracy) at equal B; dollars spent; regret to oracle-full; allocation efficiency
  (score − score of the cheapest fixed policy) / (oracle-full − that score); the cost-success Pareto front per allocator.
- **How agents spend (RQ1)**, from the trace: reservation against spend (slack), refunds, money left unspent at the end;
  declared difficulty against hidden difficulty (a confusion matrix and Spearman's ρ); the tier chosen against the
  knapsack's tier; escalations and how often they paid; make or buy share at each price p against oracle-full's; team size and
  depth as B grows; memory held; wall-clock.
- **Allocator overhead**: what the allocator spent on its own model calls (an LLM allocator pays for its thinking
  from B; a solver's compute is free), reported apart from what it spent on jobs.
- **Prompt fidelity**: whether a worker received the job's canonical text; live against replay score.
- **Statistics**: 3 seeds for live runs; paired comparisons on the same projects; bootstrap confidence intervals over
  projects; differences inside the seed spread are reported as within seed variance, with both numbers.

## 12. Validity checks (what makes the guarantee)

Computed from the outcome table before any allocator runs, at $0. Thresholds are mine and tentative; they and their
results are published with the benchmark. A failed check means the generator is retuned and the affected cells are
calibrated again, within the budget.

| Check | Holds when |
|---|---|
| V1 Allocation matters | At τ = 0.5, oracle-full beats the best fixed policy by at least 10 points |
| V2 The budget binds | At τ ≤ 0.5, giving every job to `muse-spark-1.3` covers under 60% of the value |
| V3 Tiers separate | Each family has a level where Luna ≥ 80% and Qwen ≤ 40%, and one where Spark − Luna ≥ 20 points |
| V4 Headroom | Spark ≤ 60% at level 5 in at least two families |
| V5 Observability works | VOI ≤ 2 points at ρ = 1 and ≥ 8 points at ρ = 0 |
| V6 Make or buy responds | Oracle-full's share of F1 and F2 jobs bought falls by at least 40 points from the cheapest swept price to the dearest |
| V7 No shortcut | A no-LLM heuristic scores ≤ 10%; agents have no web tools |
| V8 Replay matches live | On a 10% live subset of the replayed allocators, live and replay scores agree within seed variance |

## 13. Harness

- **Where**: `bench/purse/` in this repo on `feature/whitepaper`, a separate package that HASHIRU does not import:
  - one generator and verifier per family;
  - project assembly;
  - the outcome table (JSONL, with the price map's date);
  - the oracles, knapsack and cascade;
  - replay, metrics and the validity checks;
  - the `csp` A2A agent and the `grid_check` MCP server.
- **One interface**: an allocator takes (project, budget, market, statistics) and returns its answers plus an
  allocation log. HASHIRU's adapter runs an in-process `Hub` with config overrides: `budget_usd` = B, `models` = the
  market, `a2a_peers`, `mcp_servers` with `grid_check`, approvals off (thresholds 0), and `task_timeout` = T. The LLM
  split adds `default_mode=plan`, an automatic `approve` and `max_escalations=0`. The solvers act through the same tools,
  as a scripted CEO in place of the LLM, so their hires go through the same ledger and trace.
- **Needed in HASHIRU** (small changes, tests first):
  - a switch that turns off the built-in web tools (`get_website` and `arxiv_search` are always offered today, so a
    CEO could look an answer up);
  - a memory cap setting (free memory is measured from the machine today, `ledger.free_mb`, so M cannot be set, and
    runs would depend on what else is running);
  - the W9 fix, merged first.
- **Records per run**: config, seeds, model response names, the price map's date, Ollama version, hardware, git SHA,
  `uv.lock` hash.
- **Tests with fake models**, $0: generators are deterministic per seed; verifiers accept the right answer and reject
  near misses; grids have unique solutions; ρ produces the stated agreement between visible and hidden levels; the
  knapsack matches brute force on small cases; replay is deterministic.

## 14. Cost and time

Estimates until the pilot measures real token use (reasoning tokens dominate):

| Item | Calls | Estimate |
|---|---|---|
| Calibration, `muse-spark-1.3` | 480 | about $10 |
| Calibration, `gpt-6-luna` | 960 | about $3 |
| Calibration, local | 1,920 | $0 (several hours on the M5) |
| HASHIRU live: 3 ρ × 3 τ × 10 projects × 3 seeds | 270 projects | about $16 |
| LLM split (plan calls; execution by replay) | 270 plans | about $2 |
| HASHIRU variants on 30 projects × 3 seeds (names models; Spark CEO) | 180 projects | about $12 |
| Make or buy: 4 prices × 10 projects × 3 seeds | 120 projects | about $5 |
| **Total** | | **about $48** |

Order: a pilot (5 jobs per cell at levels 1, 3 and 5) gives a firm estimate before any batch; if it exceeds the cap,
the Spark-CEO ablation is cut first. Build: about 5 days (generators 1.5, oracles and solvers and replay 1, the HASHIRU
adapter and harness 1.5, `csp` and `grid_check` 0.5, validity checks 0.5). Calibration: 1 day. Runs: 2 to 3 days.

## 15. Risks

| Level | Risk | Mitigation |
|---|---|---|
| HIGH | Generator levels do not map to model difficulty (V3 or V4 fails) | The pilot measures it per family; retune before full calibration; drop a family that cannot be tuned |
| HIGH | Reasoning tokens make Spark far dearer than estimated | Pilot first; fewer Spark repeats; a smaller grid |
| MED | `gpt-6-luna` or Spark reject `temperature=0` (some reasoning models accept only the default) | The pilot's first call shows it; `llm.py` would then drop the parameter for them, tests first |
| MED | The knapsack wins in the informed condition | A finding, not a failure; RQ1 explains where agents lose |
| MED | Local calls are slow at long contexts | Calibrate overnight; the deadline T applies to everyone alike |
| MED | The CEO rewrites prompts badly | Prompt fidelity and replay separate allocation from prompting |
| LOW | Name collision | Checked before the name is fixed |
| LOW | Meta Model API rate limits | Throttle in the harness |

## 16. Calls I made (check these)

1. **Jobs are given, not discovered**: the benchmark measures allocation, not decomposition.
2. **Four families**: F1 to F4 above. No code generation, because running model-written code needs a sandbox.
3. **K = 8 jobs per project; values 1, 2, 4.**
4. **Informed by default**, so the 2x2 compares allocators, not information; the uninformed condition is a HASHIRU
   variant.
5. **`gemma4:12b` in the market**, so memory is a real second currency at M = 12 GB.
6. **Replay for the up-front cells, fixed teams and the cascade.** This saves most of the money; V8 checks it on a live
   subset.
7. **In the paper, the benchmark sits inside the contributions, not as a fourth one**: "How agents spend" and "Agents
   against optimizers" are measured on it.
8. **Name**: candidates are PURSE (Projects Under Reserved Spending Envelopes), BURSAR (Budgeted Use of Reserved
   Spending by Agents that Recruit) and TENDER (a call for bids, as in make or buy).

## 17. Build order, after approval

1. Branch `feature/whitepaper` from `develop`, after merging `bugfix/unpriced-models`.
2. Generators and verifiers, tests first; the validity checks running on a synthetic outcome table.
3. Oracles, knapsack, cascade and replay, tested against brute force.
4. The web-tools switch and the memory cap in HASHIRU (tests first), the `csp` agent, `grid_check`, the HASHIRU adapter
   with fake models.
5. Ask before the pilot (real calls), then before calibration, then before each batch of runs.
