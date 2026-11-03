# Plan: is there room for finetuning in HASHIRU and PURSE?

Status: **scope decided 2026-11-02 by the user: splitter only (Phases 0 to 2).** Phases 3 and 4 (open-weight market,
trained CEO) are recorded below as later work, not planned. **Phase 0 done the same day (§8): the splitter's
learnable headroom is about 0.035, so no finetuning. Instead (§9): the $0 fixes are built, and an open-weight pilot is ready and waits for the user to launch an instance.** The user has $500 to $2,000 in Lambda credits. No paid API
calls. Lambda instances are launched only after the user approves each batch (see [[no-real-model-runs]] in memory).
Date: 2026-11-02

## 1. The question, restated

Can training a model (finetuning, RL, or a small learned model) improve how HASHIRU's agents spend, given what round 2
measured? Which piece should be trained, with what data, on what compute?

Limits:
- No paid API calls (the user's rule after round 2's $33.21). Lambda credits pay for GPUs instead.
- The whitepaper is published. Its claim is measuring how untrained agents spend ([[whitepaper-v2-framing]]).
  Training asks a new question, "can agents learn to spend?", so this is follow-up work, not an edit to the
  whitepaper.

## 2. Where the points are lost (round 2, `bench/out/r2/analysis.txt`, runs-v2 table)

| Strategy | Score |
|---|---|
| Oracle (knows every recorded outcome) | 0.826 |
| Knapsack (up front, uses the per-level success rates) | 0.788 |
| `gpt-6-luna` alone | 0.736 |
| LLM split (one call, up front) | 0.695 |
| HASHIRU, both round-2 changes | 0.599 |
| HASHIRU as first built | 0.463 |

1. **Oracle minus knapsack: 0.038.** Mostly luck. V5 (what knowing each job's hidden difficulty is worth) came out at
   0.002, so reading difficulty better gains about nothing on this table.
2. **Knapsack minus HASHIRU: 0.189.** The agents' behavior: 11 of 60 runs lost, the CEO's own turns took 32% of the
   budget, 78 of 79 hires on the expensive model had a call refused (12 of 79 right), and jobs the CEO kept were 85%
   right against 91% for a fresh `gpt-6-luna` call.
3. **Knapsack minus LLM split: 0.093.** An LLM splitting the money in one call loses to the solver, as in ZEBRA.

**The key point.** On this table, the best any trained allocator can do is match the knapsack, because there is no
value in adapting during the task (V5 fails). GPUs do not change that; only a new table that passes V5 does. Lambda
credits matter because they make a new table affordable: an open-weight market served on rented GPUs costs credits,
not API dollars.

## 3. What Lambda credits change

| Before (Mac only, no API money) | With Lambda credits |
|---|---|
| Train only a 4B model, on short text, with mlx-lm (not installed, support unchecked) | Train 8B to 32B models on GPUs with standard tools (TRL, Unsloth or similar), on the full project text |
| No new results table (calibration needs paid API calls) | Calibrate an **open-weight market** served with vLLM on Lambda: more jobs per cell, chosen tiers, as many attempts as V5 needs |
| RL parked | RL on replay is affordable, and worth it once a table passes V5 |
| CEO is `gpt-6-luna`, which cannot be trained | An open-weight CEO, trained and untrained, in the same market |

How an open-weight market is priced: at public list prices for the same models from hosted providers, read on the run
date. `bench/prices.json` already does this for models run on trial credits (the NIM entries). What runs on Lambda
costs GPU time; what the agents are charged is the list price.

Runtime change needed: HASHIRU passes `api_base` only to Ollama models (`src/hashiru/llm.py:118`). Self-hosted models
need a per-model endpoint in the config (LiteLLM's `hosted_vllm/` provider takes one). Small, tested with fake LLMs.

## 4. Options, ranked

| # | What to train | Payoff | Compute (rough guess, GPU-hours) | Verdict |
|---|---|---|---|---|
| A | **One-call splitter.** An open model learns to write the knapsack's JSON split from the project text | Closes gap 3 at best (0.695 toward 0.788). Answers ZEBRA: can training fix an LLM that splits money badly? | 2 to 10 | **First.** Cheap, single turn, scored free in replay on the current table |
| M | **Open-weight market and a new table that passes V5.** Not training, but what makes B and C worth doing | A table where adapting during the task has value; reproducible by anyone, no API keys | 10 to 50, depending on the largest model | **Second.** The gate for B and C |
| B | **Trained CEO (SFT).** Tool-call sequences (hire with full funding, ask, fire, answer) rendered from the knapsack's or oracle's plan | Targets gap 2 | 10 to 30 | After M and the roadmap's code fixes, so it measures "training on top of the fixes" |
| C | **RL-trained CEO** on the replay table (reward = score minus spend), multi-turn | The real research question: can a trained agent that adapts beat a solver that plans up front? This is the 2x2's "LLM, during the task" cell, trained | 100 to 300 | Last, and only on a table that passes V5 |
| D | Specialize the hires on the four job families | A cheap local tier priced in memory | 5 to 20 | Park. Off the thesis |
| E | Learn the call cap's margin from billed against counted tokens | A tighter guard | 0 | Small regression; do when the guard is next touched |

The GPU-hour numbers are rough guesses. Check them against a pilot before launching anything big.

## 5. Phases

### Phase 0: confirm the headroom ($0, no model calls, about 2 hours, local)

- Knapsack-to-oracle gap and V5 at every budget in the sweep (τ 0.25 to 2).
- The 0.189 behavior gap split by cause, from the traces.
- Projects available per job fold without reusing a job.

**Gate:** if gap 3 holds at most budgets, go on with A.

### Phase 1: splitter dataset, scorer, endpoint support ($0, local, about 5 hours)

- `bench/purse/finetune/dataset.py`: draw projects, label each with the knapsack split, write JSONL chat pairs. Split
  by the pool's two job folds, never by project.
- A scorer that replays any model's JSON split (reuse the `llm-split` path in `hashiru_run.py` and `replay.py`).
- Per-model `api_base` in HASHIRU's config, for vLLM endpoints.
- Tests with scripted models, and mutation checks (break the fold split, break the endpoint, confirm a test fails).

### Phase 2: train and score the splitter (Lambda; asks first)

- One GPU instance. LoRA SFT on an open model in two sizes (a 4B-class and a 14B-to-32B-class, exact models chosen
  after checking what is current and licensed for this use).
- Serve with vLLM; score untrained and trained splits in replay on held-out jobs, three seeds, intervals over
  projects. Compare with the knapsack (0.788) and the stored `gpt-6-luna` split (0.695).
- Shut the instance down when done; record GPU-hours and the commit in `run.json`.

### Phase 3: open-weight market (Lambda; asks first, pilot before the full run)

- Pick three open tiers so that V3 and V4 can pass (the top tier must fail on hard jobs), plus the existing outside
  agents.
- Pilot: levels 1, 3, 5, two jobs per cell. Then the full pool with enough jobs per cell for V5 (the whitepaper says
  four per group was too noisy; aim for 8 to 16).
- `purse check` must pass V1 to V7 before Phase 4.

### Phase 4: trained CEO (Lambda; asks first)

- B first (SFT), then C (RL) only if B moves the score.
- Hires answer from the new table (a "replayed hires" mode, to build), so CEO training never calls a hire live.

## 6. Risks

- **HIGH: the ceiling on today's table is the knapsack.** A trained splitter can at best match it. The result worth
  having is "untrained loses, trained matches". If untrained already matches, report that and stop.
- **HIGH: a new market is a new benchmark round.** Results from open-weight tiers do not compare directly with the
  whitepaper's `gpt-6-luna` and Spark numbers. Treat it as round 3, not an extension of round 2.
- **HIGH: few jobs.** 240 jobs today. The fold split keeps the test honest; Phase 3 should grow the pool.
- **MEDIUM: idle GPUs burn credits.** Every launch script ends with shutdown; check the Lambda dashboard after each
  batch.
- **MEDIUM: RL on long multi-turn episodes is unstable and slow.** Keep C behind B's result.
- **LOW: license.** Check that each open model's license allows this use and publishing the results.

## 7. Effort

Phases 0 and 1: about a day, local, $0. Phase 2: about a day, a few GPU-hours. Phase 3: two to three days, tens of
GPU-hours. Phase 4: a week or more, up to a few hundred GPU-hours.

## 8. Phase 0 results (2026-11-02, stored round-2 data only, $0)

Scripts: `phase0.py` and `phase0b.py` in the session scratchpad (not committed). Replay reproduced the stored LLM-split
score exactly (0.695), so the counterfactuals below run on the same footing.

**Budget sweep (replayed, v2 rule, 20 projects x 3 seeds; paired 95% intervals over projects):**

| τ | Oracle | Knapsack | `gpt-6-luna` alone | Oracle minus knapsack | Knapsack minus Luna |
|---|---|---|---|---|---|
| 0.25 | 0.767 | 0.715 | 0.697 | +0.053 [+0.018, +0.097] | +0.017 [-0.040, +0.068] |
| 0.5 | 0.826 | 0.788 | 0.736 | +0.038 [+0.006, +0.074] | +0.052 [-0.002, +0.108] |
| 1 | 0.848 | 0.842 | 0.768 | +0.006 [-0.039, +0.054] | +0.074 [+0.031, +0.121] |
| 2 | 0.950 | 0.902 | 0.784 | +0.047 [+0.022, +0.078] | +0.118 [+0.055, +0.183] |

The knapsack is within 0.006 to 0.053 of the oracle at every budget.

**Reading difficulty is worth nothing here.** With each project's statistics taken from the other fold, as the
allocators get them, knowing the hidden bin changes the expected score by -0.023 to -0.041 at ρ = 0.5 and -0.047 to
-0.067 at ρ = 0. It is negative because the finer cells rest on four jobs and three attempts each, so they are mostly
noise. (V5 reports +0.002 because it computes its statistics on the whole pool, in sample.)

**Why the LLM split loses 0.094 to the knapsack (τ = 0.5):**

| Split, replayed | Score | Minus knapsack |
|---|---|---|
| As run (Luna plans, then the plan runs) | 0.695 | -0.094 [-0.143, -0.043] |
| Plan kept, budget check added (drop jobs until expected cost fits) | 0.708 | -0.081 [-0.125, -0.036] |
| Plan kept, planning call free | 0.748 | -0.040 [-0.085, +0.004] |
| Both | 0.753 | -0.035 [-0.066, -0.005] |

- **0.053 is the price of the planning call.** Luna's one planning call took 14% of the budget on average, and more
  than the whole budget on one project (111%). Any cheap planner removes this, trained or not.
- **About 0.01 is budget overrun.** 22 of 60 plans expected to cost more than the budget. A plain check fixes it.
- **About 0.035 is model choice.** This is the only part a trained splitter would learn. It agrees with the knapsack on
  49% of jobs, sends 297 of 480 jobs to Luna (the knapsack sends 168) and never skips a job (the knapsack skips 39).

**Verdict.** Training a splitter on this table could win back about 0.035 [0.005, 0.066]. It would do that by
imitating a solver that is exact, free and already has the same statistics in hand. The larger losses (the planning
call, the overrun) need no training. Per the gate, Phase 1 does not start without the user's decision. The options:
1. **Stop finetuning on this table.** Take the cheap fixes into the runtime instead: plan in one call on a cheap model
   or with the knapsack as a tool, and check a plan's expected cost against the budget. They target 0.06 of the 0.094.
2. **Redirect the credits to Phase 3** (an open-weight market and a new table with 8 to 16 jobs per cell). Training only
   has something to learn once reading a job's text is worth points, which needs a table where V5 passes out of fold.
3. **Train the splitter anyway** as a small, cheap demonstration ("a small model, trained, matches the solver").
   Expected gain about 0.035, a few GPU-hours.

## 9. After Phase 0: the $0 fixes and the open-weight pilot (2026-11-02)

The user asked what to spend credits on; the recommendation was no finetuning, a capped pilot (at most about $50 of
credits) to test whether smaller open models make the hidden difficulty matter, and the $0 fixes first. The user said
"okay". Built the same day, tests first, mutation-checked (all 26 mutants caught after one test was strengthened), with
the full suites passing (475 in `tests`, 168 in `bench/tests`), ruff and mypy clean. No model calls.

**The $0 fixes (HASHIRU runtime).**
- *Quote at hire time* (the roadmap's first item). A hire by `kind` whose `budget_usd` cannot pay for one average
  attempt on its model is not made, in `hire_agent` and in `propose_plan`: the error quotes the attempt's measured cost
  and lists the measured models the budget can pay for, most reliable first. `propose_plan` now takes `kind`, and the
  plan graph hires such a hire by kind. (`router.Router.underfunded`, `tools.py`, `planning.py`, `runtime.PlanRunner`.)
- *Not built: a separate planning model.* The LLM split's planning calls averaged $0.0025 each (2.3 a run), and most
  of that is reading the project text, which `request_by_reference` already cuts. Measuring either needs live runs.
- *Per-model endpoints.* `endpoints: {model: url}` sends a model to its own OpenAI-compatible server (vLLM), priced
  per token from `prices_file` like any cloud model.

**PURSE.**
- `OPEN`: four Apache-2.0 models served by vLLM (`hosted_vllm/...`), priced at OpenRouter's list prices read
  2026-11-02, output capped at 32,768 tokens: Ministral 3 3B ($0.10 / $0.10 per M), Qwen3.5 9B ($0.17 / $0.25),
  Gemma 4 31B ($0.20 / $0.40), Qwen3.5 27B ($0.26 / $2.60). A test checks that LiteLLM charges each its list price.
- `--endpoint MODEL=URL` on `calibrate` and `run`; an open model without one is refused.
- `Tiers(small, mid, top)`: the checks, the outside agents' reference price and the hybrid policy now take the
  market's tiers; the API market's names are the default, and round 2's V1 to V7 are unchanged (rerun, same values).
- `purse/pilot.py` and `bench/lambda/pilot_report.py`: for every three-model ladder, V1 to V7 plus the value of the
  hidden bin out of fold (Phase 0's method); GO if some ladder passes V3, V4 and V5 and that value is positive.
  It also counts attempts cut at the output cap, since Qwen3.5's card warns that greedy decoding (HASHIRU calls at
  temperature 0) can repeat endlessly: a top tier that fails by looping is not headroom.

**The pilot run (needs the user: they launch and terminate the instance).**
1. Launch one GPU instance in the Lambda dashboard with your SSH key. Recommended: 1x B200 180 GB ($6.99 an hour on
   Lambda's page, 2026-11-02), so every model fits unquantized with a 131k context. Cheaper: 1x H100 SXM 80 GB
   ($4.29 an hour), with a risk that Gemma 4 31B cannot hold a 131k context next to its 62 GB of weights (the script
   then stops and says so).
2. `bench/lambda/pilot.sh ubuntu@INSTANCE_IP` on the laptop. For each model in turn it starts vLLM on the instance
   (listening only on the instance's localhost, reached over an SSH tunnel), runs 12 calibration slices in parallel
   (432 attempts a model), and stops vLLM. It loads no `.env`, so it cannot make a paid API call.
3. Terminate the instance in the dashboard when the script says so. The script cannot do it.
4. `PYTHONPATH=bench uv run python bench/lambda/pilot_report.py` prints the verdict.

Estimate (rough): 2.5 to 4 hours of instance time (installing vLLM, downloading about 145 GB of weights, loading four
models, 1,728 attempts), so about $18 to $28 on a B200 or $11 to $17 on an H100. The largest unknown is how long
Qwen3.5 27B thinks per job.

**Risks.** vLLM's flags come from each model card (read 2026-11-02) and are not yet tried on this vLLM version. The
pilot has four jobs per cell (two per fold), so its out-of-fold value is noisy; GO means "worth a full round", not a
result. A round on these models does not compare with round 2's API models.

## 10. The trained splitter (decided 2026-11-02: "pilot first then finetune")

On the same H100 (1x H100 SXM 80 GB, $4.29 an hour) once the pilot is done: `bench/lambda/finetune.sh`.

- **Task.** One reply that maps each job to an option or "skip", from the project's outline: budget, market, the
  statistics an informed allocator is told, and each job's heading with the first 160 characters of its text.
  Thinking off. The target is the knapsack's split of what the reply leaves (its own cost at list price comes first).
- **Data** (`bench/lambda/splitter_data.py`, $0). 3,000 training and 200 validation projects drawn from round 2's
  pool without any job of the 20 evaluation projects, at τ in {0.25, 0.5, 1, 2} and ρ in {0, 0.5, 1}. Prompts are
  about 1,600 tokens.
- **Model and training** (`train_splitter.py`, `splitter.yaml`). Qwen3.5 9B (Apache 2.0; a vision-language
  checkpoint with mostly linear-attention layers), LoRA r=16 on the language model's projections, loss on the answer
  only, one epoch, lr 1e-4, seed 2026; Transformers Trainer, not TRL. The adapter is merged and served by vLLM.
- **Evaluation** (`eval_splitter.py`). Round 2's 20 projects at each τ, three replay draws, under v2's pool rule, the
  reply charged at Qwen3.5 9B's list price. Compared with the untrained model (thinking off, and thinking on), the
  knapsack, and the stored gpt-6-luna split (0.695 at τ = 0.5).
- **Ceiling** (a splitter that writes every target exactly, reply paid): 0.687, 0.766, 0.833 and 0.900 at τ = 0.25,
  0.5, 1 and 2, against the knapsack's 0.715, 0.788, 0.842 and 0.902 for free. So the result worth having is "the
  untrained model is well below the ceiling and the trained one is near it"; it cannot beat the knapsack.
- **Bugs found on the instance and fixed:** zsh read `$HOST:serve.sh` as a variable modifier, so scp copied to a local
  file (now `${HOST}`, and the copy is checked); vLLM's FlashInfer sampler needed `ninja` on the PATH (now on PATH,
  and that sampler is off: calls are greedy); vLLM 0.33's tool grammar failed every Ministral request (tool calling
  off); and PURSE's hub factory built its LLM without the endpoints (fixed, with a test).

## 11. Pilot changes after the first Qwen run (2026-11-02, the user's choice)

Qwen3.5 9B thinks at temperature 0 for so long that 159 of its first 302 attempts hit calibration's 10-minute deadline
and were recorded as "failed: out of time" at $0: a measure of the clock and of 28 streams sharing the GPU, not of the
model. Those rows were dropped. The user chose: keep thinking, cap the open models' output at 8,192 tokens, and drop
Qwen3.5 27B (it would think for hours). So:
- `bench/prices.json`: `max_output_tokens` 8192 for the open models; an attempt that thinks past it fails and is
  charged. `calibrate --deadline 3600`, so the cap, not the clock, ends an attempt.
- Each model runs as 40 shards over all its jobs, sharing one table (`t-<model>.jsonl`).
- Every row obeys the 8k cap: Qwen3.5 9B is rerun in full; 19 Ministral rows that wrote more than 8,192 tokens (all
  wrong answers) are rerun; the per-slice tables made under the 32k cap are kept in `bench/out/pilot/archive-32k-cap/`.
- The arithmetic checker no longer crashes on a runaway answer of tens of thousands of digits (it compared ints;
  it now compares canonical digit strings). All 908 stored arithmetic verdicts are unchanged.
- The ladder is Ministral 3 3B < Qwen3.5 9B < Gemma 4 31B, by list price.
- Gemma 4 31B is served with FP8 weights (vLLM `--quantization fp8`, quantized at load). In bf16 its 59 GiB of
  weights left 10 GiB of KV cache on the 80 GB H100, room for about 12k tokens of context, and PURSE's longest input
  is about 84k tokens. Ministral 3 3B and Qwen3.5 9B are served in bf16. All three are priced at their list prices.

## 12. Results (2026-11-03; 1x H100 SXM, about 5.5 hours, about $24 of credits)

**Pilot: STOP** (`bench/out/pilot/report.txt`). The hidden bins do move success a lot on these models (Qwen3.5 9B on
level-3 arithmetic: 100% / 0% / 0% across low / mid / high), but knowing the bin barely changes what to buy: V5 0.006
in sample, -0.020 out of fold. Gemma 4 31B is both the strongest model and cheap per attempt, while Qwen3.5 9B's thinking
makes it dearer in practice (294 of its 432 attempts ran to the 8k cap, all billed), so "Gemma for nearly everything"
wins whatever the bin. Price does not track quality on this ladder; V1, V2, V3, V5 and V6 fail. Ministral 3 3B: 37 of
432 right. 7 of Gemma's 432 attempts are missing (LiteLLM's 600 s request timeout, correctly left out).

**Trained splitter** (`bench/out/splitter/report.txt`; round 2's 20 projects, three replay draws, reply paid at
Qwen3.5 9B's list price):

| | τ 0.25 | τ 0.5 | τ 1 | τ 2 |
|---|---|---|---|---|
| Knapsack (free) | 0.715 | 0.788 | 0.842 | 0.902 |
| Perfect splitter (target, reply paid) | 0.687 | 0.766 | 0.833 | 0.900 |
| **Trained Qwen3.5 9B** | **0.663** | **0.750** | **0.820** | **0.879** |
| Untrained, thinking off | 0.513 | 0.578 | 0.660 | 0.661 |
| Untrained, thinking on | 0.000 | 0.000 | 0.000 | 0.000 |
| gpt-6-luna split (stored) | | 0.695 | | |

- Training adds +0.15 to +0.22 over the same model untrained (paired 95% intervals exclude 0 at every τ) and agrees
  with the knapsack on 92% of jobs (untrained: 26%). It lands within 0.02 of the perfect splitter.
- It stays below the free knapsack by 0.023 to 0.052 (intervals exclude 0), as the plan predicted: it pays for its own
  reply (8.2% of the budget on average) and copies the solver.
- Against the gpt-6-luna split at τ 0.5: +0.056 [-0.031, +0.131], not significant.
- With thinking on, the untrained model never finished within 8,192 tokens at temperature 0 (80 of 80 cut off), so it
  answered nothing and spent 38.5% of the budget.
- Caveat: targets set aside the reply's cost from a chars/4 estimate (about 1,600 tokens); real prompts are about 3,200
  tokens, so the "perfect splitter" row is slightly optimistic.
- Training: LoRA r=16, one epoch, 188 steps, about 32 minutes; train loss 0.137 to about 0.02, eval loss 0.040 at step
  50. The adapter is kept locally in `bench/out/splitter/adapter` (gitignored); the merged model was not kept.
- To rerun: vllm 0.33.0, torch 2.14.0 (CUDA 13.0), transformers 5.20.2, peft 0.21.2, flash-linear-attention 0.6.0
  on Ubuntu 22.04 and driver 580.105.08 (`bench/out/splitter/requirements.txt` and `machine.txt`). Hugging Face
  revisions: Qwen/Qwen3.5-9B c2022362, google/gemma-4-31B-it 842da379, mistralai/Ministral-3-3B-Instruct-2512
  b35d4dfe. The local adapter and training data match the instance's copies byte for byte (sha256 checked).
- Published (public, the user's choice) at https://huggingface.co/guineapig/purse-splitter-qwen3.5-9b: merged model
  at the root, adapter in `adapter/`, card from `bench/lambda/MODEL_CARD.md`. The adapter loaded from the Hub onto
  the base gave the same replies as the evaluation on 3 of 3 prompts. The card links to `bench/purse/splitter.py` and
  `bench/lambda/` on GitHub's main branch, so those links work only once this code is merged.

**What it means.** A 9B model trained on the solver's splits nearly matches the solver and beats the same model
untrained by a wide margin, which answers ZEBRA's LLM-direct finding in the trained direction. It is not a reason to
use an LLM instead of the solver: the solver is free and better. A trained allocator only has something the solver
lacks on a table where reading a job's text pays, and neither round 2 nor this open-weight pilot is that table.
