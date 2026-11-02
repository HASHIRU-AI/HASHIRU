# Plan: the HASHIRU v2 whitepaper

Status: the framing the user first confirmed ("agents hold the money") was broken by the prior-art check the same day;
the user confirmed the measurement-led revision (§3) on 2026-11-01. Not security enforcement, in either version. W1 to
W3 and W6 to W8 are taken as recommended until the user says otherwise. W9 fixed on `bugfix/unpriced-models`.
Phase B done: `plan/whitepaper/prior-art.md`, `refs.bib` (46 entries, all verified), `refs-verification.md`.
Date: 2026-11-01

## 0. Decisions

| # | Question | Recommendation | Why |
|---|---|---|---|
| W1 | What the whitepaper is | **A systems paper with its own evaluation: an arXiv report first, then cut down for a venue** | v1's evidence was one baseline (Gemini 2.0 Flash alone), with p > 0.05 on 5 of the 9 tasks that had a baseline, and no baseline for the other 2. The 2026 papers nearest to v2 (BAMAS, ZEBRA, AgentCARD) all report numbers |
| W2 | Venue | **ICML 2027** (deadline not announced; trackers expect Jan 22 to 28, 2027) or **COLM 2027** (tentative, about Mar 30, 2027). MLSys 2027 (Oct 30) and AAMAS 2027 (Oct 8) have passed | About 11 weeks to ICML, enough for the plan below |
| W3 | arXiv | **A new arXiv ID** that cites 2506.04255, not a replacement of it | v1's numbers describe v1's code (tag `v1.0-paper`) |
| W4 | Models and money | **Decided 2026-11-01:** CEO `gpt-6-luna`; hires from `ollama_chat/qwen3.5:4b`, `gpt-6-luna` and `meta/muse-spark-1.3`. Proposal: $50 in all for cloud runs, a pilot first, each batch asked before it runs | See section 9 for prices and setup |
| W5 | ~~Lite's R4 and R5~~ | **Dropped**: security enforcement is not a contribution of this paper | The rules stay in the design and the implementation sections |
| W6 | Phase 8d and 8e | **Do not wait.** 8e (track records) becomes future work, unless the user wants it as an allocator in RQ2 | The paper claims only what it delivers |
| W7 | Where the source and the harness live | **Decided 2026-11-01: not HASHIRUBench.** The user asked for a new, hard benchmark built to test what §3 claims; it and the paper live here (`bench/`, `paper/`) on `feature/whitepaper`. Design in progress | Keeps experiment code and data out of the package |
| W8 | Authors and template | Kunal Pai, Parth Shah, Harshil Patel (as v1). IEEE two-column for the arXiv report (as v1); the venue's template once W2 is settled | |
| W9 | The price check fails open (found 2026-11-01) | **Fix before any paid run**, tests first | LiteLLM's bundled map (pinned 1.103.4, read offline) has no GPT-6 prices, yet `get_model_info("gpt-6-luna")` returns $0 instead of raising, so `check_priced` passes and Luna would be billed as free (pools and `max_usd` would not see it). The refreshed `model_catalog.json` prices it today; without that file it is free again. A cloud model whose input and output prices are both 0 or missing should count as unpriced |

## 1. Requirements restatement

- A whitepaper on HASHIRU v2 (the rebuild on A2A and MCP, merged into `main` on 2026-10-30) that states what is
  unique about it, shows that the design works, and measures what it buys.
- The contribution is not security enforcement. Privacy, plan mode and approvals appear as design, not as claims.
- It replaces v1's story, not v1's numbers (v2 plan, decision 4).
- Every claim is backed by the repo or the harness; every citation is checked against its venue of record.

## 2. What the reading turned up

### 2.1 v1 (arXiv 2506.04255)

- One version, 2025-06-01, cs.MA, "Submitted as part of the Research Track at AgentX 2025, organized by Berkeley
  RDI." No journal reference.
- Four contributions: a hierarchy with resource-aware hiring and firing; local-first hybrid intelligence;
  autonomous tool creation; an economic model with hiring and invocation fees.
- Evaluation (Table I, 11 tasks of 40 to 127 samples, one baseline): significant gains on GSM8K (96 vs 61),
  JEEBench (80 vs 68.3), SVAMP (92 vs 84) and MMLU Math (91.8 vs 87.2); p > 0.05 on ARC, HLE, StrategyQA, MMLU Law
  (58.4 vs 61.6, lower) and MMLU Psychology; paper review (58%) and JailbreakBench (100% of 50 prompts) with no
  baseline.
- Its own limitations (Sec. VII), and what v2 did about each:

| v1 limitation | v2 |
|---|---|
| One level of hierarchy; employees cannot hire | Managers hire with money they were given, up to a depth cap (2 by default) |
| The economic model needs calibration | v1's resource units and hand-written cost table are replaced by LiteLLM prices and memory measured from Ollama |
| The local-versus-API trade-off was not analyzed | RQ1 and RQ2 below |
| Tool creation needs alignment | Each created tool is its own `uv` process behind the hub's gate and trace; off by default; still not a sandbox |
| Memory needs work for long histories | SQLite FTS5 (bm25) store shared by sessions |

- Errata not to repeat: MMLU Math baseline 87.2 in the table and 87.7 in the text; SVAMP listed twice; ARC cited to
  a classification paper.

### 2.2 HASHIRU-lite (UCLA CS239, Fall 2026)

- The v2 core came from lite (A2A agents, MCP tools, pools, traces). Its rules R1 to R5 and its finding that rules in
  prompts get broken now go in the design and implementation sections, not the contributions.

### 2.3 Who decides how the money is split: the field since v1

Sorted by **who allocates the budget, and when.** The prior-art check (2026-11-01, `plan/whitepaper/prior-art.md`, about
130 candidates) found LLM-funded sub-agents already in agent software (the row before HASHIRU's), so the mechanism alone
does not separate HASHIRU; what nobody has done is measure it (§3).

| Line of work | Who allocates | When | What is priced | Examples |
|---|---|---|---|---|
| Budget-aware MAS design | An optimizer or a learned controller | Before the task, or per query | Model choice, tokens | BAMAS (Yang et al., AAAI 2026 ✓: an ILP picks LLMs, RL picks the topology; up to 86% cheaper), MasRouter (Yue et al., ACL 2025 ✓), ATOM (2605.26178: a difficulty predictor sets a collaboration budget), MALBO (2511.11788: Bayesian optimization of team composition), SC-MAS (2601.09434) |
| Up-front allocation by a solver or broker | A central solver or broker | Before execution (ZEBRA re-allocates mid-run only in an appendix extension, which did not help; Mixture-of-Models ran fixed preset team profiles) | Phases of a fixed pipeline | ZEBRA (Hamri and Talgam-Cohen, 2605.20485: a Lagrangian water-filling split per phase; its LLM-direct baseline, an LLM splitting the dollars, loses to a knapsack solver), Mixture-of-Models (2601.16863) |
| Hierarchical budgets in a runtime | The host program's manifest | Fixed before the task; children attenuate | Calls, bytes, CPU, memory (not money) | Agent libOS (Zhang, 2606.03895: charges roll up the ancestor chain; "do not represent ... provider billing, or monetary spend"; one fixed model) |
| Mixed deployments | The designer | Fixed per experiment | API and self-hosted cost per role | AgentCARD (Jiang et al., 2606.20629: up to 44 points over a line between homogeneous teams, +8.0 over the best homogeneous team at equal cost (its Table 11); hybrid deployment matches the best homogeneous team at 1.4x to 12x lower cost) |
| Local and cloud pairs | Fixed protocol | Fixed | Cloud dollars; local is free | Minions (Narayan et al., ICML 2025 ✓), PAPILLON (Li et al., NAACL 2025 ✓) |
| Budget awareness of one agent | The agent, over its own tool calls | During the task | Tool calls, tokens | BATS (Liu et al., COLM 2026 ✓), INTENT (2602.11541), BAGEN (2606.00198: strong agents are not necessarily budget-aware) |
| Agent economies | Market mechanisms | Simulated | Goods between buyer and seller agents | Virtual Agent Economies (Tomašev et al., 2509.10147), Magentic Marketplace (Microsoft, 2510.25779) |
| Payments | Not an allocator | Per call | Settlement | AP2, the A2A x402 extension |
| LLM-funded sub-agents | The LLM, per spawn | During the task | USD ceilings or credit transfers | Conway's Automaton (`fund_child(child_id, amount_cents)` transfers credits; children run the same runtime), wayland-core (`delegate`: the LLM may "sub-allocate a SMALLER slice of your own remaining execution envelope", `max_cost_usd` charged "to the subagent and its descendants"; ceilings, no refunds), Syrin, Pactra, AuthorityOS; AgentLance (2608.23867: a manager LLM sets enforced per-subtask budgets within its payment, one level in the experiments, a market allocator picks the workers). None evaluates how the agents divide the money. Checked in source and full text 2026-11-01 |
| **HASHIRU** | **The agents, to a configurable depth (2 by default)** | **During the task** | **Reserved pools with refunds, in dollars and measured memory; hires include priced A2A agents and whole hubs** | v1 (June 2025) predates the row above for an LLM CEO hiring and firing agents under cost and memory budgets, one level deep |

Supporting lines, cited briefly: MAST (Cemri et al., NeurIPS 2025 ✓: multi-agent gains over one agent are often small,
so a single-agent baseline is mandatory); "Do More Agents Help?" (2606.05670: accuracy and cost in multi-agent
workflows); the protocol survey (Ehtesham et al., 2505.02279); A2A under the Linux Foundation; MCP's Tasks extension
(SEP-2663); for the privacy section only, PAPILLON, CaMeL (2503.18813), FIDES (2505.23643), AgentLeak (2602.11510).
For the framing, Coase's theory of the firm (1937: hierarchy versus market, make or buy), to verify. From memory,
to find and verify: RouteLLM, FrugalGPT, Hybrid LLM, AutoGen, Magentic-One, CrewAI, LangGraph, GAIA. Also local:
`../hashiru_v2_poc` (July 2026), a router that extrapolates capability from a benchmark taxonomy; a possible
allocator baseline.

## 3. Framing (revised 2026-11-01 after the prior-art check; confirmed by the user the same day)

**Superseded.** The framing the user confirmed earlier today ("in HASHIRU the agents hold the money; everyone else
splits it outside the agents") does not hold. Agent software in 2026 already lets an LLM fund sub-agents it spawns,
with the money enforced and in some cases recursively: Conway's Automaton (`fund_child(child_id, amount_cents)`;
children run the same runtime), wayland-core (`delegate` with `max_cost_usd` charged "to the subagent and its
descendants", clamped to the parent's envelope), Syrin, Pactra and AuthorityOS. AgentLance (2608.23867) evaluates a
manager LLM that sets enforced per-subtask budgets, one level deep, with a market allocator choosing the workers. The
first two were checked in their source (`src/agent/tools.ts`; `crates/wcore-tools/src/delegate.rs`) and AgentLance in
its full text. HASHIRU v1 (June 2025) predates all of them, but it was one level deep.

**What survives.**
- Nobody measures how LLM agents divide money they hold across a hiring tree, or compares that with an optimizer at
  equal dollars. The one data point points the other way: ZEBRA's LLM-direct baseline (an LLM splits the dollars up
  front) loses to a knapsack solver.
- Reserved pools with refunds (the others use ceilings, except Automaton's transfer and recall).
- Dollars and measured memory in one tree.
- Priced A2A agents and whole hubs as hires (make or buy across organizations).

**Thesis (measurement-led).** Agent software now lets LLMs fund the agents they spawn, but whether they spend
well is unknown. HASHIRU v2 is built so that agents must allocate: they hire, fund and fire agents from pools reserved
out of their own, in dollars and memory, across local models, cloud models and priced remote agents, and every
allocation is recorded. We use it to measure how LLM agents spend money they hold, and how that compares with an
optimizer at equal dollars.

**The hypothesis RQ2 tests.** An up-front split (ZEBRA, BAMAS) cannot react to what the task turns out to need; agents
can fire, refund, rehire and escalate while it runs. Whether that adaptivity beats a better up-front split is the
question, and a 2x2 separates the two factors: who allocates (an LLM or a solver) by when (up front or during the
task). HASHIRU is LLM-during; ZEBRA is solver-up-front; its LLM-direct baseline is LLM-up-front.

**Gap sentence.** No arrangement measures how LLM agents divide money they hold when they hire other agents during a
task: agent software lets them fund sub-agents without evaluating the allocation, and the budget-aware systems that
are evaluated split the money with an optimizer or controller outside the agents.

**Contributions (three).**
- **HASHIRU v2, an economy for agents.** Each agent holds a pool reserved from its hirer's (dollars, plus measured
  memory for a local model) and spends it by hiring, to a configurable depth: it declares how hard a job is, the hub
  picks the cheapest model that fits the job, the pool and the data's sensitivity, and firing refunds what was not
  spent. A hire can be a local model, a cloud model, a priced A2A agent or a whole hub; the ledger is exact and can be
  rebuilt from the trace.
- **How agents spend.** We measure how LLM agents allocate money they hold: reservation against use, refunds, the tier
  chosen for each job, make or buy as an outside agent's price changes, and how the team changes as the budget grows.
- **Agents against optimizers.** At equal dollars, we compare agent-allocated teams with a single agent, fixed
  homogeneous and hybrid teams, and up-front splits by a solver and by an LLM.

**Positioning.** HASHIRU v2 shares LLM-funded sub-agents with 2026 agent software (Automaton, wayland-core, Syrin,
Pactra, AuthorityOS) and manager-set budgets with AgentLance, but departs in two dimensions: (1) *Measured, not only
provided:* to our knowledge, it is the first to evaluate agent-held budgets as an allocator across a hiring tree,
against solver, LLM-up-front and fixed-team baselines at equal dollars. (2) *Reserved pools in two currencies, with
make or buy:* a hire reserves money (and memory) that firing refunds, and the market includes priced remote agents and
whole teams.

Departure (1) and the last two contributions hold only if the paper delivers RQ1 and RQ2, so the paid runs (W4) are now
essential.

**Other features, in economic terms (design sections, not contributions).** Sensitivity limits where work may go
(private work is made locally or bought with identifiers masked). Plan mode is a quote: the hub prices the plan
(models, reservations, memory) before anything is spent, and approving it approves its hires. Approvals are spending
authority: a hire or a turn above a threshold waits for the user. Pools are enforced in code, so an agent cannot spend
money it does not hold.

If the results change the story (for example, agents turn out to over-reserve badly, or the make-or-buy panel is the
strongest), the thesis is re-picked around the strongest panel before the abstract is written.

**Fixed vocabulary.** *Pool* (an agent's money), *reservation*, *spend*, *refund* (unspent money returned on firing),
*hire* and *fire*, *make* (hire a local model) and *buy* (hire a cloud model or a priced remote agent), *hub*,
*declared difficulty*, *cloud* (any non-local model and every remote A2A peer), *local* (served by the configured
Ollama).

**Math budget.** One equation, `left(a) = R_a − spent(a) − Σ_{c ∈ hires(a)} hold(c)`, because it explains refunds and
why concurrent hires cannot spend the same dollars.

## 4. Outline (arXiv report: about 12 pages plus appendix; the venue version cuts to 8 or 9)

| § | Section | Content | Figures and tables | Pages |
|---|---|---|---|---|
| 1 | Introduction | Teams of agents cost money; today something outside the agents splits it; the gap sentence; "To address these limitations"; three contributions; headline results | Fig. 1: a hiring tree with pools in two currencies, one bought A2A agent and one hired hub | 1.25 |
| 2 | Related work | The table in §2.3, as prose by who allocates; local and cloud; agent economies | | 1.25 |
| 3 | From v1 to v2 | The table in §2.1; what came from lite | Table 1 | 0.5 |
| 4 | Agents as budget holders | Roles, pools and the one equation, memory as a currency, hiring by declared difficulty, escalation, refunds, one deadline per request | Fig. 2: one request's money flow; Table 2: the worked pool example | 1.75 |
| 5 | Make or buy over open protocols | A2A agents, MCP tools, the cost extension, remote agents, hubs hiring hubs, HASHIRU as a model, an agent and a tool | Fig. 3: components and protocols; Table 3: who talks to whom | 1.25 |
| 6 | Quotes, authority and sensitivity | Plan mode as a priced quote (plans compiled to graphs), approvals, private data as a limit on where work may go (masking, local-only) | Fig. 4: a plan and its graph | 1 |
| 7 | Implementation | Python 3.13, about 6,300 lines, exact pins, 440 tests, mutation checks, model recon, the trace | | 0.5 |
| 8 | Evaluation | RQ1 to RQ4 and the case studies (§5) | Figs. 5 to 8, Tables 4 and 5 | 3 |
| 9 | Limitations | Scope boundaries: small local models call tools poorly, prices are list prices, declared difficulty is the agent's own judgement, one API key is one trust domain, `max_usd` can be exceeded by calls in flight | | 0.5 |
| 10 | Conclusion and availability | Source-available under PolyForm Noncommercial 1.0.0 (never "open source") | | 0.25 |
| A | Appendix | Config, the cost extension, the trace format, prompts, case-study traces, reproducibility (models, prices on the run date, Ollama version, hardware, seeds, `uv.lock` hash) | | |

## 5. Evaluation

Results open with "Overall, our experiments show the following results:" and bold-led findings. Each configuration
runs 3 times; a difference smaller than the spread is reported as within seed variance, with both numbers. Money is
compared at an equal cap (the same `budget_usd`), reporting what each configuration actually spent. Every allocation
(reservation, spend, refund, model, tier, escalation, peer price) is read from the trace.

**RQ1, how agents spend.** Given money, how do LLM agents allocate it?
- Sweep the CEO's pool ($0, $0.05, $0.25, $1) on the RQ2 tasks.
- Measure: reservation against spend (slack), refunds, team size and depth, make versus buy (local against cloud
  hires), declared difficulty against the tier that succeeded, escalations, and how each changes with the pool.
- Allocator ablation: CEO `gpt-6-luna` against a stronger CEO (`meta/muse-spark-1.3`, or `gpt-6.1-sol` at $2/$10).
  Does a stronger allocator spend better, and is it worth its own cost?
- Cost: shared with RQ2's runs.

**RQ2, does agent allocation pay off?** At an equal cap, how does it compare with other ways to split the money?
- Configurations: one agent on one cloud model (MAST's lesson); a fixed homogeneous team; a fixed hybrid team
  (AgentCARD's planner and executor split); a solver's up-front split (ZEBRA- or BAMAS-style, chosen by an offline
  sweep); an LLM's up-front split (the CEO divides the money before any hire and cannot change it, ZEBRA's LLM-direct
  baseline); HASHIRU with agents allocating during the task by declared difficulty; and HASHIRU with the CEO naming
  models (v1's way). The solver and LLM up-front splits against HASHIRU form the 2x2 in §3.
- Tasks: a new benchmark built for this question (the user's call, 2026-11-01, replacing the GAIA proposal), designed
  in `plan/2026-11-01-allocation-benchmark.md` (working name PURSE): procedural jobs with exact answers, a knob for how
  much difficulty is hidden, an oracle per budget, and validity checks run before any allocator. That design supersedes
  the configurations and metrics listed here for RQ1 to RQ3 where they differ.
- Metrics: success, dollars spent, memory used, wall-clock, hires per task; a cost against success Pareto plot.

**RQ3, make or buy.** When priced outside agents are available, do agents buy, and does buying pay?
- Offer A2A peers through the cost extension (a specialist agent such as the example calculator, and a second hub)
  at several declared prices, from cheaper than making to dearer.
- Measure the share of jobs bought at each price (a demand curve), success when bought and when made, and the money
  saved or lost.
- Cost: the peers run locally or on `gpt-6-luna`; small.

**RQ4, overhead.** What do the protocol boundaries cost in time? One A2A hop, a gateway MCP call, a hired hub against
an in-process manager, and an approved plan run as a graph against the CEO carrying it out. Fake models, $0.

**Case studies** (existing real runs, no new spend): a private note with identifiers masked for the cloud CEO, tool
creation on `gemma4:12b` fixing its own code from the errors, a hub buying from a priced A2A peer, memory recalled
across two processes, and the debate plan graph (`traces/chat-20261030-081949-*.jsonl`).

## 6. Phases

### Phase A: Lock the framing (1 day)
- [ ] The user settles W1 to W3 and W6 to W8.
- [x] Confirm or revise §3 (what is unique, thesis, gap, contributions, vocabulary): the measurement-led revision,
  confirmed 2026-11-01.
- [ ] Choose RQ2's task suites and the optimizer-allocated baseline; set up the paper directory, template and `refs.bib`.

### Phase B: Related work and bibliography (2 days)
- [ ] Verify every entry with Semantic Scholar, CrossRef, DBLP, OpenReview or the ACL Anthology, and cite the venue of
  record. Anything unverifiable is marked `[CITATION NEEDED]` and listed.
- [ ] Read the full text of BAMAS, ZEBRA, ATOM, MALBO, Mixture-of-Models, Agent libOS, AgentCARD, BAGEN, Minions and
  PAPILLON, check each number cited above against the paper, and confirm that none lets agents allocate money to
  agents they hire. One that does changes §3 before anything else is written.
- [ ] Write §2 by who allocates, ending in the gap sentence.

### Phase C: System sections (3 days)
- [ ] §3 to §7 from the README, `docs/architecture.md` and the plan files.
- [ ] Figures 1 to 4 as vector graphics (redraw `docs/hiring-tree.svg` with both currencies and a bought agent);
  captions that stand alone; colorblind-safe.
- [ ] Count tests and mutation checks exactly from the plans and git history.

### Phase D: Harness and fixes, no real models (3 days)
- [x] W9: the price check counts a cloud model with no price as unpriced; tests first (2026-11-01, on
  `bugfix/unpriced-models`): `has_price` needs a positive token price, and the error names `hashiru models --refresh`.
  The new test failed first; two mutations of the check are caught; 441 tests pass, ruff and mypy clean. With the
  real `.env`, the hub starts with the refreshed map and refuses `gpt-6-luna` without it.
- [ ] Pilot check: no cloud call has run in v2 yet, so the first one must show a nonzero `cost_usd` in the trace (LiteLLM
  prices the response's model name, which may be a dated snapshot).
- [ ] An allocation extractor over traces (reservations, spends, refunds, tiers, make or buy, escalations, peer
  prices) and a ledger cross-check with `ledger.replay`.
- [ ] RQ2's baselines (single agent, fixed teams, the optimizer split) as configurations of the same hub where
  possible; RQ3's priced peers; RQ4's microbenchmarks; loaders for the task suites; per-run config, price and
  environment records; tests for the harness with fake models.

### Phase E: Real runs (4 to 6 days, each batch approved first)
- [ ] E1, local only ($0): RQ3's local peers, RQ4's local timings; Ollama running with `qwen3.5:4b`.
- [ ] E2, cloud pilot: 5 tasks per configuration and pool, then a cost estimate for the user.
- [ ] E3, the full runs within the approved cap. Traces, result JSON, the price map's date and environment records are
  kept with the paper.

### Phase F: Results and the front matter (2 to 3 days)
- [ ] §8 in the results format above; re-pick the thesis from the strongest panel.
- [ ] Abstract (Mode 1), introduction and limitations last. No em dashes.

### Phase G: Review and release (2 days)
- [ ] A multi-perspective review of the full draft, a second citation check, and a voice audit.
- [ ] Compile; the user submits to arXiv; then `CITATION.cff` and the README cite the new paper next to v1.

## 7. Risks

| Level | Risk | Mitigation |
|---|---|---|
| HIGH | ~~A paper found in Phase B already lets agents allocate money to agents they hire~~ Happened (2026-11-01) | §3 moved the claim from the mechanism to measuring it |
| HIGH | The measurement claim depends on RQ1 and RQ2 being delivered | Paid runs are on the critical path; a pilot first; if agents lose to the solver, that is the finding, and RQ1 explains where and why |
| HIGH | Agents allocate badly (BAGEN: strong agents are not necessarily budget-aware), so agent allocation loses to a fixed or optimized split | That is still a finding (RQ1 says why); hard limits keep losses bounded; the allocator ablation shows whether a stronger CEO fixes it |
| HIGH | The evaluation needs paid runs | A pilot, a cost estimate and the user's approval before each batch; `gpt-6-luna` as CEO keeps it cheap |
| HIGH | Prices: the pinned LiteLLM prices GPT-6 models at $0 without the refreshed map (W9) | Fix the check before any paid run; record the price map's date with every result |
| MED | Small local models call tools poorly, so "make" looks weak | That is part of the trade being measured; report model sizes; use the largest local model that fits |
| MED | LLM runs vary even at temperature 0 | 3 runs per configuration; within seed variance where the spread covers the difference |
| MED | Several citations are recent arXiv preprints, and some numbers above come from abstracts or summaries | Phase B reads each paper; preprints are labeled as such |
| LOW | ICML 2027's dates are not announced | The arXiv report does not depend on them; COLM 2027 as the fallback |
| LOW | Readers take "source-available" for "open source" | Say PolyForm Noncommercial 1.0.0 in the availability statement |

## 8. Estimate

| Phase | Estimate |
|---|---|
| A Framing | 1 day |
| B Related work | 2 days |
| C System sections | 3 days |
| D Harness and fixes | 3 days |
| E Real runs | 4 to 6 days |
| F Results and front matter | 2 to 3 days |
| G Review and release | 2 days |
| **Total** | **17 to 20 working days (about 4 weeks)** |

B and C can run in parallel with D.

## 9. Models, prices and setup (2026-11-01)

Prices from LiteLLM's current map (refreshed 2026-11-01 into `model_catalog.json`), per million tokens; Meta's match
its pricing page (dev.meta.ai/docs/pricing-rate-limits).

| Model | Where | $ in | $ out | Notes |
|---|---|---|---|---|
| `gpt-6-luna` | OpenAI | 0.10 | 0.50 | The CEO; tools and reasoning; 922K context |
| `gpt-6.1-terra` | OpenAI | 1.00 | 5.00 | |
| `gpt-6.1-sol` | OpenAI | 2.00 | 10.00 | Candidate stronger allocator |
| `gpt-6-astra` | OpenAI | 10.00 | 50.00 | Above `auto_max_usd` ($30) |
| `meta/muse-spark-1.3` | Meta Model API | 1.25 | 4.25 | In `MODELS`; the router's first pick for hard jobs |
| `meta/muse-spark-1.3-contributor` | Meta Model API | 0.10 | 0.20 | Meta may train on what it is sent: never for private data |
| `meta/muse-wonder-1.0` | Meta Model API | 4.00 | 20.00 | Preview |
| `ollama_chat/qwen3.5:4b` | local | 0 | 0 | Costs about 3.5 GB of memory at a 16K context |

Setup in `.env` (gitignored, dockerignored, owner-only): `CEO_MODEL=gpt-6-luna`,
`MODELS=ollama_chat/qwen3.5:4b,gpt-6-luna,meta/muse-spark-1.3`, `OPENAI_API_KEY`, `META_API_KEY` (the Meta Model API
at `https://api.meta.ai/v1`; the old Llama API and LiteLLM's `meta_llama/` provider do not take this key).
`BUDGET_USD=0.50` and `MAX_USD=2.00` are still the example's values. Checked with sockets blocked: the settings load and
the hub's prices are right. Not yet done: Ollama was not running; `meta` is not in `recon.KEYED`, so `hashiru models`
and `MODELS=auto` ignore Meta's models (name them in `MODELS`); no key has been sent to its provider (`--check` was not
run).
