# Prior art: who allocates the budget in multi-agent LLM systems

A stress test of the novelty claim for the HASHIRU v2 whitepaper, run on 2026-11-01. It covers papers, preprints,
open-source projects, frameworks and products through November 2026. No LLM APIs were called.

**Claim tested.** "In every budget-aware multi-agent system, something other than the agents decides how the money is
split (an optimizer, a learned controller, a solver, a broker, a host manifest, or the designer). In HASHIRU the LLM
agents themselves hold the money and allocate it to agents they hire, recursively, during the task."

**Read-depth key.**

- **full**: full text read.
- **code**: source and README read.
- **docs**: documentation read.
- **summary**: secondary source only.
- **abstract only**: abstract or landing page only.

Rows marked **(S)** were read in full by a search sub-agent and were not re-read for this file. Every other row, and
every quote in Section 3, was checked against the source.

## 1. Verdict

**Broken.** Both halves of the claim fail. The first sentence ("something other than the agents decides how the money
is split") is false in 2026. HASHIRU's mechanism is also not new: LLM agents that hand an enforced money budget to
sub-agents they spawn during a task, recursively, already exist. All of these predate November 2026:

| Group | Source | What the agent does | Date |
|---|---|---|---|
| Recursive, money, enforced, amount chosen by the LLM | **Conway Research Automaton** (open source, about 7.6k stars) | A parent agent spawns child agents and moves real credits to them, with `fund_child(amount_cents)` or, in its goal mode, the planner's per-task cost estimate. Children run the full runtime, so they can do the same. Unused credits are recalled when the goal completes | Feb 2026 |
| | **wayland-core** `delegate` tool (open-source Rust CLI) | The parent LLM gives each sub-agent a `max_cost_usd` (plus token, time and depth caps). The cap is clamped to the parent's remaining envelope and charged to "the subagent and its descendants". Child spend rolls up to every ancestor | Field added Aug 2026 |
| | **Syrin** `RLMLoop` | In `BudgetSplit.MANUAL` mode the LLM calls `spawn_agent(..., budget=0.15)` in USD. Recursion goes up to depth 6. An amount above the parent's remaining budget is rejected. The developer must opt in to this mode | May 2026 |
| | **Pactra** (preprint plus code) | An MCP tool `pactra_spawn(label, budgetUsdc)` registers a child spending mandate, enforced on-chain against every ancestor. Agents were scripted in the evaluation | Sep 2026 |
| | **AuthorityOS** (hackathon demo) | An LLM Manager gives specialists shares of a USDC budget, and specialists re-delegate shares to sub-agents. Enforced on-chain. Fixed at two levels | Jun 2026 |
| One level, or a non-dollar unit | **AgentLance** (arXiv 2608.23867) | The LLM manager splits its own dollar payment into enforced per-subtask caps. A market allocator, not the manager, picks the workers. The only one of these evaluated on benchmarks with real LLMs | Aug 2026 |
| | **OSA** | The parent LLM sets a per-sub-agent `max_budget_usd` ceiling | 2026 |
| | **Devin** "managed Devins" | The coordinating Devin launches child sessions "with specific prompts, skills, tags, and ACU limits" | 2026 |

**What survives (the caveat).** Four things remain distinctive. Each was checked against the closest systems above.

1. **No one has evaluated agent-held budgets as an allocator in a recursive tree.**
   - None of the recursive systems reports a task evaluation. Pactra's agents are "scripted and deterministic".
   - AgentLance evaluates manager-set budgets, but one level deep and with a broker choosing the workers.
   - ZEBRA tests an LLM that splits dollars across fixed phases ("LLM-direct") and finds that a knapsack solver beats
     it on every aggregate metric.
   - So no source measures how LLM agents divide money they hold across a hiring tree, or compares that with an
     optimizer-allocated split at equal dollars. This is RQ2 of the whitepaper plan, and it is the strongest claim
     left.
2. **Reserved pools with refunds.**
   - In HASHIRU a hire's pool is reserved out of the hirer's, so siblings cannot spend each other's money, and unspent
     money returns on firing (`src/hashiru/ledger.py`, `left()`).
   - wayland-core, Syrin, OSA, Pactra and AuthorityOS use ceilings instead: caps that are checked or rolled up, with
     nothing reserved.
   - Automaton is the exception. It transfers credits and recalls them, so it has both properties. But in its goal
     mode the amount is code-computed as max(LLM estimate, configured default), and a heal step auto-funds depleted
     agents.
3. **Two resources in one hiring tree.** HASHIRU enforces dollars (for cloud models and priced remote agents) and also
   admits local models against measured free memory. No other source does both. Agents allocate only the dollars,
   though; memory is a hub-measured constraint, not something agents hand out.
4. **Hires outside the process.** A hire can be a remote A2A agent priced through an Agent Card extension, or a whole
   HASHIRU hub. AuthorityOS, Virtuals ACP and Masumi let agents pay remote agents, but none gives a remote hire a pool
   carved from the payer's.

**Three cautions for the paper.**

- **Depth.** HASHIRU's default `max_depth` is 2 levels below the CEO (`src/hashiru/conf/config.yaml`). That equals
  AuthorityOS's fixed depth and is shallower than Syrin's ceiling of 6. Claim recursion "to a configurable depth".
- **Do not claim to be first to let agents allocate money.** Cite Automaton, wayland-core, Syrin, Pactra, AuthorityOS
  and AgentLance as the nearest prior art. A reviewer who knows any one of them would reject the original wording.
- **Counter-evidence and counter-trend.** ZEBRA's LLM-direct result is direct evidence against LLM allocators. Several
  frameworks also moved budgets away from the LLM:
  - OpenHands deprecated its LLM-set `max_turns`.
  - Strands deprecated its LLM-parameterised swarm tool in favour of "developer-controlled safety limits".
  - Meerkat removed its per-spawn budget.
  - A popular Claude Code skill says to set payment policy "never as an agent-callable tool".

  The paper should address why agents should allocate at all. The optimizer-allocated baseline in RQ2 answers that.

## 2. Candidates examined

**Verdict key.**

- **match**: an LLM agent sets an enforced money or token budget for agents or subtasks it creates or hires, during
  the task.
- **partial (a)**: an optimizer, controller, solver or broker outside the agents makes the split.
- **partial (b)**: agents spawn sub-agents, but with no budget.
- **partial (c)**: payment rails, or soft budgets that nothing enforces.
- **partial (d)**: a host, owner, operator or designer sets the budget.
- **no**: not about allocation among agents.

### 2.1 Matches and close partials

| Name | Source | Who allocates | When | Unit | Recursive | Verdict | Reason | Read |
|---|---|---|---|---|---|---|---|---|
| Conway Research Automaton | github.com/Conway-Research/automaton | The LLM, through `fund_child(child_id, amount_cents)`. In goal mode, the planner LLM's `estimatedCostCents` per task (code takes the max with a default). Unused credits are recalled on completion | During | Credits (USD cents), transferred | Yes: each child clones and runs the full automaton | **Match** | Closest overall (transfer plus recall). Real-money "sovereign" agents with no task evaluation. Heal step auto-funds depleted agents. Transfers above half the balance are blocked | code |
| wayland-core `delegate` tool | github.com/FerroxLabs/wayland-core, `crates/wcore-tools/src/delegate.rs` | The parent LLM, through the tool's `budget` argument. Code clamps it to the parent's remaining envelope | During | USD, input/output tokens, wall time, tool time, processes, depth | Yes (`max_agent_depth`); spend rolls up to ancestors | **Match** | Nested ceilings, not reserved pools; no refund step. Field first appears in v0.12.26-rc.1 (2026-08-05) | code |
| Syrin `RLMLoop`, `BudgetSplit.MANUAL` | github.com/syrin-labs/syrin-harness | The LLM, through `spawn_agent(budget=...)`, if the developer opts in to MANUAL (the default, FULL, hands the child the parent's whole remainder) | During | USD | Yes, depth up to 6 | **Match** (opt-in) | Rejects amounts above the parent's remaining budget; no reservation or refund shown | code, docs |
| Pactra | Zenodo 22833162; github.com/SomeshTalligeriDEV/Pactra | The agent, through MCP `pactra_spawn(budgetUsdc)`. The owner signs the root | During | USDC (testnet) | Yes, up to `maxDepth` | **Match** on mechanism | Ancestor-checked ceilings, not pools; evaluation uses scripted agents | code, docs; preprint (S) |
| AuthorityOS | github.com/git-guuud/AuthorityOS | The LLM Manager sets each specialist's share; a specialist's LLM sets a sub-agent's share of its own budget | During | USDC (testnet) | 2 levels; sub-agents cannot spawn | **Match** (depth-capped demo) | Hackathon demo with no evaluation; money moves from the user's account at spend time | code |
| AgentLance, "Markets, Not Planners" | arXiv 2608.23867 | The LLM manager splits its own payment into per-subtask caps. A market allocator and VCG pricing choose the workers and set their pay | During | Dollars (simulated from model cost) | No: one level | **Match** on the split; partial overall | Evaluated with real LLMs, but the manager does not choose whom to hire, and workers cannot subcontract | full |
| OSA (Optimal System Agent) | github.com/Miosa-osa/OSA, `tools/builtins/delegate/tool.ex` | The parent LLM, through `max_budget_usd` per sub-agent | During | USD, turns, model tier | Team nesting to depth 3 (S) | **Match-ish** | Each child's cap is independent; nothing is carved from a parent pool | code; nesting (S) |
| Devin, managed Devins | docs.devin.ai/work-with-devin/advanced-capabilities | The coordinating Devin sets each child session's ACU limit | During | ACU (billed compute units) | Not documented | **Match-ish** | A cap per session; the session sleeps at the limit (S); no pool | docs |
| Meerkat `agent_spawn` v0.4.9 | docs.rs/crate/meerkat-tools/0.4.9 | The parent LLM sets `budget.max_tokens`, capped by config | During | Tokens | Yes (`allow_nested_spawn`) | **Match-ish**, tokens only | Later removed | code (S) |
| EpochX | arXiv 2603.27304 | The lead solver (human or agent) sets subtask bounties at or below the parent's; the platform locks credits | During | Platform credits (bounties) | Yes, by design | Partial | Subtasks go back to an open market, so the delegator does not choose the hire; no case shows an LLM splitting a bounty | full |
| Sovereign-OS | arXiv 2603.14011 | The LLM CEO writes `estimated_token_budget` per task; a code CFO approves it against the Charter; an auction picks the worker | During | Tokens, USD | No | Partial (a)/(d) | An estimate gated by a code treasury; the hire is chosen by auction | full |
| TheBotCompany | arXiv 2603.25928 (ASE '26) | The LLM manager Athena sets a cycle budget per milestone for the permanent manager Ares; the orchestrator enforces it; the user sets the dollar limit | During | Cycles | No | Partial | Cycles, not money; the recipient is a fixed role, and hired workers get no budget | full |
| Goose `delegate` | github.com/aaif-goose/goose, `summon.rs` | The parent LLM sets `max_turns`, model and provider per delegate | During | Turns | No: "Delegated tasks cannot spawn further delegations" | Partial | An enforced limit set by the LLM, but not money and not recursive | code (S) |
| Claude Managed Agents | platform.claude.com/docs/en/managed-agents/budgets | The developer sets one dollar budget per session, shared by all threads; the agent sets the lifetime of workflow runs it starts | Before / during | USD (developer); time (agent) | Runs do not nest | Partial (d) | "there are no per-thread caps" | docs (S) |
| ZEBRA, LLM-direct baseline | arXiv 2605.20485 | A dedicated allocator LLM splits dollars across fixed pipeline phases; ZEBRA itself uses a knapsack solver | Before execution | Dollars | No | Partial (a) | A separate allocator over designer-fixed phases, and it loses to the solver | full |
| Agent Contracts | arXiv 2601.08815 (COINE@AAMAS 2026) | Designer code or a coordinator (proportional, equal or negotiated) | Before execution | Tokens, calls, USD, time | Envisioned only | Partial (d) | Budgets are constants in the experiment; agents drafting budgets is future work | full |
| Agent libOS | arXiv 2606.03895 | A host-authored Task Authority Manifest that derived manifests may only narrow; charges roll up the ancestor chain | Before execution | Calls, tokens, bytes, CPU (not money) | Yes (attenuation) | Partial (d) | The host sets the ceiling; the counters are not "monetary spend" | full |
| Concordat | arXiv 2610.37461 | A library that splits an allowance on handoff; the paper does not say who sets it | During | Tokens, tool calls, money | Through handoffs | Partial (d) | Enforced linear split, but not shown to come from the LLM and not evaluated | full |
| SatGate MCP proxy | github.com/SatGate-io/satgate | Harness code calls the `satgate/delegate` JSON-RPC method, which carves credits from the parent macaroon | During | Credits | Yes, with depth caveats | Partial (d) | The mechanism carves from the parent, but it is not exposed as an LLM tool | docs (S) |
| Virtuals ACP GAME plugin 0.1.x | PyPI `acp-plugin-gamesdk` 0.1.1 | The buyer LLM filled `price` in `initiate_job`, held in on-chain escrow | During | Crypto | Sellers can also buy | Partial (c) | A fee to an independent agent, not a pool; later versions take the price from the seller | code (S) |
| Intelligent AI Delegation | arXiv 2602.11865 (Google DeepMind) | A human or AI delegator issues budget mandates | Conceptual | Money, resources | Yes, conceptually | Partial (conceptual) | A framework with no system and no enforcement | full (S) |
| Haven-AI sub-budgets | github.com/d-hinders/Haven-AI (PR #3444) | The owner governs each issuance; the agent's key only signs | During | ERC-7710 token budgets | One hop | Partial (d) | Agent A passes a narrower budget to B, but the owner decides it | code (S) |
| OpenClaw / Molty21 bot funds a child | news.bitcoin.com, cryptoast.fr (Feb 2026) | An agent paid a child's server and API credits from its own Lightning wallet | During | BTC (Lightning) | Not stated | Partial (anecdote) | Who chose the amount, and any enforcement, are not reported | summary (S) |

### 2.2 Frameworks, SDKs and products where the developer or host sets the budget

| Name | Source | Who sets the budget | Unit | LLM-set per-child limit? | Verdict | Read |
|---|---|---|---|---|---|---|
| Claude Agent SDK | code.claude.com/docs/en/agent-sdk/subagents | Developer, `max_budget_usd`, shared by the whole tree | USD | No (sub-agents get `model`, `effort`, `maxTurns` from their definitions) | Partial (d) | docs (S) |
| Claude Code Agent tool | code.claude.com/docs/en/sub-agents | Agent author sets `maxTurns` | Turns | The LLM may pass `model` and `effort`, but no budget | Partial (b) | docs (S) |
| Anthropic task budgets | platform.claude.com/docs/en/build-with-claude/task-budgets | Developer, per request; "a soft hint, not a hard cap" | Tokens | No | Partial (c)/(d) | docs (S) |
| Anthropic multi-agent research system | anthropic.com/engineering (Jun 2025) | Scaling rules embedded in prompts | Sub-agents, tool calls | Soft only | Partial (c) | docs (S) |
| OpenAI Agents SDK | openai.github.io/openai-agents-python | Developer `max_turns`; handoffs carry no budget | Turns | No | Partial (b)/(d) | docs (S) |
| OpenAI Codex | github.com/openai/codex, `multi_agents_spec.rs` | Config or backend; one token budget shared across the tree | Tokens | The LLM may set `model` and `reasoning_effort` only | Partial | code (S) |
| Google ADK | google.github.io/adk-docs | Developer `RunConfig.max_llm_calls` | LLM calls | No | Partial (d) | docs (S) |
| LangGraph, LangChain deepagents | GitHub | Developer `recursion_limit`; the task tool takes a description and type | Steps | No | Partial (b) | code (S) |
| CrewAI hierarchical | docs.crewai.com | Developer `max_iter`, `max_rpm`, `max_execution_time` | Iterations, time | No | Partial (b)/(d) | docs (S) |
| AutoGen / AG2, Magentic-One / UI, Microsoft Agent Framework | GitHub | Developer `max_turns`, `max_stalls`, `max_round_count` | Turns | No | Partial (d) | docs (S) |
| Pydantic AI | ai.pydantic.dev | Developer `UsageLimits`; delegates share `ctx.usage` | Tokens, requests | No | Partial (d) | docs (S) |
| OpenHands | software-agent-sdk `task/definition.py`, `manager.py` | The agent definition, or inherited from the parent; LLM-set `max_turns` deprecated | Iterations, USD | No (removed) | Partial (d) | code (S) |
| AutoGPT Platform AutoPilot | GitHub `agent-collaboration-architecture.md` (Nov 2026) | Code computes one tree ceiling from the user's plan; the caller picks model and effort | Credits, head-count | No | Partial (a)/(d) | docs (S) |
| MetaGPT | GitHub | Developer `team.invest()`; `NoMoneyException` at the cap | USD | No | Partial (d) | code (S) |
| Gemini CLI | GitHub | Agent definition or settings (`maxTurns`, `maxTimeMinutes`) | Turns, time | No | Partial (d) | docs (S) |
| Strands `swarm` tool | github.com/strands-agents/tools | Originally the LLM set `max_handoffs`, `max_iterations` and timeouts. Deprecated in favour of "developer-controlled safety limits" | Handoffs, time | Was yes; now developer | Partial | code (S) |
| OpenClaw `sessions_spawn` | GitHub | The LLM sets `model`, `thinking` and `runTimeoutSeconds` | Time | Time only | Partial | code (S) |
| Swarms AutoSwarmBuilder | github.com/kyegomez/swarms | The boss LLM emits `max_loops` and `max_tokens` per agent spec, once | Loops, tokens per response | One-shot, not recursive | Partial | code (S) |
| smolagents, Letta, Agno Team, CAMEL Workforce / OWL, Agent Zero `call_subordinate`, Kilo / Roo orchestrator, Cursor, Kimi K2.5 swarm, Manus | GitHub, docs | None per child, or developer `max_steps` | Steps | No | Partial (b) | docs (S) |
| Gas Town (Yegge) | github.com/gastownhall/gastown | None per agent; the hosted Kilo version takes a user-set token budget | Tokens | No | Partial (b)/(d) | summary |
| Small projects with rule or config budgets (OrganAIze, Roku, Instructor-PHP, gptme, mc-code, litany, Warlock.js, oasis, TanStack AI, Bernstein, ven-AI, Vectis) | GitHub | Code rules, config or constants | Various | No | Partial (d) | code (S) |
| LiteLLM, Portkey, Lightning Labs agent tools, Docker agent | docs | Admin or developer | USD, sats | No | Partial (d) | docs (S) |
| SWE-agent, BabyAGI, SuperAGI, legacy AutoGPT | GitHub | User-set single-agent limits | USD, steps | No | No | docs (S) |

### 2.3 Optimizers, controllers, solvers and brokers (partial a or d)

| Name | Source | Who allocates | When | Unit | Recursive | Verdict | Reason | Read |
|---|---|---|---|---|---|---|---|---|
| BAMAS | arXiv 2511.21572 (AAAI 2026) | An ILP solver picks the LLMs; an RL policy picks the topology | Before execution | Dollars (rescaled) | No | Partial (a) | Solver and policy outside the agents | full |
| MasRouter | arXiv 2502.11133 (ACL 2025) | A learned cascaded controller picks mode, roles and an LLM per agent | Per query, before execution | Cost penalty, no budget | No | Partial (a) | Learned router | full |
| ATOM | arXiv 2605.26178 | A complexity predictor sets the "active electron budget"; an RL policy picks agents | Per query, before execution | Number of agents (proxy for tokens) | No | Partial (a) | A controller sets the cap | full |
| ZEBRA | arXiv 2605.20485 | LLM curve estimates plus a water-filling knapsack solver | Before execution | Dollars | No | Partial (a) | A solver splits across fixed phases | full |
| Mixture-of-Models (NSED) | arXiv 2601.16863 | A Dynamic Expertise Broker (knapsack); agents only spend quadratic-voting budgets | Per session | Team and depth; votes | No | Partial (a) | The broker was fixed to preset profiles in the experiments | full |
| MALBO | arXiv 2511.11788 | Bayesian optimization of which LLM fills which role | Offline | Dollars, as an objective | No | Partial (a) | Design-time optimizer, no budget | full |
| SC-MAS | arXiv 2601.09434 | One learned controller picks roles, edges and LLMs | Per query, before execution | Cost penalty | No | Partial (a) | Learned controller | full |
| AgentBalance | arXiv 2512.11426 | Backbone-then-topology design under token-cost and latency budgets | Design time | Tokens, latency | No | Partial (a) | Optimizer | abstract only |
| AgentCARD | arXiv 2606.20629 | The designer fixes role assignments and deployment per experiment | Fixed | Dollars per task | No | Partial (d) | A benchmark, not an allocator | full |
| Self-Resource Allocation | arXiv 2504.02051 | An LLM planner or orchestrator assigns tasks to agents with known costs | During | Assignment cost; no budgets | No | Partial | The LLM allocates work, not money | abstract only |
| Puppeteer | arXiv 2505.19591 | RL-trained orchestrator with a step penalty | During | Steps | No | Partial (a) | Learned policy; no budget per agent | full (S) |
| MaAS | arXiv 2502.04180 | A controller samples from an agentic supernet | Per query | Cost-aware | No | Partial (a) | Learned controller | full (S) |
| Agent Exchange (AEX) | arXiv 2507.03904 | Auction platform; the user sets the budget | Per task | Money | No | Partial (a) | Broker | full (S) |
| CoRL | arXiv 2511.02755 | RL-trained controller | Per query | Cost | No | Partial (a) | Learned controller | full (S) |
| AgentTTS | arXiv 2508.00890 | Offline search over per-subtask compute | Offline | Compute | No | Partial (a) | Search outside the agents | full (S) |
| DOVA | arXiv 2603.13327 | A selection function over six thinking tiers | Per query | Thinking tiers | No | Partial (a) | Function outside the agents | full (S) |
| Learning Self-Orchestration in Decentralized Agent Swarms | arXiv 2610.22322 | Learned bids decide who acts | During | No money moves | No | Partial (a) | No budgets | full (S) |
| Theoriq | theoriq.ai | "Optimizer" agents form collectives | During | Tokens | Not documented | Partial (a) | Optimizer role | docs (S) |
| Bounded Agents (APC) | arXiv 2608.15888 | Infrastructure, "non-negotiable by the agent" | Session start | Six ceilings | Chain | Partial (d) | Set by the host | full (S) |
| RGAO / Code-Agent | arXiv 2605.05657 | The designer presets three budget tiers; a rule-based router picks the topology | Before execution | Six-dimension vector | Yes | Partial (d) | Designer | full (S) |
| Token Budgets (affine types) | arXiv 2606.04056 | Operator code calls `budget.split` | During | Tokens, USD | Yes | Partial (d) | Enforced split, but written by the programmer | full (S) |
| Fault-Tolerant Budget Conservation | arXiv 2610.00349 | Allocation policy is left to the operator | During | Budget vectors | Yes | Partial (d) | Enforcement layer only | full (S) |
| LDP delegation contracts | arXiv 2603.18043 | The delegator attaches a budget to the contract | During | Tokens, USD | Depth field | Partial (c) | "policy expression, not platform enforcement" | full (S) |
| Spawn Freely, Act Sparingly (PRV) | arXiv 2609.01035 | A code governor escrows and debits a risk budget | During | Risk | Yes | No | Risk, not money; theory and synthetic data | abstract only |
| Spore.fun case study | arXiv 2506.04236 | A fixed designer rule triggers funded reproduction | During | Crypto | Yes (5 generations) | Partial (d) | "reproduction is entirely algorithmic" | full (S) |
| HASHIRU v1 | arXiv 2506.04255 | The CEO hires under one global cost and memory budget | During | Dollars, VRAM share | No: "employee agents cannot spawn additional sub-agents" | Partial | One level; no per-hire pools | full |

### 2.4 Agents that spawn sub-agents with no budget, or with designer-fixed caps (partial b or d)

| Name | Source | Budget per child | Read |
|---|---|---|---|
| Recursive Language Models | arXiv 2512.24601 | None | full (S) |
| THREAD | arXiv 2405.17402 | None | full (S) |
| ReDel | arXiv 2408.02248 | None | full (S) |
| Self-Organized Agents | arXiv 2404.02183 | None | full (S) |
| MegaAgent | arXiv 2408.09955 | None | full (S) |
| Recursive Agent Harnesses | arXiv 2606.13643 | None | full (S) |
| WebSwarm | arXiv 2607.08662 | None | full (S) |
| AgentSys | arXiv 2602.07398 | None | full (S) |
| AgentSpawn | arXiv 2602.07072 | None | full (S) |
| Kimi K2.5 Agent Swarm (PARL) | arXiv 2602.02276 | None | full (S) |
| AOrchestra | arXiv 2602.03786 | The orchestrator picks each sub-agent's model from a price table; the budget is only a prompt reminder (c) | full (S) |
| AsyncThink | arXiv 2510.26658 | Fixed per-worker token cap (d) | full (S) |
| Adaptive Parallel Reasoning | arXiv 2504.15466 | Fixed context windows (d) | full (S) |
| Context-Folding | arXiv 2510.11967 | 32K budget, at most 10 branches (d) | full (S) |
| SearchSwarm | arXiv 2606.09730 | 50-turn sub-agent limit (d) | full (S) |
| Conductor | arXiv 2512.04388 | Fixed token caps (d) | full (S) |
| Minions | arXiv 2502.15964 (ICML 2025) | The remote LM "writes code that decomposes" a task into local jobs; no budget (b) | full |
| PAPILLON | arXiv 2410.17127 (NAACL 2025) | Fixed local-remote pipeline; no budget | full |

### 2.5 One agent managing its own budget (no allocation to other agents)

| Name | Source | Who sets the budget | Unit | Verdict | Read |
|---|---|---|---|---|---|
| BATS / Budget Tracker | arXiv 2511.17006 | The user or experimenter; the agent only sees what remains | Tool calls | No: single agent | full |
| INTENT | arXiv 2602.11541 | The user; a world model plans tool use | Dollars per tool call | No: single agent | full |
| BAGEN | arXiv 2606.00198 | Not an allocator; measures whether agents can estimate their remaining spend | Tokens, cost, time | No | full |
| Plan-and-Budget | arXiv 2505.16122 | The LLM estimates complexity; a hand-set decay formula assigns tokens to sub-questions | Tokens | Partial (a), single agent | full (S) |
| TALE | arXiv 2412.18547 | An estimated budget placed in the prompt | Tokens | Partial (c), single agent | full (S) |
| SelfBudgeter | arXiv 2505.11274 | The model predicts its own budget; RL rewards compliance | Tokens | Partial (c), single agent | full (S) |
| BudgetMLAgent | arXiv 2411.07464 | Cascade limits fixed by the designer | Calls, dollars | Partial (d) | full (S) |
| "Do More Agents Help?" | arXiv 2606.05670 | Step limits fixed by the experimenters | Tokens, steps | No: an evaluation study | full |
| MAST | arXiv 2503.13657 (NeurIPS 2025) | Not about budgets | none | No: a failure taxonomy | full |

### 2.6 Agent economies, marketplaces and payment rails (partial c or d unless noted)

| Name | Source | Who allocates the money | Recursive | Verdict | Read |
|---|---|---|---|---|---|
| Virtual Agent Economies | arXiv 2509.10147 | Market mechanisms and auctions (position paper) | No | Partial (conceptual) | full |
| Magentic Marketplace | arXiv 2510.25779 | Customer agents pay business agents for goods | No | Partial (c) | full |
| NLSOM / Economy of Minds | arXiv 2305.17066 | Members "may offer other EOM members money in exchange for certain services" | No | Partial (conceptual) | full (S) |
| Masumi | developers.cardano.org | Agents pay agents for services, and those agents pay others | Yes (purchases) | Partial (c): buying services, not funding pools | summary (S) |
| Virtuals ACP (Butler) | 4pillars research; virtuals.io | The Butler hires agents with the user's money, through escrow | No | Partial (c)/(d) | summary (S) |
| MetaMask x Venice hackathon projects (DelegAI, Brain Budget, Briefcase, DeleGate) | hackathon pages | A chief agent re-delegates ERC-7710 slices to fixed roles. Where documented, the amounts come from the user or from code | One hop | Partial (d) | summary (S) |
| everything-claude-code `agent-payment-x402` skill | GitHub skill file | Policy is set in the orchestration layer, "never as an agent-callable tool" | No | Partial (d), a counter-norm | docs (S) |
| Google AP2 | ap2-protocol.org | Users issue intent mandates to agents; the spec has no sub-agent mandates | No | Partial (d) | docs (S) |
| Coinbase AgentKit, x402, a2a-x402 | GitHub, docs | Limits set by the operator or user | No | Partial (c)/(d) | docs (S) |
| Nevermined | nevermined.io docs | Human card delegations; admin-set group budgets | No | Partial (d) | docs (S) |
| Kite, Skyfire KYAPay, Payman, Crossmint, Cloudflare agent wallets, Amazon Bedrock AgentCore payments, FiGuard | vendor docs | Caps set by the user, owner or developer | No | Partial (d) | docs (S) |
| Fetch.ai uAgents payments | docs | A buyer/seller payment protocol | No | Partial (c) | docs (S) |
| Olas Mech marketplace | docs | Agents pay per request; operators fund the agents | No | Partial (c)/(d) | docs (S) |
| Coral Protocol | docs | The consumer funds the session escrow | No | Partial (d) | docs (S) |
| ERC-8004 | EIP | Identity and reputation only | n/a | No | docs (S) |
| ElizaOS wallet plugins | GitHub | One wallet per agent | No | Partial (c) | docs (S) |
| Anthropic Project Vend phase 2 | anthropic.com | The experimenters created the agents; the CEO agent approves decisions but allocates no budgets | No | Partial | summary (S) |
| Andon Labs Vending-Bench Arena | andonlabs.com | Peer agents send each other money; no sub-agents | No | Partial (c) | summary (S) |
| AI Village | theaidigest.org | No money split between agents | No | No | summary (S) |
| ClawWork, iLands | GitHub | Single-agent balances | No | No | summary (S) |
| CEO-Bench, CFO-Bench | benchmarks | The LLM splits simulated capital across business units, not agents | No | No | summary (S) |
| The Coasean Singularity | NBER essay | Theory | n/a | No | summary (S) |
| Self-Sovereign Agent | arXiv 2604.08551 | Wallet models for agents that copy themselves (position paper) | n/a | No | full (S) |
| Comprehension-Gated Agent Economy | arXiv 2603.15639 | A designer-defined robustness gate caps economic permissions | No | Partial (d) | full (S) |

**Not found or not checked in depth:** Morpheus, Hyperspace, Talus, MIT NANDA, Stripe/OpenAI ACP, Catena ACK, ChatDev
(no per-agent budgets according to a sub-agent's scan), The AI Scientist and Agent Laboratory.

## 3. Evidence for matches and close partials

### Conway Research Automaton

Sources: https://github.com/Conway-Research/automaton; `src/agent/tools.ts`, `src/agent/system-prompt.ts`,
`src/orchestration/orchestrator.ts` and `src/replication/spawn.ts`.

> "A successful automaton replicates. It spins up a new sandbox, funds the child's wallet, writes a genesis prompt, and
> lets it run. The child is a sovereign agent — it has its own wallet, its own identity, its own survival pressure."

> `fund_child`: "Transfer credits to a child automaton. Requires wallet_verified status." (parameter `amount_cents`:
> "Amount in cents to transfer")

> "SPAWN: Create child with role, fund with estimated task cost, send assignment"

> "COMPLETE: Goal achieved. Recall unused credits from agents. Reset to IDLE."

- `fundAgentForTask` funds `Math.max(estimated, configuredDefault)` cents, where `estimated` is the planner's
  `estimatedCostCents`.
- `spawn.ts` runs `git clone https://github.com/Conway-Research/automaton.git` on the child's sandbox and builds it
  there, so each child has the same tools.
- The orchestration mode was added on 2026-02-24 ("feat: Plan A orchestration + memory subsystem").

### wayland-core `delegate` tool

Sources: https://github.com/FerroxLabs/wayland-core; `crates/wcore-tools/src/delegate.rs` and
`crates/wcore-budget/src/execution.rs`.

> "Sub-allocate a SMALLER slice of your own remaining execution envelope to each subagent, so one runaway subagent
> cannot consume the whole session and starve its siblings. Every field can only lower a cap: a value larger than what
> you yourself hold is clamped down to your own limit, never granted."

> `"max_cost_usd": { "type": "number", "description": "USD charged to the subagent and its descendants." }`

> `"max_agent_depth": { "type": "integer", "description": "How deep the subagent may itself delegate." }`

> "Runtime view onto a budget. Cheap to clone; tree-shaped — counters recorded on a child also roll up to all
> ancestors."

The `budget` field is absent at commit 23a98927 (2026-07-04) and present at 9150ff1f (2026-08-05).

### Syrin `RLMLoop`

Sources: https://github.com/syrin-labs/syrin-harness; `docs/advanced/rlm-loop.md`, `src/syrin/rlm/_loop.py` and
`src/syrin/rlm/_budget.py`.

> "# MANUAL — LLM specifies explicit budget per spawn in the tool call"
> "# LLM calls: spawn_agent(task="...", agent="DataAgent", budget=0.15)"

> "Budget for this child in USD (MANUAL split only). Optional."

> `raise ValueError(f"manual_amount {manual_amount} exceeds parent_budget {parent_budget}")`

The default is `BudgetSplit.FULL`: "By default (`BudgetSplit.FULL`) each spawned child receives the parent's full
remaining budget."

### Pactra

Sources: https://github.com/SomeshTalligeriDEV/Pactra, `packages/mcp/SKILL.md`; Zenodo record 22833162, "Pactra:
Hierarchical On-Chain Budget Enforcement for Delegated Autonomous Agent Systems", 2026-09-18.

> "`pactra_spawn(label, budgetUsdc, trancheUsdc?, concentrationPct?)` — Register a child mandate under this one for a
> sub-agent. The child can only ever be narrower than its parent; the contract refuses a wider one whoever asks."

The repo description says "Every accepted draw debits the requesting node and every ancestor up to the root". The
evaluation README says "The agents are **scripted and deterministic**".

### AuthorityOS

Sources: https://github.com/git-guuud/AuthorityOS; `orchestrator/src/adapters/reasoning/core.ts` and
`orchestrator/src/engine/engine.ts`.

> "Fraction (0–1) of the mission budget delegated to this specialist. Shares should sum to ≤ 1."

> "Fraction (0–1) of this agent's budget to redelegate to the sub-agent."

> "Children don't spawn further (bounds the tree); only nodes do."

The README adds: "USDC moves from the user's account, bounded on-chain by every budget cap in the chain, and the agents
themselves never hold a cent."

### AgentLance

Source: https://arxiv.org/html/2608.23867.

> "If it chooses subcontracting, it becomes a manager, decomposes the original task into subtasks, assigns a budget to
> each subtask, and opens a separate subcontract market. Labor agents then bid on these subtasks and are selected using
> the same allocation and VCG-style payment rules as in the primary market."

> "If the sum of the proposed subtask budgets exceeds the primary-market payment, the decomposition is rejected and the
> winner executes the task independently."

Algorithm 1 keeps only worker bids with b_j(s_k) ≤ B_k. Subcontracting was enabled only for GAIA.

### OSA

Source: https://github.com/Miosa-osa/OSA, `lib/optimal_system_agent/tools/builtins/delegate/tool.ex`.

> "Per-subagent USD spend ceiling. The subagent stops itself once it crosses this cap. Omit for no cap. Use on wide
> fan-outs to bound total spend."

### Devin, managed Devins

Source: https://docs.devin.ai/work-with-devin/advanced-capabilities.

> "Spin up managed Devins — launch child sessions with specific prompts, skills, tags, and ACU limits"

### EpochX

Source: https://arxiv.org/html/2603.27304.

> "Once a task is claimed, the lead solver p_c may complete it directly or decompose it into subtasks and delegate them
> to other participants. In this case, Credits function not only as a reward for final delivery, but also as a budget
> that can be reallocated during execution."

### Sovereign-OS

Source: https://arxiv.org/html/2603.14011.

> "produces a TaskPlan: a DAG of PlannedTask items, each with a task_id, description, dependency list, required_skill
> mapped to a Charter competency, estimated_token_budget, and priority (high/low). An LLM generates the plan as
> structured JSON"

> "This separation of concerns ensures that no task can consume resources without explicit budgetary authorization,
> even if the LLM planner requests it."

### TheBotCompany

Source: https://arxiv.org/html/2603.25928.

> "Milestone directives (Athena) specify the milestone title, description, and cycle budget."

> "Budget management enforces a user-configured 24-hour rolling spending limit"

### ZEBRA, LLM-direct baseline

Source: https://arxiv.org/html/2605.20485.

> "the controller, given the same pipeline description and total budget, outputs a per-phase split. This baseline asks
> whether an LLM can self-allocate budget via internal reasoning instead of an explicit solver"

> "ZEBRA solves a continuous nonlinear knapsack via water-filling to allocate the global budget across phases before
> execution."

### Agent Contracts

Source: https://arxiv.org/html/2601.08815.

> "Initial Allocation. Before execution, the total budget B must be divided among agents."

> "Agents spawn sub-agents that themselves have contracting capability, enabling hierarchical self-organization"

The second quote is listed as an implication of the framework; it is not implemented in the experiments.

### Agent libOS

Source: https://arxiv.org/html/2606.03895.

> "An explicit Host-authored Task Authority Manifest establishes a durable transition_ceiling; every manifest derived
> from it preserves or narrows that ceiling"

> "These counters therefore do not represent exact physical-request counts, provider billing, or monetary spend."

### Concordat

Source: https://arxiv.org/html/2610.37461.

> "Handing work to another agent splits the budget: the child receives an allowance that is removed from the parent,
> so the same allowance cannot be spent twice."

## 4. Number checks

| Figure as cited | Status | What the paper says |
|---|---|---|
| BAMAS "up to 86% lower cost" | **Confirmed** | "BAMAS achieves comparable accuracy while reducing costs by up to 86%" (arXiv 2511.21572). |
| AgentCARD "up to 44% more accurate than cost-equivalent homogeneous teams; matches the strongest homogeneous team at up to 12x lower cost" | **Confirmed, with two qualifiers** | Abstract: "improve accuracy by up to 44% over cost-equivalent homogeneous teams, or match the strongest homogeneous team at up to 12× lower per-task cost through hybrid deployment." See the qualifiers below the table. |
| Minions "5.7x cost reduction at 97.9% of remote accuracy" | **Confirmed** | "MinionS reduces costs by 5.7× on average while recovering 97.9% of the performance of the remote model alone" (arXiv 2502.15964). The figure is for MinionS, the extended protocol, with an 8B local model. Plain Minion is 30.4× at 87%. |
| PAPILLON "85.5% quality, 7.5% leakage" | **Confirmed** | "Our best pipeline maintains high response quality for 85.5% of user queries while restricting privacy leakage to only 7.5%." (arXiv 2410.17127) |
| MAST "14 failure modes" | **Confirmed** | "This process identifies 14 unique modes, clustered into 3 categories" (arXiv 2503.13657). |

AgentCARD qualifiers:

- **The 44 is percentage points against a weak baseline.** It is +44.0 on SWE-bench (GPT-5.4 planner, MiniMax-M2.7
  executor), measured against a straight line between two homogeneous teams. Against the stricter baseline, the best
  homogeneous team at matched cost, the same cell is +8.0 (Table 11).
- **The 12× is the top of a range.** The paper gives "1.4×–12×", and the upper end needs hybrid deployment.

Two rows in `plan/2026-11-01-whitepaper-v2.md` §2.3 disagree with the papers:

- **ZEBRA** is listed as "During the task, per phase". Its main method allocates once, "before execution".
  Mid-pipeline re-allocation is only an Appendix F extension, and "The result is negative".
- **Mixture-of-Models** is listed as "a broker picks a team and deliberation depth per session". In the experiments,
  "the Dynamic Expertise Broker was configured with fixed, pre-determined agent profiles rather than solving the
  Knapsack optimization (Eq. 2) at runtime for every prompt."

## 5. The departure, worded to fit the evidence

> LLM agents that give sub-agents they spawn an enforced money budget, even recursively, already exist in 2026 agent
> software (Conway's Automaton, wayland-core, Syrin, Pactra, AuthorityOS), and AgentLance evaluates a manager that
> splits its own payment one level deep in a brokered market. HASHIRU's departure is to treat this as an allocation
> mechanism to be measured: its LLM agents hire, fund and fire agents from dollar pools reserved out of their own,
> recursively and during the task, and no prior work measures how agents divide money held this way or compares it
> with an optimizer-allocated split at equal dollars.

Notes on the wording:

- **The second sentence needs RQ2.** It holds only if the whitepaper reports that comparison. Without it, the claim
  shrinks to the pool semantics (reserved pools with refunds, where wayland-core, Syrin, OSA, Pactra and AuthorityOS
  use ceilings), and Automaton already has transfer plus recall.
- **Say "dollar pools".** Agents allocate only dollars. Memory is a second, hub-measured constraint (a local model
  must fit in measured free memory), not something agents hand to their hires.
- **Do not say "unbounded" recursion.** Say "recursively" or "to a configurable depth". The default `max_depth` is 2.
