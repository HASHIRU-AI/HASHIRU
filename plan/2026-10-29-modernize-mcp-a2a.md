# Plan: HASHIRU v2 on MCP (tools) and A2A (agents)

Status: DRAFT v2, awaiting confirmation. No code changed yet.
Date: 2026-10-29

## 0. Decisions so far

| # | Question | Answer |
|---|---|---|
| 1 | Protocol SDKs directly vs an agent framework (ADK) | **SDKs directly** (`mcp`, `a2a-sdk`); HASHIRU owns the orchestration loop |
| 2 | What employees look like | **Managers or workers**, following [HASHIRU-lite](https://github.com/kunpai/com-sci-239-multi-agent-class-project) |
| 3 | Tool-creation sandbox | **`uv` subprocess now**, optional container backend later |
| 4 | Paper continuity | **This is v2.** No need to reproduce v1 HASHIRUBench numbers; v1 stays reproducible at a tag |

## 1. Requirements restatement

HASHIRU v2 is a CEO → manager → worker hierarchy where:

- **Every agent is an A2A service** with its own Agent Card. Agents talk only over A2A.
- **Every tool is an MCP tool.** Managers (and the CEO) run a tool loop over MCP; workers answer once with no tools.
- **Budgets follow the hiring tree.** Each agent has a dollar pool reserved from its hirer. Local agents also hold real memory (MB of the loaded model). Firing returns unspent dollars and frees memory.
- **v1 features that HASHIRU-lite left out come back:** autonomous tool creation, persistent memory with retrieval, the Gradio UI, the `Mode` toggles, multi-session use.
- **Engineering baseline:** Python 3.13, uv, typed package, pytest, ruff, mypy, CI, no secrets in source.

## 2. Starting point: HASHIRU-lite

HASHIRU-lite (`~1,100` lines, 90 tests, same author) already implements the v2 core. Reading it:

| Lite piece | What it does | Keep / change for v2 |
|---|---|---|
| `server.py` | One Starlette app: MCP tools at `/mcp`; each hired agent mounted as its own A2A app at `/agents/{name}/` with an Agent Card | Keep design. Split into `a2a_host` and `mcp_server` modules; scope by session |
| `agent.py` | `run_agent`: manager/CEO = tool loop (max 12 turns, parallel tool calls via `asyncio.gather`); worker = one LLM call. `ask_agent` is a runtime built-in A2A client, not an MCP tool, so the trace shows the real sender. `caller` is injected by the runtime and hidden from the LLM | Keep. Add A2A streaming for UI progress; make turn cap configurable |
| `ledger.py` | Agent registry; `left()` computes each pool as reservation − own spend − children's holdings; `replay()` rebuilds accounts from a trace | Keep math. Replace module-level dict with a per-session object |
| `llm.py` | All calls through LiteLLM (`acompletion`), real cost via `completion_cost`, `MAX_USD` wallet kill-switch, fail-fast on unpriced cloud models, `prices.json` overrides | Keep. Exact-pinned LiteLLM (lite notes malicious LiteLLM releases reached PyPI in early 2026) |
| `tracelog.py` | One JSONL trace per run (`start`, `llm`, `tool`, `msg`, `label`, `error`); label propagation (`private`, `untrusted`) | Keep tracing as core. Labels move with the policy layer (Phase 6) |
| `policy.py`, `check.py` | R1–R5 as past-time checks over traces; online `--enforce` monitor at the chokepoint | Optional Phase 6, after lite's own Phase 2 lands (demo 2026-12-03) |
| Tier | `ollama_chat/…` or `ollama/…` = local, anything else = cloud | Keep |
| Depth cap | Max 2 levels below the CEO, enforced in code | Keep, make configurable |
| SDK pins | Python 3.13, `mcp==2.2.0`, `a2a-sdk[http-server]==1.2.2`, `litellm==1.103.4` | Start from these; re-check advisories at port time |

What lite deliberately left out and v2 must add: `ToolCreator`, Gradio, memory auto-injection, multiple sessions, persistent memory.

Gaps in lite that matter once HASHIRU is hosted:
- **Global state.** `ledger.AGENTS` and `tracelog.events` are module globals: one run per process.
- **Unauthenticated endpoints.** `/mcp` and `/agents/*` are open, and `caller` is trusted. Safe on `127.0.0.1`; unsafe on a public Hugging Face Space, where anyone could call an employee or pose as another caller and spend its budget.

## 3. Target architecture

```
   Gradio UI · external A2A clients · MCP hosts (Claude Code, Cursor, ...)
                │ (public, authenticated)
                ▼
   ┌───────────────────────── Session (one per user/chat) ─────────────────────────┐
   │  CEO (manager role, depth 0, pool = session budget)                            │
   │   ├─ Manager (depth 1, tool loop over MCP, may hire)                           │
   │   │    └─ Worker (depth 2, one LLM call, no tools)                             │
   │   └─ Worker                                                                    │
   │  Ledger (pool tree: $ reservations + local MB)   Trace (JSONL chokepoint)      │
   └───────────────┬───────────────────────────────────────────┬───────────────────┘
                   │ MCP (internal)                            │ A2A (internal)
     ┌─────────────┼──────────────────┐            ┌───────────┴──────────────┐
     ▼             ▼                  ▼            ▼                          ▼
  core tools   user-tools server   3rd-party    local agent host           remote A2A agents
  (hire, fire, (uv subprocess,     MCP servers  /s/{session}/agents/{n}/   (Agent Card URL in
  check_usage, ToolCreator output, (fetch, fs,  one card per hire          config; cloud tier,
  memory, ...) hot reload)         search ...)                             $-billed only)
```

### Role and budget model (from lite)

| | CEO | Manager | Worker |
|---|---|---|---|
| Depth | 0 | 1 or 2 | 1 or 2 |
| Tools | all MCP tools + `ask_agent` | all MCP tools + `ask_agent` | none |
| Can hire | yes | yes, until max depth | no |
| Loop | tool loop | tool loop | one LLM call |
| Pool | session budget | reserved from hirer | reserved from hirer |

- `hire_agent(name, role, model, prompt, budget_usd)` moves `budget_usd` from the hirer's pool; a local model is preloaded in Ollama and its size counts against free memory.
- `fire_agent(name)` fires the subtree, unmounts the A2A apps, unloads unused local models, and returns unspent dollars to the hirer.
- v1's abstract "resource units" (RAM+VRAM ÷ 16 × 100) and the hand-written cost table are replaced by measured MB and LiteLLM prices.

### v1 → v2 mapping

| v1 | v2 |
|---|---|
| `GeminiManager` (Gemini-only CEO loop, `repr`/`eval` history) | `run_agent` tool loop via LiteLLM; any provider as CEO |
| `AgentCreator` / `AskAgent` / `FireAgent` / `GetAgents` | `hire_agent` (MCP) / `ask_agent` (A2A built-in) / `fire_agent` (MCP) / Agent Card registry |
| `AgentCostManager` dict | `MODELS` config + LiteLLM price map + `prices.json` |
| `BudgetManager` singleton (two global budgets) | Per-session pool tree (`ledger`) + `MAX_USD` wallet cap |
| `ToolManager` (`importlib`, `pip.main`) | MCP client over core, user-tools and third-party servers |
| `ToolCreator` (in-process exec) | Writes a FastMCP module to the user-tools server (uv subprocess, hot reload, `tools/list_changed`, rollback) |
| `MemoryManager` + per-query SentenceTransformer | Memory MCP tool, persistent store, cached embedder, optional auto-injection |
| `Mode` enum | Config flags (Hydra group `modes`) checked at the chokepoint |
| `deleteAgents.py`, `ollama.create` per agent | Not needed: system prompt sent per call; models loaded/unloaded with `keep_alive` |

## 4. Package layout (target)

```
src/hashiru/
  config.py          frozen dataclass configs + Hydra compose API
  llm.py             LiteLLM wrapper, wallet cap, price check        (from lite llm.py)
  ledger.py          pool tree, reservations, memory accounting      (from lite ledger.py)
  trace.py           JSONL events                                    (from lite tracelog.py)
  session.py         per-session registry, ledger, trace, mount prefix, auth token
  runtime.py         run_agent, ask_agent, call_tool chokepoint      (from lite agent.py)
  a2a_host.py        mount/unmount agents, Agent Cards, executor
  tools/core.py      hire_agent, fire_agent, check_usage, deliver_to_user
  tools/memory.py    persistent memory + retrieval
  tools/web.py       get_website (+ search via 3rd-party MCP)
  toolsmith/         ToolCreator, user-tools server, loader, rollback
  ui/                Gradio app + trace panel
  interop/           CEO as public A2A agent; HASHIRU as MCP server
conf/                models, budget, modes, mcp_servers, a2a_peers
tests/               ported lite tests + new ones
```

v1 code (`src/manager`, `src/tools`, `app.py`, `start.py`, `deleteAgents.py`) is deleted on the v2 branch once Phase 5 replaces the UI. v1 lives on at tag `v1.0-paper`.

## 5. Implementation phases

Branching: `develop` as the v2 integration branch off `main`; one `feature/*` branch per phase, merged `--no-ff`; `develop` → `main` at release.

### Phase 0: Preserve v1, stop the leak (½ day) — DONE 2026-10-29

- [x] Remove the hardcoded OpenWeatherMap key (in `weather_tool.py`, `system.prompt` and `system_.prompt`); read `OPENWEATHER_API_KEY` instead. Hotfix merged to `main`. History not rewritten (decided: revoking the key makes it worthless). **Key revocation is the owner's action.**
- [x] Tag `v1.0-paper` on `de2b514` (exact paper state); the tag message lists the known v1 bugs.
- [x] Create `develop`. `main`, `develop` and the tag pushed.

### Phase 1: Port the lite core (3–4 days) — DONE 2026-10-29 on `feature/v2-core`

- [x] `pyproject.toml` + `uv.lock`, Python 3.13, lite's pins + `hydra-core==1.3.8` (1.3.2 had two advisories; neither reached our code paths); `ruff`, `mypy`, `pytest`; GitHub Actions CI.
- [x] Port lite into `src/hashiru/`: `llm`, `ledger`, `trace`, `runtime` (was `agent`), `tools` + `hub` + `ollama` (was `server`), typed dataclasses instead of dicts.
- [x] **Sessions:** a `Hub` hosts many `Session`s; agents mount under `/s/{session}/agents/{name}/`; each session has its own CEO, pool tree, memory copy and trace file.
- [x] **Auth:** the user, the CEO and each manager get a bearer token (workers never call anything, so they get none). `AuthGate` admits `/mcp` for any issued token and an agent's A2A routes only for its hirer's token in the same session. MCP tools read the caller from `ctx.headers`, so lite's hidden `caller` argument is gone. Closing a session revokes its tokens.
- [x] Config via Hydra compose API (`src/hashiru/conf/config.yaml`; env vars as defaults, `key=value` overrides on the CLI).
- [x] Port lite's tests (fake LLM/Ollama, no keys) plus new config, auth, hub and session-isolation tests (102 passing after the review fixes below). The smoke test serves the real hub and checks HTTP 401/403 against it.
- [x] CLI: `hashiru ask "..."` and `hashiru chat` (from lite's `run.py`).

Departures from the plan above: the Hydra config lives inside the package (`src/hashiru/conf/`) so installed wheels carry it; lite's label propagation came along now (recorded, not enforced) because it is small and tested, and Phase 6 only adds the checks; the CLI quiets two known-benign SDK log messages (OpenTelemetry context detach, A2A dispatcher).

Exit check: a real `hashiru ask` on local `qwen3.5:4b` (no keys) ran check_usage → hire → ask → fire → deliver → reply in 1 min 45 s, $0 spent.

Code review (independent reviewer, probes re-run after the fixes), all fixed with regression tests (102 passing):
- HIGH: session ids could collide and merge two users' sessions → 48 random bits, uniqueness check, label validation.
- HIGH: an agent could be hired as `user` and receive the human's token → `user`/`ceo` reserved.
- HIGH: fired agents kept their token and kept running → tokens revoked on fire/close, tools refuse dead callers, runs stop at their next LLM turn.
- MED: fire/close during a local model load leaked routes and tokens and broke `replay` → re-check after load, close unmounts every agent, `replay` skips unknown names, per-model load/unload lock.
- MED: `get_website` could reach loopback/private/link-local addresses (Ollama, cloud metadata) and read unbounded bodies → every redirect hop checked, 2 MB cap.
- LOW: phantom agents on unexpected load errors; one MCP transport error aborted a whole run (now an error observation; parallel calls in a TaskGroup); unload raced concurrent hires; gate allowed unknown paths and websockets (now deny by default); MCP sessions not bound to their creator (gate now sets `scope["user"]`).

Known limitations carried forward:
- A fired agent's in-flight LLM call still completes (one call). Cancelling running executors belongs with A2A cancel support in Phase 4, together with passing an absolute deadline down each hop instead of fixed per-hop timeouts.
- The wallet cap can be exceeded by one call per concurrently running agent (documented on `Wallet`).
- `get_website` resolves names itself and again in the HTTP client, so DNS rebinding between the two is not covered.
- Possible simplification: one `Mount("/s")` that looks agents up in the ledger, instead of a route per agent.

### Phase 2: Tools over MCP (2–3 days) — DONE 2026-10-29 on `feature/mcp-tools`

- [x] **Third-party MCP servers, hub-side.** Instead of each agent run connecting to N servers, the hub holds one connection per configured server (`mcp_servers`: a local `command` over stdio or a remote `url` over streamable HTTP) and offers its tools on `/mcp` as `<server>__<tool>` (`gateway.py`, sharing `mcplink.py` with created tools). Agents keep one tool list, and every call keeps the gate, trace and 20k result cap. A stdio server gets MCP's minimal environment plus only the `env` its config lists. Results carry the labels its config gives (`untrusted` by default). A server that fails to start is skipped; one that dies is unregistered.
- [x] **Config file:** `--config my.yaml` or `HASHIRU_CONFIG`, layered as defaults < file < command line; `examples/hashiru.yaml` shows the official fetch server plus commented filesystem and HTTP examples.
- [x] **Memory:** the `memory` tool searches, adds, deletes and lists. With `memory_db` (the CLI default) memory is one SQLite file shared by all sessions and persistent across runs, ranked with FTS5 bm25. This replaces v1's per-query SentenceTransformer, so v2 has no torch dependency. With `memory_db=''` each session gets its own copy of `memory.json` (reproducible benchmarks). `memory_autoinject` adds the best matches to the user's message (v1 behavior), records a `recall` event and labels the CEO `private`.
- [x] **Web:** `get_website` moved to `web.py`; new built-in `arxiv_search` (arXiv Atom API, public-address checks, labeled `untrusted`). General web search comes from a configured MCP server.
- [x] **Tool scoping:** `hire_agent(..., tools=[...])` limits a manager to a subset of its hirer's tools (ask_agent included); the runtime offers only those and refuses the rest.
- [ ] Removing v1's `python_sandbox_tool` and `weather_tool` happens with the rest of the v1 code in Phase 5.
- [x] Tests: 31 in `tests/test_phase2.py` (180 total), including a stdio test server and an in-process HTTP MCP server.

Code review (independent reviewer), all fixed with regression tests; three fixes mutation-checked:
- HIGH: a limited manager could hire a manager without `tools` and get every tool → an omitted `tools` inherits the hirer's.
- MED: one schema the hub could not proxy (a `true` subschema, `_x`, `model_config`) dropped the whole server and leaked its connection → tools are offered one at a time; `true` subschemas work; a failed server is fully disconnected.
- MED: third-party tools had no size limits (a 300-tool server made a 7.2 MB tool list for every manager) → created-tool limits apply, descriptions are cut to 1,000 characters, at most 64 tools per server.
- MED: shared memory dropped the `untrusted` label across sessions → each memory keeps its writer's labels; reading it (search or recall) applies them.
- LOW: CLI overrides of keys only in the config file; bad value shapes and YAML errors gave tracebacks; servers connected one after another (now in parallel, `mcp_start_timeout`); recall on messages without words; unbounded memory keys and lists; SQLite connections not closed; created tools could take `<server>__` names; arXiv XML with a DTD; only the first page of a tool list was read; `memory.json` was ignored by the persistent store (now seeds a new one). A recency-ordering bug (timestamp ties) was found by the new tests and fixed.
- [x] Real checks: `uvx mcp-server-fetch` through the gateway fetched example.com; live `arxiv_search` found the HASHIRU paper; two separate `hashiru ask` processes on local `gemma4:12b` ($0): run 1 saved "research papers → exactly three bullets" to SQLite, run 2 recalled it, hired a researcher, called `arxiv_search`, and answered in three bullets.

### Phase 3: Autonomous tool creation (3–4 days) — DONE 2026-10-29 on `feature/tool-creation`

- [x] `create_tool(name, code, dependencies)` and `delete_tool(name)` for managers, registered only when `tool_creation=true` (off by default). The code is a plain Python function named `name` with type hints and a docstring; no MCP imports needed.
- [x] Cheap checks before anything runs: name `[a-z][a-z0-9_]{0,31}`, not taken by any hub tool, code parses (`ast`, never executed in the hub) and defines the function, at most 10 PyPI-only requirements, 100 KB cap.
- [x] **One process per tool** instead of one shared user-tools server: `uv run --no-project --with mcp==<hub's> --with <deps> python -I -u child.py`. Tools cannot conflict over packages, a crash takes down one tool, and nothing needs restarting when a tool is added. The child gets MCP's minimal environment (no API keys) and a scratch working directory.
- [x] The hub holds a stdio MCP session to each child and re-exports its tool on `/mcp` through a proxy whose signature is synthesized from the child's schema, so created tools go through the same auth gate, caller check and trace chokepoint as built-in tools. Results are labeled `untrusted`.
- [x] The runtime re-reads its tool list every turn (instead of `list_changed` notifications), so a tool is callable on the turn after it is created.
- [x] Rollback on any failure (load error, missing docstring, wrong tool name, timeout, cancellation): process stopped, code file removed, error and the end of the child's stderr returned to the agent. Exceptions inside a tool come back with their message so the agent can fix its code.
- [x] Persistence: `tools_dir/<name>.py` + `manifest.json` (dependencies, sha256, creator, time). Tools restart when the hub starts; a file whose hash does not match is skipped.
- [x] Tracing: `create_tool` is a normal traced tool call; its args hold the code and its result the sha256.
- [x] Tests: 47 new (request checks, proxy schema, real `uv` processes, restart, tamper, crash, timeouts, end-to-end smoke through HTTP); 149 in total.
- [x] Real run, local `gemma4:12b`, $0: the CEO's first two attempts failed to parse, it fixed the code from the error, created `calculate_days_between_dates`, called it next turn and answered 301 (correct). Repeated after the review fixes with the same result.
- [ ] Container backend: `Toolsmith.command()` is the single place that builds the launch command.

Code review (independent reviewer, real `uv` processes), all fixed with regression tests; four fixes were also mutation-checked (tests fail with the fix undone):
- HIGH: the off switch failed open (`HASHIRU_TOOL_CREATION=no` turned it on) → boolean settings accept only true/false.
- HIGH: nothing bounded what a tool sends the hub (a 5 MB docstring reached every manager's prompt every turn; 80 MB results) → docstring ≤ 1,000 chars, schema ≤ 10 KB / 32 parameters, results cut to 20,000 chars for the agent and refused above 1 MB inside the tool.
- HIGH: one bad saved tool (bad manifest, non-UTF-8 stderr, wrong types) crashed hub startup → each saved tool loads under its own catch-all and is re-checked like a new request.
- MED: dead tool processes stayed registered → noticed on a failed call or by an idle ping and unregistered; stderr read through a pipe into a 64 KB ring buffer (was an unbounded temp file); fd 1 redirected at import; created-tool errors keep `untrusted`; delete only by the creating session (or anyone once it has ended); saved tools cannot shadow built-ins or use URL dependencies and run from a hashed copy of their bytes; the child's exact schema (enums, `$defs`) is published; explicit `None` arguments are kept; multi-block and bare-dict results are converted fully; the tool description no longer claims the child cannot see API keys.
- LOW: start timeouts waited 10 s extra and swallowed cancellation; creates could race shutdown; CRLF code was rejected on restart; hostile code could crash the parser check.

Known limitations carried forward:
- Created tools are shared by every session: their descriptions and outputs (and any state they keep) cross sessions. Fine for one trusted user; multi-user hosting needs per-session tool instances (Phase 6 policy work).
- The child is not sandboxed: it runs as the user and can read files, other processes' environments (e.g. `ps -E` on macOS) and HASHIRU's own source. A container or OS sandbox backend would fix this.
- The MCP SDK's stdio reader has no line-length limit, so a deliberately hostile child can still send one huge line before the hub's caps apply. Bounding that needs a hub-owned stdio transport.

### Phase 4: A2A beyond the local host (3–4 days)

- [ ] **A2A streaming:** agents emit `working` status updates; the UI shows them live.
- [ ] **Remote agents:** `conf/a2a_peers` lists Agent Card URLs. `hire_agent` can hire a remote agent as a worker or manager; it is always cloud tier and dollar-billed; auth follows the card's security schemes.
- [ ] **HASHIRU cost extension:** an Agent Card extension declaring pricing and reporting per-task usage in task metadata, consumed by the ledger. Our own local agents publish it too. (Candidate research contribution.)
- [ ] `Mode` toggles from v1 as config flags enforced at the chokepoint: `agent_creation`, `local_agents`, `cloud_agents`, `tool_creation`, `tool_invocation`, `memory`. Budget enforcement replaces the two budget modes.

### Phase 5: UI and interop (3 days)

- [ ] Gradio chat as a thin client of a session; a side panel shows the live agent tree, pools, and trace events (pattern from lite's `chat.html`/`viewer.html`).
- [ ] Auth0 login kept as optional; `SESSION_SECRET_KEY` required when auth is on.
- [ ] Publish the CEO as a public A2A agent (`/.well-known/agent-card.json`) and HASHIRU as an MCP server (`hashiru.ask`, budget resource) for Claude Code, Cursor, etc.
- [ ] Single-process, single-port mode for Hugging Face Spaces, with internal routes protected (Phase 1 auth).
- [ ] Delete v1 code on `develop`.

### Phase 6 (optional): Policy layer from lite (2–3 days, after 2026-12-03)

- [ ] Port lite's labels (`private`, `untrusted`), R1–R5 checks, `check.py`, and `--enforce` once lite's course Phase 2 is final.
- [ ] Generalize: policies loaded from config so users can add their own.

### Phase 7: Evaluation and docs (2 days + run time)

- [ ] New HASHIRUBench v2 runs (no comparison to v1 required), using traces for cost and behavior analysis.
- [ ] README, architecture diagram, `CITATION.cff` (v2), examples for MCP servers and remote A2A agents.

## 6. Risks

| Level | Risk | Mitigation |
|---|---|---|
| HIGH | Public deployment exposes internal MCP/A2A routes and trusts `caller` | Phase 1 session tokens; `caller` from token; loopback-only option |
| HIGH | Tool creation runs LLM-written code | uv subprocess isolates dependencies, not the host. Off by default on public deployments; container backend later |
| HIGH | LiteLLM supply chain (malicious releases in early 2026) | Exact pin + hashes in `uv.lock`; check advisories before every bump |
| MED | Small local models are weak at tool calling, so local managers fail | Lite's fallback: cloud managers with local workers; `max_turns` cap returns `failed` |
| MED | Concurrent sessions contend for one Ollama and machine memory | Ledger memory check is per machine, not per session; global lock around local hires |
| MED | MCP/A2A SDK churn | Pin; keep protocol calls in `runtime.py` and `a2a_host.py` only |
| MED | Python 3.13 on Hugging Face Spaces | Verify in Phase 1; fall back to Docker Space |
| LOW | Diverging from lite while its Phase 2 is still in progress | Port the policy layer only after 2026-12-03 |

## 7. Estimate

| Phase | Estimate |
|---|---|
| 0 Preserve v1 | ½ day |
| 1 Port lite core | 3–4 days |
| 2 MCP tools | 2–3 days |
| 3 Tool creation | 3–4 days |
| 4 A2A extras + modes | 3–4 days |
| 5 UI + interop | 3 days |
| 6 Policy layer (optional) | 2–3 days |
| 7 Eval + docs | 2 days + bench time |
| **Total** | **~3–4 weeks** |

Phases 2, 3 and 4 depend only on Phase 1 and can run in parallel.
