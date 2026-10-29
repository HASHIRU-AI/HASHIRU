# Plan: HASHIRU v2 on MCP (tools) and A2A (agents)

Status: Phases 0–5 done on `develop`; Phase 6 waits for HASHIRU-lite (after 2026-12-03); Phase 7 (docs) next.
Date: 2026-10-29

## 0. Decisions so far

| # | Question | Answer |
|---|---|---|
| 1 | Protocol SDKs directly vs an agent framework (ADK) | **SDKs directly** (`mcp`, `a2a-sdk`); HASHIRU owns the orchestration loop |
| 2 | What employees look like | **Managers or workers**, following [HASHIRU-lite](https://github.com/kunpai/com-sci-239-multi-agent-class-project) |
| 3 | Tool-creation sandbox | **`uv` subprocess now**, optional container backend later |
| 4 | Paper continuity | **This is v2.** No need to reproduce v1 HASHIRUBench numbers; v1 stays reproducible at a tag |
| 5 | UI (decided at Phase 5) | **OpenAI style instead of Gradio:** an OpenAI-compatible API (so Open WebUI, LibreChat and the SDKs work) and a built-in ChatGPT-style page on that API |
| 6 | Benchmarking (decided after Phase 5) | **No benchmark runs for v2.** Phase 7 is docs only |

## 1. Requirements restatement

HASHIRU v2 is a CEO → manager → worker hierarchy where:

- **Every agent is an A2A service** with its own Agent Card. Agents talk only over A2A.
- **Every tool is an MCP tool.** Managers (and the CEO) run a tool loop over MCP; workers answer once with no tools.
- **Budgets follow the hiring tree.** Each agent has a dollar pool reserved from its hirer. Local agents also hold real memory (MB of the loaded model). Firing returns unspent dollars and frees memory.
- **v1 features that HASHIRU-lite left out come back:** autonomous tool creation, persistent memory with retrieval, a web UI (v1's was Gradio; v2's is OpenAI-style, see decision 5), the `Mode` toggles, multi-session use.
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
   Chat page / OpenAI-style apps · external A2A clients · MCP hosts (Claude Code, Cursor, ...)
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
  ui/                chat page (static HTML/JS/CSS) + agents panel
  interop/           OpenAI-compatible API, CEO as public A2A agent, HASHIRU as MCP server
conf/                models, budget, modes, mcp_servers, a2a_peers
tests/               ported lite tests + new ones
```

v1 code (`src/manager`, `src/tools`, `src/models`, `app.py`, `start.py`, `deleteAgents.py`, `requirements.txt`) was deleted in Phase 5, once the new UI replaced it. v1 lives on at tag `v1.0-paper`.

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
- [x] v1's `python_sandbox_tool` and `weather_tool` were removed with the rest of the v1 code in Phase 5.
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

### Phase 4: A2A beyond the local host (3–4 days) — DONE 2026-10-29 on `feature/a2a-peers`

- [x] **Budget enforcement in code** (`enforce_budget`, on by default; v1 enforced budgets, lite only recorded them). A hire cannot reserve more than the hirer has left; a local model must fit in free memory unless already loaded; a non-local agent whose pool is spent makes no more LLM calls. All awaits in `hire_agent` happen before the money check and the reservation, so concurrent hires in one turn cannot spend the same money. `enforce_budget=false` keeps lite's accounting-only behavior for policy experiments.
- [x] **Modes:** `allow_hiring` (v1 agent creation), `allow_local_agents`, `allow_cloud_agents` (remote peers count as cloud); `tool_creation` (Phase 3) and `memory_autoinject` (Phase 2) cover the rest. v1's tool-invocation switch is dropped: managers without tools are workers.
- [x] **Deadlines instead of per-hop timeouts:** a user request gets one absolute deadline (`task_timeout`, default 30 min) that travels in A2A message metadata; every agent's run is bounded by it, so a timed-out agent stops instead of running on orphaned (fixes a Phase 1 known limitation).
- [x] **Cancellation:** firing an agent or closing a session cancels its in-flight runs; the request gets a `failed: stopped` reply (fixes the other Phase 1 limitation, the one extra LLM call).
- [x] **A2A task mode:** found while testing cancellation, every A2A request in lite's message-mode pattern left the a2a-sdk's two worker tasks pending forever. Replies now complete their task (`peers.finish`), and nothing is left running.
- [x] **Remote agents:** `a2a_peers` (url, headers, optional `price_usd_per_message`); hired as `a2a/<peer>` workers (never managers: they cannot use our tools or hire in our tree). The hub fetches the Agent Card, takes the price from the cost extension or config (the larger; unpriced peers are refused) and proxies the peer behind its own gate and trace. Each message checks the pool and the wallet, carries the hire prompt and the deadline, and charges the larger of the declared and reported cost.
- [x] **HASHIRU cost extension** (`docs/a2a-cost-extension.md`): Agent Cards declare per-message and per-token prices; replies report `hashiru_cost_usd`; requests carry `hashiru_deadline`. Every agent a hub hosts publishes it.
- [ ] **A2A streaming** of progress events moves to Phase 5, where the public CEO endpoint and the UI need it; the trace already records every event as it happens.
- [x] Tests: 18 in `tests/test_phase4.py` (197 total) with a stand-in remote agent (`tests/a2a_peer.py`); the A2A leak, deadline, cancellation and race fixes are each mutation-checked or shown failing without the fix.
- [x] Real run, local `gemma4:12b`, $0 in model cost: the CEO checked its pool, hired the stand-in peer `a2a/echo` (price $0.01 read from its card through a bearer-token-protected endpoint), relayed a message with the hire prompt and deadline, was charged $0.01, and reported the reply. Repeated after the review fixes: same result, and the peer's reply now carries `untrusted` up to the user.

Code review (independent reviewer with a hostile stand-in peer), all fixed with regression tests (`tests/test_phase4_review.py`, 225 tests in total); four fixes mutation-checked:
- HIGH: a peer's reported cost was unbounded (one reply of `1e9` exhausted the process wallet for every session) → only finite, non-negative reports count, and a charge never exceeds what the pool held.
- HIGH: the A2A client followed the endpoint named in the peer's card, sending our credentials and requests there (SSRF) → the card is fetched once at hire (size-capped, no redirects) and pinned to the configured URL.
- HIGH: a cloud manager that reserved everything it had left could never take another turn → hires keep `turn_reserve_usd` (default $0.01) for a cloud hirer's own turns.
- MED: concurrent asks to one agent overdrew its pool → every hosted agent (ours or a peer) serves one request at a time, and the peer price is reserved before sending.
- MED: token-priced peers were billed as free → their cost is estimated from text length. MED: peer relays ignored deadlines and fires → our agents and peers share one request path (`peers.serve_request`) with deadline, tracking, cancellation and cost reporting. MED: peer replies were unlabeled and unbounded → labeled per peer config (`untrusted` by default) and cut to 20,000 characters. MED: an empty message was billed and then lost → refused up front, and task-mode replies no longer depend on message text. MED: a cancelled LLM call's cost disappeared → the provider call is shielded and charged (and traced) when it returns. MED: two local hires could both take the last free memory → checking, reserving and loading local models is one step under a hub-wide lock. MED: every peer hire shared one A2A context → each hire is its own conversation.
- LOW: price changes after hire (fixed by pinning); unfinished or failed peer tasks counted as answers; reported costs left out hires and double-counted concurrent runs; caller deadlines could extend our work (now only shorten it); NaN prices, bad peer URLs and non-positive timeouts were accepted; float rounding lost a paid message; fired runs could not close their MCP sessions (runs are now stopped before tokens are revoked); a cancelled local hire left its model loaded; agents' task stores grew without bound (16 finished tasks kept).
- Found while re-testing: v2's default port collided with HASHIRU-lite's (9100), and a busy port crashed with a traceback → default 9200 and a one-line error.

Known limitations carried forward: the wallet cap can still be exceeded by one call per agent running at the same moment in different sessions; a peer's real cost is only as honest as its report (capped by the pool).

### Phase 5: UI and interop (3 days) — DONE 2026-10-29 on `feature/ui-interop`

Changed at the start of the phase (decision 5): no Gradio. HASHIRU speaks the OpenAI Chat Completions API, and its
own chat page is a client of that API.

- [x] **Conversations** (`interop/conversations.py`): every front end talks to one registry. A conversation is a hub session plus its chat; OpenAI-style clients resend the whole chat, so a request continues a conversation only when its earlier messages match that conversation's history (ignoring whitespace and reasoning blocks that chat apps fold into replies); an edited or regenerated chat gets a new session that is shown the chat so far. One turn at a time per conversation (409 otherwise); at most `max_sessions` open (the least recently used idle one is closed, 429 when all are busy); idle ones close after `session_ttl`. A client that stops listening stops the turn and closes the conversation, firing its agents.
- [x] **OpenAI-compatible API** (`/v1/models`, `/v1/chat/completions`, streaming and not): model `hashiru`. The agents' work streams as `reasoning_content` (shown as thinking by Open WebUI, LibreChat, ...); `deliver_to_user` text and the CEO's reply stream as `content`; a `hashiru` object carries the session, each trace event (without arguments or results) and the cost. `/v1/hashiru/info` and `/v1/hashiru/sessions/{id}` (GET pools, DELETE to close) serve the agents panel.
- [x] **Chat page** (`ui/`, served at `/`): ChatGPT-style layout (chat list in localStorage, centered thread, composer, stop button), each answer with a collapsible "Worked for Ns · k hires" list of what the agents did, and an Agents panel with the live hiring tree, pools and this turn's events. Static files with no build step and no external requests; strict CSP (`default-src 'self'`), and model output is rendered by a small Markdown renderer that builds DOM nodes and never parses HTML. Light and dark themes; phone layout with drawers.
- [x] **Public A2A agent:** the CEO's card at `/.well-known/agent-card.json` (streaming, cost extension, bearer security scheme when a key is set), JSON-RPC at `/a2a`; one conversation per A2A context; progress as `working` status updates; the caller's `hashiru_deadline` can shorten the turn's deadline. A2A streaming (deferred from Phase 4) is done here.
- [x] **MCP server** at `/public/mcp` for Claude Code, Cursor, ...: tools `ask` (progress notifications per agent step; `session` continues a conversation) and `end_session`; resource `hashiru://sessions/{session}`.
- [x] **Access:** `api_key` (bearer) guards `/v1`, `/a2a` and `/public/mcp`; listening beyond loopback requires one. Without a key, those routes answer only loopback `Host` names and loopback or absent `Origin`s, so a web page cannot reach them by DNS rebinding or a cross-site request (JSON-only bodies also force a CORS preflight). Internal routes keep per-session tokens.
- [x] **Single-port Hugging Face Spaces mode:** `hashiru serve` (graceful on SIGINT/SIGTERM: conversations closed, models unloaded) and a `Dockerfile` for a Docker Space (port 7860, uid 1000, `public_url` from `SPACE_HOST`, page framable by huggingface.co). README front matter switched from `sdk: gradio` to `sdk: docker`.
- [ ] Auth0 login: dropped for now. A key holder is one trust domain; per-user accounts can come from the chat app in front (Open WebUI has its own) or a later OAuth layer.
- [x] **v1 code deleted** on `develop` (it lives at tag `v1.0-paper`).
- [x] Tests: 41 new (266 total, plus 2 opt-in browser tests) in `tests/test_interop_unit.py`, `tests/test_interop.py` and `tests/test_interop_review.py`, using the official `openai` client, an A2A client and an MCP client against a served hub. One found a real bug: two conversations with the same history shared a fingerprint, and the first to move on unregistered the other.
- [x] **Empty final replies** (found in the chat page's first real two-worker run): `qwen3.5:4b`'s CEO hired and asked both workers, then answered "(empty)". Cause: Ollama's default context window (4096 tokens) was full: the prompt with both workers' replies was 3,839 tokens, the model's hidden reasoning used the remaining 257 and stopped (`finish_reason: length`) before any answer. Fixes: local models now run with `ollama_num_ctx` (default 16384) for loading and for every call, and the ledger records a loaded model's measured size from `/api/ps` (qwen3.5:4b: 3,537 MB at 16384, versus the 3,232 MB file size used before); an agent whose reply is empty is asked once more, and if it is empty again its caller gets "failed: <name> gave an empty answer" with a hint to raise `ollama_num_ctx`. Not yet confirmed with a real run after the context fix (the repeat with only the re-ask still answered "(empty)"). The bug dated from Phase 1, where the CLI printed a blank line.
- [x] Real runs, local `qwen3.5:4b`, $0: the chat page in headless Chrome (empty state, a live turn, agents panel, dark mode, 390 px phone layout with no horizontal scroll) and a streamed `curl` turn showing hire → ask → reply → fire as reasoning, then the answer; `hashiru serve host=0.0.0.0` with a key (401 without it, 200 with it, card names the public URL). The Docker image itself was not built here (no Docker daemon running).

Code review (independent reviewer, hub runs on test ports): no HIGH findings, no gate bypass (path tricks, HEAD/OPTIONS, Host and Origin forms all fail closed) and no XSS path. All findings fixed with regression tests (`tests/test_interop_review.py`, 13 tests; every fix mutation-checked), plus opt-in browser tests (`tests/test_ui_browser.py`):
- MED: normalizing a resent chat used a regex that was quadratic on unclosed `<think>` tags (one 154 KB request stalled the whole hub for 20 s; 2 MB would take minutes) → a linear scan (2 MB in under 0.1 s).
- MED: a non-streaming request kept running after its client left (Starlette only notices for streams), and the OpenAI client's retries would each open another conversation with its own budget → the handler watches for the disconnect and stops the turn.
- MED: deeply nested quotes or lists in a reply overflowed the page's stack, and the chat stayed broken in localStorage → nesting and line-length caps (deeper or longer text renders plain) and a plain-text fallback around the renderer. Testing this found that Markdown tables never rendered (a chained `append`); fixed.
- LOW: a sweep could close a conversation that a client had just claimed; concurrent opens could exceed `max_sessions`; requests during shutdown opened conversations that were never closed, and shutdown closed them one by one (past Docker's 10 s); `/public/mcp` refused a proxy's Host header with a key set (421); without a key, a page on another localhost port passed the Origin check; closed conversations left fingerprint entries behind; a consumer that failed mid-turn left its conversation busy until garbage collection; a pasted key with a trailing newline locked everyone out; the public card declared a price of $0 (a peer hub would treat it as free); idle conversations expired only when a new one opened (now swept every minute).
- Documented: with tool creation on and no key, agent-written tools could open conversations through the local API (each with a fresh budget; `max_usd` still caps them).

Known limitations: one API key is one trust domain (key holders can reach each other's conversations by id, and clients that send identical chats without the session header continue whichever conversation last had that history); Auth0 login is not ported; the Dockerfile was not built here (no Docker daemon), and whether a browser sends `Origin: null` for the page's own POSTs under `Referrer-Policy: no-referrer` was checked in Chrome only (the page works there without a key).

### Phase 6 (optional): Policy layer from lite (2–3 days, after 2026-12-03)

- [ ] Port lite's labels (`private`, `untrusted`), R1–R5 checks, `check.py`, and `--enforce` once lite's course Phase 2 is final.
- [ ] Generalize: policies loaded from config so users can add their own.

### Phase 7: Docs (1–2 days)

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
| 7 Docs (no benchmarking) | 1–2 days |
| **Total** | **~3–4 weeks** |

Phases 2, 3 and 4 depend only on Phase 1 and can run in parallel.
