# Model Recon (Phase 8b′) Implementation Plan — DONE 2026-10-30 on `feature/model-recon`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `hashiru models` shows which models this machine can actually use (local ones from Ollama, cloud ones whose
provider key is set), classifies each (tier, tools, reasoning, price, context), can refresh LiteLLM's model map and
check the keys with the providers, and `MODELS=auto` makes the hub hire from those picks with routing built from the
tiers.

**Architecture:** `catalog.py` is pure: it turns a LiteLLM-style model map, a "which providers have keys" predicate
and Ollama's model descriptions into `Card`s, gives each a tier, picks a few per tier and builds `routing`.
`recon.py` does the I/O: the cache file (a refreshed map and the providers' model lists), `refresh` (download the
map), `check` (ask each keyed provider which models it lists), `survey` (everything usable now) and `settle` (turn
`models=auto` into concrete `models` and `routing` before the Hub is built). The Hub registers the cached map at
start, so a refreshed price list also prices models named by hand.

**Tech Stack:** Python 3.13, LiteLLM's model map and helpers (`validate_environment`, `get_valid_models`), Ollama's
`/api/tags` and `/api/show`, httpx. No new dependencies.

**Spec:** this conversation (2026-10-30): "a recon step where if we 'refresh' we can use litellm registry ... to get
the freshly available models, and then check with the api keys set in .env to see which models we can actually use
... if litellm has parameter values, or some signal, we can autoclassify". The user skipped further questions.

## Global Constraints

- **No network unless asked.** Starting the hub (with or without `models=auto`) reads only the cache file, the
  environment and the local Ollama. `--refresh` downloads LiteLLM's public map
  (`https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json`); `--check` calls each
  keyed provider's model-listing endpoint (no inference, no cost). Neither sends user data.
- No real model runs (memory: no-real-model-runs); fakes for Ollama, the map download and the provider lists.
- No code-reviewer subagent (memory); full suite, ruff, mypy, mutation checks.
- New settings: `model_catalog: model_catalog.json` (env `HASHIRU_MODEL_CATALOG`), `auto_per_tier: 1`,
  `auto_max_usd: 30.0` ($ per million output tokens). `models: auto` (env `MODELS=auto`) turns recon on.
- `models=auto` with a non-empty `routing` is a configuration error ("routing cannot be set with models=auto").
- Branch `feature/model-recon` from `feature/plan-graph` (unmerged; both touch config and docs).

## Decisions made here (my calls, the user can override)

1. **Usable cloud model:** in the map with `mode: chat`, `supports_function_calling: true` (every pick can be a
   manager), input and output price both > 0 (drops the map's $0 experimental and `:free` entries, which would
   escape the pools), not past its `deprecation_date`, its provider's key set (`litellm.validate_environment`), and
   not a special-purpose or floating name: `ft:` fine-tunes, names containing `audio`, `search`, `realtime`,
   `computer-use`, `robotics`, `tts`, `transcribe`, `image`, `embed`, `moderation`, `safeguard`, `customtools`, or
   ending in `-latest` or containing `-exp`. A dated snapshot (`-YYYY-MM-DD`, `-YYYYMMDD`, `-MMDD`) is dropped when its
   undated alias is also usable.
2. **Usable local model:** pulled in Ollama, not an Ollama `-cloud` tag (runs on ollama.com, unpriced), not
   embedding-only, and with `tools` among its capabilities (an Ollama without `capabilities` counts as yes).
3. **Tiers:** cloud by output price per million tokens: < $2 easy, < $10 medium, else hard. Local by parameter count
   (`details.parameter_size`): < 8B easy, < 30B medium, else hard (unknown: easy). Constants in `catalog.py`.
4. **Auto picks:** every usable local model; for cloud, per provider and tier, the `auto_per_tier` best with output
   price ≤ `auto_max_usd`: reasoning models first, then cheapest for easy and medium, dearest for hard, ties by name
   descending (later version numbers usually sort later). If the provider was checked: only models it listed
   (matched with the provider prefix and snapshot suffix stripped), and none if it listed nothing (the key failed).
5. **Auto routing:** for each difficulty, that tier's picks, then the higher tiers, then the lower ones (best first),
   so escalation moves up. Within a tier, easy and medium put local first (small first for easy, large first for
   medium) then cloud cheapest first; hard puts cloud dearest first, then local largest first.
6. **The refreshed map** is validated before it is cached: a JSON object of objects; an entry with a non-finite or
   negative cost is dropped; fewer than 100 valid entries refuses the refresh and keeps the old cache.
7. `ceo_model` stays explicit (no `auto`); `hashiru models` prints a hint when the CEO model lacks tools.
8. **Changed while building:** keys are asked of LiteLLM only for `recon.KEYED`, 21 plain API-key providers.
   `validate_environment("github_copilot/...")` starts GitHub's device login and polls (it hung the first real
   `hashiru models`), and for providers whose keys LiteLLM does not know it answers yes. And the Hub registers only
   the models it uses from a refreshed map: registering all of it took 0.6 s and logged a warning per entry.

## Review Focus

1. A download that is not a model map (an HTML error page, a truncated file, entries with negative prices): the
   refresh fails or drops those entries and the old cache stays. → Task 2 `test_a_bad_download_keeps_the_old_cache`.
2. A provider that lists nothing on `--check` (bad key or network): its models are not picked and the table says the
   check failed; other providers are unaffected. → Task 1 `test_a_failed_check_drops_that_provider_only`.
3. Anthropic lists dated ids while the map uses aliases: the alias still counts as listed. → Task 1
   `test_listed_names_match_without_prefix_or_date`.
4. Ollama down when `models=auto` starts: cloud picks only, with a note; nothing usable at all is a clear error.
   → Task 2 `test_settle_without_ollama_uses_cloud_picks` and `test_settle_with_nothing_usable_says_why`.
5. Starting with `models=auto` opens no socket except to Ollama. → Task 2 `test_settle_opens_no_socket`.

---

### Task 1: Classification (`catalog.py`)

**Files:** Create `src/hashiru/catalog.py`; Test `tests/test_catalog.py`

**Interfaces (produces):**
- `Card(model, provider, local: bool, tools: bool, reasoning: bool, vision: bool, context: int | None,
  in_usd: float, out_usd: float, size_b: float | None, deprecated: str, tier: str)` frozen; prices per million tokens
- `cloud_cards(model_map: Mapping[str, Any], keyed: Callable[[str], bool], today: str) -> list[Card]`
- `local_cards(described: Mapping[str, Mapping[str, Any]]) -> list[Card]` (Ollama tag → `/api/show` JSON)
- `base(model: str) -> str` (no provider prefix, no snapshot suffix)
- `picks(cards: Sequence[Card], per_tier: int, max_usd: float, listed: Mapping[str, Collection[str]]) -> list[Card]`
- `routing(cards: Sequence[Card]) -> tuple[tuple[str, tuple[str, ...]], ...]`

- [x] **Step 1: Write the failing tests** (a synthetic map; `keyed` = provider in {"openai", "anthropic"}):
  `test_cloud_cards_keep_only_usable_chat_models` (drops embedding, no tools, $0, deprecated, unkeyed provider,
  `ft:`, audio/search/`-latest`/`-exp`, a snapshot whose alias is usable; keeps a snapshot without alias),
  `test_tiers_follow_price_and_size` (out $1.6 easy, $8 medium, $15 hard; 4B easy, 12B medium, 70B hard, unknown easy),
  `test_local_cards_need_tools_and_skip_cloud_and_embedding`, `test_picks_one_per_provider_and_tier`
  (reasoning first; easy/medium cheapest, hard dearest under `max_usd`; ties by name descending; all local kept),
  `test_a_failed_check_drops_that_provider_only`, `test_listed_names_match_without_prefix_or_date`,
  `test_routing_moves_up_then_down` (exact lists for a small set).
- [x] **Step 2: See them fail.**  **Step 3: Implement.**  **Step 4: Pass.**  **Step 5: Commit** (with Task 2).

### Task 2: Recon I/O and `models=auto` (`recon.py`, config, hub)

**Files:** Create `src/hashiru/recon.py`; Modify `src/hashiru/config.py`, `src/hashiru/conf/config.yaml`,
`src/hashiru/hub.py`, `src/hashiru/ollama.py`, `.gitignore`; Test `tests/test_catalog.py`, `tests/conftest.py`
(FakeOllama gains `describe`)

**Interfaces (produces):**
- `MAP_URL: str`; `load(path: str) -> dict[str, Any]` (cache: `{"map", "fetched", "listed": {provider: {"at",
  "models"}}}`, `{}` when absent or unreadable); `use_cached_map(path: str) -> None` (registers the cached map with
  LiteLLM)
- `async refresh(path: str, transport: httpx.AsyncBaseTransport | None = None) -> int` (entries kept; raises
  ValueError when the download is not a usable map)
- `async check(path: str, providers: Iterable[str]) -> dict[str, list[str]]` (runs `listed(provider)` in a thread,
  saves to the cache); `listed(provider: str) -> list[str]` wraps `get_valid_models(check_provider_endpoint=True,
  custom_llm_provider=provider)`
- `Survey(cards: list[Card], picks: list[Card], notes: list[str], keyed: list[str])`;
  `async survey(settings: Settings, ollama: Any) -> Survey`
- `async settle(settings: Settings, ollama: Any = None) -> Settings` (identity unless `models == ("auto",)`)
- `Ollama.describe() -> dict[str, dict[str, Any]]` (`/api/tags`, then `/api/show` for each, with `size`)
- `Hub.__init__` calls `use_cached_map(settings.model_catalog)` before `prices_file`, and refuses unsettled
  `models=auto` with "models=auto must be settled first (hashiru.recon.settle)".

- [x] **Step 1: Write the failing tests:** `test_refresh_caches_a_valid_map` (MockTransport; entries with bad
  prices dropped), `test_a_bad_download_keeps_the_old_cache`, `test_check_saves_each_providers_list`,
  `test_the_hub_prices_models_from_the_cached_map`, `test_settle_turns_auto_into_models_and_routing`,
  `test_settle_without_ollama_uses_cloud_picks`, `test_settle_with_nothing_usable_says_why`,
  `test_settle_opens_no_socket`, `test_auto_with_routing_is_a_config_error`, `test_an_unsettled_auto_hub_says_so`.
- [x] **Step 2: See them fail.**  **Step 3: Implement.**  **Step 4: Pass.**
- [x] **Step 5: Commit** `feat(models): model recon, refresh, key checks and models=auto`.

### Task 3: `hashiru models` and the CLI

**Files:** Modify `src/hashiru/cli.py`; Test `tests/test_catalog.py`

**Interfaces:** `hashiru models [--refresh] [--check] [--all] [overrides] [--config]` prints the map's source,
the keyed providers (and checks), and a table `Tier | Model | Where | Tools | Reasoning | Context | $/M in | $/M out |
Note`, picks marked `*` (only picks unless `--all`); `ask`, `chat` and `serve` call `recon.settle` first; `serve`
prints the auto picks per tier.

- [x] **Step 1: Write the failing tests:** `test_models_command_lists_picks_and_counts_the_rest`,
  `test_models_command_refreshes_and_checks` (fakes), `test_serve_settles_auto_first`.
- [x] **Step 2: See them fail.**  **Step 3: Implement.**  **Step 4: Full suite, ruff, mypy; mutation checks** (filter
  rules, tier thresholds, pick order, check filtering, cache validation, no-network settle, routing order).
- [x] **Step 5: Commit** `feat(cli): hashiru models`.

### Task 4: Docs

README (Models: recon and `MODELS=auto`), `.env.example`, `docs/architecture.md`, master plan (decision 15,
Phase 8b′). Commit `docs: model recon in README, architecture and plan`.

## Follow-ups (not in this plan)

- A refresh button on the chat page and `/v1/hashiru/models`.
- Track records (8e) reorder the auto routing by how models actually did.
- `ceo_model=auto` (pick a tool-capable medium model) once picks have been seen in real use.
