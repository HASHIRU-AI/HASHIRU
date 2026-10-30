# Plan Graphs (Phase 8c″) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An approved plan is compiled into a LangGraph graph (hires, steps, fires and the CEO's answer as nodes) and
the hub runs that graph, instead of telling the CEO "carry it out" and letting it improvise.

**Architecture:** `propose_plan` gains `steps` (each step sends one agent a task and names the steps whose results it
needs). `program.py` compiles the plan into a small IR: it inserts a hire node per planned hire, a step node per step
(waiting on its hire and its inputs), a fire node per hire after its last step, and one answer node; it reports
unknown agents, unknown inputs, cycles and unused hires as plan problems. `plangraph.py` turns that IR into a
LangGraph `StateGraph` (list edges are the joins) and runs it through a `Runner`; `runtime.PlanRunner` implements the
runner as the CEO, through the same `hire_agent`, `ask_agent` and `fire_agent` paths the CEO's own calls take, so the
trace, labels, R3, obfuscation, budgets, escalation and approvals all still apply.

**Tech Stack:** Python 3.13, LangGraph (`langgraph==1.2.15`), pytest with scripted fake LLMs.

**Spec:** this conversation (2026-10-30): "compile a plan into langgraph where we might add any intermediate nodes etc.
to make a cohesive executable plan", "i need langgraph". Builds on 8c′ in `plan/2026-10-29-modernize-mcp-a2a.md`.

## Global Constraints

- New dependency pinned exactly: `langgraph==1.2.15` (released 2026-10-12; newer releases are under two weeks old).
- No real model runs: verify with fake-LLM tests only (memory: no-real-model-runs).
- No code-reviewer subagent: self-verify with the full suite, ruff, mypy and mutation checks (memory).
- Every action a graph node takes goes through `runtime.perform` / `runtime.ask_agent` as the CEO; nodes never call
  an LLM except the answer node, which uses `runtime.think` (charged and traced to the CEO).
- LangGraph node names: `hire-<agent>`, `step-<id>`, `fire-<agent>`, `answer` (`:` is reserved by LangGraph; step
  ids and agent names use `[a-z0-9_]`, so prefixes cannot collide).
- New setting `plan_execution: graph | ceo`, default `graph` (env `HASHIRU_PLAN_EXECUTION`); `ceo` keeps 8c′.
- Files stay under ~400 lines; style follows the surrounding code (short docstrings, plain wording, `__all__`).
- **LangSmith tracing is always off** for plan graphs (`langsmith.tracing_context(enabled=False)` around the run):
  LangGraph pulls in `langsmith`, and with `LANGSMITH_TRACING=true` in the environment it POSTs every node's state
  (step results, which may be private) to LangSmith. Seen in a probe on 2026-10-30; R3 forbids it.

## Decisions made here (my calls, the user can override)

1. **Hires start at once, in parallel** (they wait only for the graph's start), because `propose_plan` already checked
   every hire's money and memory together. Each hire is **fired after its last step**, which returns its unspent
   reservation and frees a local model. Agents the CEO hired in earlier turns may be used by steps; they are neither
   hired nor fired.
2. **A failed step does not stop the graph.** Steps that need its result are skipped; independent steps still run;
   the CEO's answer is told what failed or was skipped. A failed hire fails its steps. (Replanning stays a follow-up.)
3. **A plan without `steps` gets one step per hire** (its `task`, no inputs), so every proposed plan with hires
   compiles. A plan with no hires, or one the CEO only described (`proposed=False`), or one whose steps do not
   compile, falls back to the 8c′ prompt.
4. **A step's message** is its task, plus "What the steps it builds on returned:" and each input's reply.
5. **The answer node** is one CEO LLM call with no tools: the turn's text plus every step's outcome. Empty once →
   nudged; empty twice → `runtime.empty`.
6. **A hire's system prompt** is the plan's optional `prompt`, else its `task`.
7. **Pregel supersteps:** LangGraph starts a superstep's nodes together and the next superstep after all of them
   finish, so a short chain beside a long step waits for it. Accepted for now; fan-out/fan-in plans are unaffected.
8. `recursion_limit` is set to the node count + 2 so long chains do not hit LangGraph's default of 25.

## Review Focus

0. `LANGSMITH_TRACING=true` (or `LANGCHAIN_TRACING_V2=true`) in the hub's environment: no tracer is made and no
   connection is opened. → Task 2 test `test_langsmith_tracing_stays_off_whatever_the_environment_says`.
1. A hire the user denies at an approval question: its steps fail with the reason, dependents are skipped, the answer
   still comes back (no hang, no crash). → Task 4 test `test_a_denied_hire_fails_its_steps_and_the_answer_says_so`.
2. Two steps on the same agent with no dependency between them: both run (the agent lock serializes them) and both
   results reach the answer. → Task 2 test `test_two_steps_on_one_agent_both_run`.
3. A chain longer than LangGraph's default recursion limit runs to the end. → Task 2 `test_a_long_chain_runs`.
4. A stale approved plan never runs on a later turn: `begin` clears `session.approved` on every turn. → Task 3
   `test_begin_clears_a_stale_approved_plan`.
5. Step messages and replies are `msg` events from and to the CEO (so R3 and obfuscation apply as for the CEO's own
   asks). → Task 4 `test_approve_runs_the_plan_as_a_graph` asserts the trace's `msg` events.

---

### Task 1: The plan's IR (`program.py`)

**Files:**
- Create: `src/hashiru/program.py`
- Test: `tests/test_plangraph.py`

**Interfaces:**
- Produces:
  - `Step(id: str, agent: str, task: str, after: tuple[str, ...] = ())` frozen, `.public() -> dict`
  - `Node(id: str, kind: str, agent: str = "", step: Step | None = None, waits: tuple[str, ...] = ())` frozen;
    `kind` in `"hire" | "step" | "fire" | "answer"`
  - `Program(nodes: tuple[Node, ...])` frozen, `.steps() -> list[Step]`
  - `parse_steps(raw: Any) -> tuple[list[Step], list[str]]` (steps, problems): ids match `tools.NAME`, unique; `after`
    a list of strings
  - `default_steps(hires: Iterable[tuple[str, str]]) -> list[Step]` from (name, task) pairs
  - `compile_steps(steps: Sequence[Step], hired: Sequence[str], live: Collection[str]) -> tuple[Program | None,
    list[str]]`: Program is None when there are problems

- [ ] **Step 1: Write the failing tests**

```python
def test_compile_inserts_hire_fire_and_answer_nodes():
    steps = [Step("a", "w1", "find"), Step("b", "w2", "find more"), Step("c", "w1", "combine", ("a", "b"))]
    program, problems = compile_steps(steps, ["w1", "w2"], live=())
    assert problems == []
    waits = {n.id: n.waits for n in program.nodes}
    assert waits == {"hire-w1": (), "hire-w2": (), "step-a": ("hire-w1",), "step-b": ("hire-w2",),
                     "step-c": ("hire-w1", "step-a", "step-b"), "fire-w1": ("step-a", "step-c"),
                     "fire-w2": ("step-b",), "answer": ("fire-w1", "fire-w2")}

def test_steps_on_an_agent_hired_earlier_get_no_hire_or_fire():
    program, _ = compile_steps([Step("a", "old", "x")], [], live={"old"})
    assert [(n.id, n.waits) for n in program.nodes] == [("step-a", ()), ("answer", ("step-a",))]

def test_compile_reports_what_cannot_run():
    steps = [Step("a", "nobody", "x"), Step("b", "w1", "y", ("zz",)), Step("c", "w1", "z", ("d",)),
             Step("d", "w1", "z", ("c",))]
    program, problems = compile_steps(steps, ["w1", "w2"], live=())
    text = "\n".join(problems)
    assert program is None
    for expected in ("nobody", "zz", "c, d", "w2"):  # unknown agent, unknown input, cycle, unused hire
        assert expected in text

def test_parse_steps_checks_ids_and_inputs():
    steps, problems = parse_steps([{"id": "a", "agent": "w1", "task": "x"}, {"id": "a", "agent": "w1"},
                                   {"id": "Bad", "agent": "w1"}, {"id": "e", "agent": "w1", "after": "a"}])
    assert [s.id for s in steps] == ["a"] and len(problems) == 3

def test_default_steps_give_each_hire_its_task():
    assert default_steps([("w1", "say hi")]) == [Step("w1", "w1", "say hi")]
```

- [ ] **Step 2: Run them to see them fail** — `uv run pytest tests/test_plangraph.py -q` → ImportError.
- [ ] **Step 3: Implement `program.py`.** Topological order by Kahn's algorithm in plan order; steps left over form
  the cycle, named sorted ("steps c, d wait for each other"). Node order: hires (plan order), steps (topological),
  fires, answer. The answer waits for every fire node and every sink step on a live agent.
- [ ] **Step 4: Run the tests** → PASS.
- [ ] **Step 5: Commit** (with Task 2).

### Task 2: LangGraph build and run (`plangraph.py`)

**Files:**
- Create: `src/hashiru/plangraph.py`
- Modify: `pyproject.toml` (`langgraph==1.2.15`), `uv.lock`
- Test: `tests/test_plangraph.py`

**Interfaces:**
- Consumes: `Program`, `Node`, `Step` (Task 1).
- Produces:
  - `Outcome(status: str, text: str)` frozen; status `"done" | "failed" | "skipped"`
  - `class Runner(Protocol)`: `async hire(agent) -> str | None` (None when hired, else why not);
    `async ask(agent, text) -> str`; `async fire(agent) -> None`;
    `async answer(results: list[tuple[Step, Outcome]]) -> str`; `note(node: Node, outcome: Outcome) -> None`
  - `message(step: Step, results: Mapping[str, Outcome]) -> str`
  - `build(program: Program, runner: Runner) -> CompiledStateGraph`
  - `async run(program: Program, runner: Runner) -> str` (the answer)
  - `mermaid(program: Program) -> str`

- [ ] **Step 1: Write the failing tests** (a `FakeRunner` that records calls and replies `"<agent> did <task>"`,
  fails agents in `fail`, and refuses hires in `refuse`):

```python
def test_the_graph_hires_runs_steps_in_order_fires_and_answers():
    runner = FakeRunner()
    answer = asyncio.run(run(compiled(DIAMOND), runner))   # a; b; c after a and b
    assert answer == "answered 3"
    assert set(runner.calls[:2]) == {("hire", "w1"), ("hire", "w2")}
    c = next(text for kind, agent, text in runner.asks if text.startswith("combine"))
    assert "w1 did find" in c and "w2 did find more" in c
    assert runner.calls.index(("fire", "w1")) > max(i for i, c in enumerate(runner.calls) if c[:2] == ("ask", "w1"))
    assert runner.calls[-1] == ("answer",)

def test_a_failed_step_skips_what_needs_it_and_the_rest_runs():
    runner = FakeRunner(fail={"w2"})
    asyncio.run(run(compiled(DIAMOND), runner))
    status = {s.id: o.status for s, o in runner.results}
    assert status == {"a": "done", "b": "failed", "c": "skipped"}

def test_a_refused_hire_fails_its_steps():
    runner = FakeRunner(refuse={"w2"})
    asyncio.run(run(compiled(DIAMOND), runner))
    assert {s.id: o.status for s, o in runner.results}["b"] == "failed"
    assert ("fire", "w2") not in runner.calls

def test_two_steps_on_one_agent_both_run(): ...      # a and b on w1, no inputs: both done
def test_a_long_chain_runs(): ...                    # 30 steps in a chain: all done
def test_mermaid_draws_the_compiled_graph():
    text = mermaid(compiled(DIAMOND))
    assert "hire-w1 --> step-a" in text and "fire-w2 --> answer" in text
```

- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: `uv add langgraph==1.2.15`, implement `plangraph.py`.** State is a TypedDict with `results` (step id →
  Outcome) and `hired` (agent → "" or why not), both merged by a dict-union reducer, and `answer: str`. Nodes with no
  waits get `START` edges; one wait is a plain edge; several are one list edge (a join). `answer → END`.
  `ainvoke(..., {"recursion_limit": len(program.nodes) + 2})`.
- [ ] **Step 4: Run the tests** → PASS.
- [ ] **Step 5: Commit** `feat(planning): compile plans into LangGraph graphs`.

### Task 3: Steps in the plan (`planning.py`, `session.py`, `config.py`)

**Files:**
- Modify: `src/hashiru/planning.py`, `src/hashiru/session.py`, `src/hashiru/config.py`,
  `src/hashiru/conf/config.yaml`, `src/hashiru/interop/activity.py`
- Test: `tests/test_planning.py`

**Interfaces:**
- Consumes: Task 1, `plangraph.mermaid`.
- Produces: `Hire.prompt: str = ""`; `Plan.steps: tuple[Step, ...] = ()`, `Plan.program: Program | None = None`
  (set when `plan_execution=graph` and the steps compile); `Plan.public()` adds `steps` and, with a program,
  `graph` (Mermaid); `Session.approved: Plan | None` (the approved plan the CEO's next run carries out);
  `Settings.plan_execution: str = "graph"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_propose_plan_compiles_steps_and_shows_them(hub, session):
    session.planning = True
    out = asyncio.run(planning.propose(hub, session, "ceo", STEPPED))  # w1 finds, w2 writes after it
    assert out["problems"] == [] and [s["id"] for s in out["steps"]] == ["find", "write"]
    assert "step-find --> step-write" in out["graph"]
    table = session.plan.table()
    assert "| write | w2 | find |" in table and "runs these steps" in table

def test_steps_that_cannot_run_are_plan_problems(hub, session): ...   # unknown agent and a cycle in problems
def test_a_plan_without_steps_gets_a_step_per_hire(hub, session): ... # PLAN → steps == [w1]
def test_begin_clears_a_stale_approved_plan(hub, session): ...         # approved set, begin("hi") → None
def test_plan_execution_setting():
    assert load_settings().plan_execution == "graph"
    with pytest.raises(ValueError, match="plan_execution"):
        load_settings(["plan_execution=fast"])
```

- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement.** `PROPOSE_TOOL` gains `hires[].prompt` and `steps` (`id`, `agent`, `task`, `after`);
  `PLAN_RULES` asks for steps. `propose` parses steps (or defaults them), compiles against the planned hires' names
  and the CEO's live hires, adds compile problems, and keeps the program only in graph mode. `table()` adds
  `| Step | Agent | After | Task |` and, with a program, "Approving runs these steps as a graph: ...". `begin`
  clears `session.approved`; on approve with a program it sets `session.approved` and returns `run_prompt(plan)`;
  otherwise `carry_out` (which now lists the steps). The `propose_plan` activity line counts steps; a `step` event
  gets a line ("step write (w2) skipped: ...").
- [ ] **Step 4: Run** `uv run pytest tests/test_planning.py -q` → PASS (8c′ conversation tests that script the CEO
  carrying a plan out run with `plan_execution=ceo`).
- [ ] **Step 5: Commit** (with Task 4).

### Task 4: The CEO runs the graph (`runtime.py`)

**Files:**
- Modify: `src/hashiru/runtime.py`
- Test: `tests/test_plangraph.py` (conversation tests with a scripted LLM)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `perform(hub, session, mcp, me, name, args, turn=None) -> dict` (dispatch after parsing);
  `PlanRunner(hub, session, mcp, plan, text)` implementing `Runner`; `run_agent` runs `session.approved` as a graph in
  the CEO's run; `step` trace events (`node`, `agent`, `status`, `why`).

- [ ] **Step 1: Write the failing tests**

```python
def test_approve_runs_the_plan_as_a_graph(tmp_path):
    first, second, events, llm = ...  # /plan, then approve; GraphLLM scripts the CEO and two workers
    assert second["reply"] == "the CEO's answer"
    assert [e["status"] for e in events if e["kind"] == "step"] == ["done", "done"]
    msgs = [(e["agent"], e["to"]) for e in events if e["kind"] == "msg" and e["agent"] != "user" and e["to"] != "user"]
    assert msgs == [("ceo", "w1"), ("w1", "ceo"), ("ceo", "w2"), ("w2", "ceo")]
    assert "w1's findings" in llm.worker_inputs["w2"][0]        # the join carried w1's reply
    assert llm.answer_tools == [None] and "w2's draft" in llm.answer_prompt
    assert [e["action"] for e in events if e["kind"] == "approval"] == ["planned", "planned"]
    assert fired == ["w1", "w2"]                                  # both fired after their steps

def test_a_denied_hire_fails_its_steps_and_the_answer_says_so(tmp_path): ...
def test_plan_execution_ceo_keeps_the_ceo_in_charge(tmp_path): ...   # the 8c′ prompt, no step events
```

- [ ] **Step 2: Run them to see them fail.**
- [ ] **Step 3: Implement.** In `run_agent`, after the MCP session opens: when `name == "ceo"` and `session.approved`
  is set, take it and `return await plangraph.run(plan.program, PlanRunner(...))`. `PlanRunner.hire` calls
  `perform(..., "ceo", "hire_agent", args)` with the hire's name, role, prompt, budget, tools and model or difficulty;
  `ask` is `ask_agent(hub, session, "ceo", ...)`; `fire` calls `fire_agent`; `answer` checks `spent_out` and
  `approvals.turn`, then `think` with no tools on the turn's text plus the outcomes; `note` emits `step` events.
- [ ] **Step 4: Run the full suite, ruff and mypy** → all pass.
- [ ] **Step 5: Mutation checks:** skip-on-failure, fire-after-last-step waits, the join list edge, the recursion
  limit, `begin` clearing `approved`, the fallback to `carry_out`, the answer offering no tools: each mutation must
  fail a test.
- [ ] **Step 6: Commit** `feat(planning): run approved plans as LangGraph graphs (Phase 8c″)`.

### Task 5: Docs

**Files:** `README.md` (plan mode section, layout), `docs/architecture.md` (plan mode bullet), master plan
(decision 14, phase 8c″ with follow-ups).

- [ ] Write them; commit `docs: plan graphs in README, architecture and plan`.

## Follow-ups (not in this plan)

- Replanning when a step fails (stop and show a revised plan) instead of skipping its dependents.
- Steps that call a tool directly (no agent), and conditional edges ("if the check fails, do X").
- A LangGraph checkpointer (SQLite) so a running plan survives a restart (Phase 9's durable sessions).
- Drawing the graph on the chat page (the plan carries Mermaid as `graph`).
- Eager scheduling past supersteps (decision 7) if uneven chains turn out to matter.
