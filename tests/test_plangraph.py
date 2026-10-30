"""Plan graphs: an approved plan's steps compile into a LangGraph graph (hire, step, fire and answer nodes) that the
hub runs as the CEO (fake runners and fake LLMs, no network)."""
import asyncio
import socket

from langchain_core.tracers import langchain as langchain_tracers

from hashiru.plangraph import Outcome, mermaid, message, run
from hashiru.program import Program, Step, compile_steps, default_steps, parse_steps

DIAMOND = [Step("a", "w1", "find"), Step("b", "w2", "find more"), Step("c", "w1", "combine", ("a", "b"))]


class FakeRunner:
    """Each agent replies "<agent> did <first line of its message>"; agents in `fail` fail and hires in `refuse` are
    refused. Every call is recorded in order."""

    def __init__(self, fail=(), refuse=()):
        self.fail, self.refuse = set(fail), set(refuse)
        self.calls: list[tuple[str, str]] = []
        self.sent: dict[str, list[str]] = {}
        self.results: list[tuple[Step, Outcome]] = []
        self.notes: list[tuple[str, str]] = []

    async def hire(self, agent):
        self.calls.append(("hire", agent))
        return "the user said no" if agent in self.refuse else None

    async def ask(self, agent, text):
        self.calls.append(("ask", agent))
        self.sent.setdefault(agent, []).append(text)
        await asyncio.sleep(0)
        return f"failed: {agent} broke" if agent in self.fail else f"{agent} did {text.splitlines()[0]}"

    async def fire(self, agent):
        self.calls.append(("fire", agent))

    async def answer(self, results):
        self.calls.append(("answer", ""))
        self.results = results
        return f"answered {len(results)}"

    def note(self, node, outcome):
        self.notes.append((node.id, outcome.status))


def compiled(steps, live=()) -> Program:
    program, problems = compile_steps(steps, sorted({s.agent for s in steps} - set(live)), live)
    assert program is not None, problems
    return program


# ---- compiling ----

def test_compile_inserts_hire_fire_and_answer_nodes():
    program, problems = compile_steps(DIAMOND, ["w1", "w2"], live=())
    assert problems == [] and program is not None
    waits = {n.id: n.waits for n in program.nodes}
    assert waits == {"hire-w1": (), "hire-w2": (), "step-a": ("hire-w1",), "step-b": ("hire-w2",),
                     "step-c": ("hire-w1", "step-a", "step-b"), "fire-w1": ("step-a", "step-c"),
                     "fire-w2": ("step-b",), "answer": ("fire-w1", "fire-w2")}
    assert [n.kind for n in program.nodes] == ["hire", "hire", "step", "step", "step", "fire", "fire", "answer"]
    assert program.steps() == DIAMOND


def test_steps_run_in_dependency_order_whatever_order_they_were_given():
    program, _ = compile_steps(list(reversed(DIAMOND)), ["w1", "w2"], live=())
    assert program is not None
    assert [s.id for s in program.steps()] == ["b", "a", "c"]


def test_steps_on_an_agent_hired_earlier_get_no_hire_or_fire():
    program, problems = compile_steps([Step("a", "old", "x"), Step("b", "w1", "y", ("a",))], ["w1"], live={"old"})
    assert problems == [] and program is not None
    assert [(n.id, n.waits) for n in program.nodes] == [
        ("hire-w1", ()), ("step-a", ()), ("step-b", ("hire-w1", "step-a")), ("fire-w1", ("step-b",)),
        ("answer", ("fire-w1",))]
    program, _ = compile_steps([Step("a", "old", "x")], [], live={"old"})
    assert program is not None and [(n.id, n.waits) for n in program.nodes] == [("step-a", ()),
                                                                                ("answer", ("step-a",))]


def test_compile_reports_what_cannot_run():
    steps = [Step("a", "nobody", "x"), Step("b", "w1", "y", ("zz",)), Step("c", "w1", "z", ("d",)),
             Step("d", "w1", "z", ("c",)), Step("e", "w1", "z", ("e",))]
    program, problems = compile_steps(steps, ["w1", "w2"], live=())
    text = "\n".join(problems)
    assert program is None
    for expected in ("step a: nobody", "step b waits for zz", "c, d, e", "w2 is hired but no step"):
        assert expected in text, expected
    assert compile_steps([], [], live=()) == (None, [])  # nothing to run: the CEO answers by itself


def test_parse_steps_checks_ids_and_inputs():
    steps, problems = parse_steps([{"id": "a", "agent": "w1", "task": "x"}, {"id": "a", "agent": "w1"},
                                   {"id": "Bad", "agent": "w1"}, {"id": "e", "agent": "w1", "after": "a"},
                                   "not a step"])
    assert steps == [Step("a", "w1", "x")] and len(problems) == 4
    assert parse_steps([{"id": "b", "agent": "w2", "task": "y", "after": ["a"]}]) == ([Step("b", "w2", "y", ("a",))],
                                                                                    [])
    assert parse_steps("nope") == ([], ["steps must be a list of objects"])


def test_default_steps_give_each_hire_its_task():
    assert default_steps([("w1", "say hi"), ("w2", "")]) == [Step("w1", "w1", "say hi"), Step("w2", "w2", "")]


# ---- running ----

def test_the_graph_hires_runs_steps_in_order_fires_and_answers():
    runner = FakeRunner()
    assert asyncio.run(run(compiled(DIAMOND), runner)) == "answered 3"
    assert set(runner.calls[:2]) == {("hire", "w1"), ("hire", "w2")}            # hires start at once
    combine = runner.sent["w1"][1]
    assert combine.startswith("combine") and "[a]\nw1 did find" in combine and "[b]\nw2 did find more" in combine
    last_w1 = max(i for i, c in enumerate(runner.calls) if c == ("ask", "w1"))
    assert runner.calls.index(("fire", "w1")) > last_w1                        # fired after its last step
    assert runner.calls.index(("fire", "w2")) > runner.calls.index(("ask", "w2"))
    assert runner.calls[-1] == ("answer", "")
    assert [(s.id, o.status, o.text) for s, o in runner.results] == [
        ("a", "done", "w1 did find"), ("b", "done", "w2 did find more"), ("c", "done", "w1 did combine")]
    assert sorted(runner.notes) == [("step-a", "done"), ("step-b", "done"), ("step-c", "done")]


def test_a_step_without_inputs_gets_its_task_alone():
    assert message(Step("a", "w1", "find"), {}) == "find"


def test_a_failed_step_skips_what_needs_it_and_the_rest_runs():
    runner = FakeRunner(fail={"w2"})
    assert asyncio.run(run(compiled(DIAMOND), runner)) == "answered 3"
    outcomes = {s.id: (o.status, o.text) for s, o in runner.results}
    assert outcomes == {"a": ("done", "w1 did find"), "b": ("failed", "failed: w2 broke"),
                        "c": ("skipped", "it needs b, which failed")}
    assert runner.sent["w1"] == ["find"] and {("fire", "w1"), ("fire", "w2")} <= set(runner.calls)


def test_a_refused_hire_fails_its_steps():
    runner = FakeRunner(refuse={"w2"})
    asyncio.run(run(compiled(DIAMOND), runner))
    outcomes = {s.id: (o.status, o.text) for s, o in runner.results}
    assert outcomes["b"] == ("failed", "w2 was not hired: the user said no")
    assert outcomes["c"][0] == "skipped" and "w2" not in runner.sent
    assert ("fire", "w2") not in runner.calls and ("fire", "w1") in runner.calls


def test_a_skipped_step_skips_what_needs_it_too():
    steps = [Step("a", "w1", "x"), Step("b", "w1", "y", ("a",)), Step("c", "w1", "z", ("b",))]
    runner = FakeRunner(fail={"w1"})
    asyncio.run(run(compiled(steps), runner))
    assert [o.text for _, o in runner.results][1:] == ["it needs a, which failed", "it needs b, which was skipped"]
    assert len(runner.sent["w1"]) == 1


def test_two_steps_on_one_agent_both_run():
    runner = FakeRunner()
    asyncio.run(run(compiled([Step("a", "w1", "x"), Step("b", "w1", "y")]), runner))
    assert sorted(runner.sent["w1"]) == ["x", "y"] and [o.status for _, o in runner.results] == ["done", "done"]


def test_a_long_chain_runs():
    steps = [Step(f"s{i}", "w1", f"step {i}", (f"s{i - 1}",) if i else ()) for i in range(30)]
    runner = FakeRunner()
    asyncio.run(run(compiled(steps), runner))  # more supersteps than LangGraph's default recursion limit of 25
    assert len(runner.results) == 30 and all(o.status == "done" for _, o in runner.results)


def test_steps_on_agents_hired_before_run_without_hiring_or_firing():
    runner = FakeRunner()
    asyncio.run(run(compiled([Step("a", "old", "x")], live={"old"}), runner))
    assert runner.calls == [("ask", "old"), ("answer", "")]


def test_mermaid_draws_the_compiled_graph():
    text = mermaid(compiled(DIAMOND))
    for edge in ("__start__ --> hire-w1", "hire-w1 --> step-a", "step-a --> step-c", "step-b --> step-c",
                 "step-c --> fire-w1", "fire-w2 --> answer", "answer --> __end__"):
        assert edge in text, edge


def test_langsmith_tracing_stays_off_whatever_the_environment_says(monkeypatch):
    for key, value in {"LANGSMITH_TRACING": "true", "LANGCHAIN_TRACING_V2": "true", "LANGSMITH_API_KEY": "lsv2_x",
                       "LANGSMITH_ENDPOINT": "http://127.0.0.1:9"}.items():
        monkeypatch.setenv(key, value)
    tracers, connects = [], []
    real = langchain_tracers.LangChainTracer.__init__

    def tracer(self, *args, **kwargs):
        tracers.append(self)
        real(self, *args, **kwargs)

    def offline(self, address):
        connects.append(address)
        raise OSError("offline")

    monkeypatch.setattr(langchain_tracers.LangChainTracer, "__init__", tracer)
    monkeypatch.setattr(socket.socket, "connect", offline)
    asyncio.run(run(compiled(DIAMOND), FakeRunner()))
    assert tracers == [] and connects == []  # step results may be private: they never reach LangSmith (R3)
