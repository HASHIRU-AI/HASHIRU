"""The trace writer and the four label rules."""
import json

from hashiru.trace import SOURCES, Trace


def test_start_writes_jsonl(tmp_path):
    path = tmp_path / "t.jsonl"
    trace = Trace(path, task="t1", budget_usd=0.5)
    trace.emit("llm", task="t1", agent="ceo", text="hi")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["kind"] for e in lines] == ["start", "llm"] and [e["seq"] for e in lines] == [0, 1]
    assert lines[0]["budget_usd"] == 0.5


def test_rule1_sources():
    assert SOURCES == {"read_file": ["private"], "memory": ["private"], "get_website": ["untrusted"],
                       "arxiv_search": ["untrusted"]}


def test_rule2_labels_are_sticky_per_agent_and_task():
    trace = Trace(None, task="t1")
    trace.receive("mgr", "t1", ["private"])
    trace.receive("mgr", "t1", [])
    assert trace.labels("mgr", "t1") == {"private"}
    assert trace.labels("mgr", "t2") == set() and trace.labels("wkr", "t1") == set()


def test_rule3_outputs_carry_sender_labels():
    trace = Trace(None, task="t1")
    trace.receive("mgr", "t1", ["untrusted", "private"])
    assert trace.outgoing("mgr", "t1", "here is the summary") == ["private", "untrusted"]


def test_rule4_exact_done_or_failed_is_clean():
    trace = Trace(None, task="t1")
    trace.receive("mgr", "t1", ["private"])
    assert trace.outgoing("mgr", "t1", "done") == [] and trace.outgoing("mgr", "t1", " failed\n") == []
    assert trace.outgoing("mgr", "t1", "done: the notes say X") == ["private"]


def test_label_event_only_when_set_grows():
    trace = Trace(None, task="t1")
    trace.receive("mgr", "t1", ["private"])
    trace.receive("mgr", "t1", ["private"])
    label_events = [e for e in trace.events if e["kind"] == "label"]
    assert [(e["agent"], e["labels"]) for e in label_events] == [("mgr", ["private"])]
