"""Obfuscation, the default way private data reaches cloud models: sensitive values become reversible placeholders
(or, for amounts, dates, ages and health details, local generalizations), found by Presidio's patterns and a local
model, and restored on the way back (fake LLMs, no network)."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from conftest import ctx, make_settings
from test_interop import serve

from hashiru import runtime
from hashiru.config import load_settings
from hashiru.hub import Hub
from hashiru.interop import PRIVATE_HEADER
from hashiru.obfuscation import DETECT_PROMPT, PLACEHOLDER, Detector, Span, Vault
from hashiru.session import USER
from hashiru.trace import OBFUSCATED, PRIVATE

TEXT = "Alice Chen (alice@example.com, +1 415-555-0123) earns $100k; her key is sk-abc123def456ghi789jkl012."


def span(text, value, kind, general=""):
    start = text.index(value)
    return Span(start, start + len(value), kind, general)


# ---- the vault ----

def test_the_vault_gives_each_value_one_placeholder_for_the_whole_session():
    vault = Vault()
    first = vault.mask(TEXT, [span(TEXT, "Alice Chen", "PERSON"), span(TEXT, "alice@example.com", "EMAIL_ADDRESS"),
                              span(TEXT, "$100k", "MONEY", "a six-figure amount")])
    assert first == ("⟦PERSON_1⟧ (⟦EMAIL_1⟧, +1 415-555-0123) earns a six-figure amount; "
                     "her key is sk-abc123def456ghi789jkl012.")
    again = "Alice Chen met Bob Li."
    assert vault.mask(again, [span(again, "Alice Chen", "PERSON"), span(again, "Bob Li", "PERSON")]) \
        == "⟦PERSON_1⟧ met ⟦PERSON_2⟧."


def test_overlapping_spans_keep_the_longer_one():
    text = "write to alice@example.com"
    spans = [span(text, "example.com", "URL"), span(text, "alice@example.com", "EMAIL_ADDRESS")]
    assert Vault().mask(text, spans) == "write to ⟦EMAIL_1⟧"


def test_the_vault_restores_placeholders_even_when_a_model_rewrites_the_brackets():
    vault = Vault()
    vault.mask(TEXT, [span(TEXT, "Alice Chen", "PERSON"), span(TEXT, "alice@example.com", "EMAIL_ADDRESS")])
    assert vault.unmask("Dear ⟦PERSON_1⟧, I wrote to [EMAIL_1] and ⟦ PERSON_1 ⟧.") == \
        ("Dear Alice Chen, I wrote to alice@example.com and Alice Chen.", True)
    assert vault.unmask("Nothing here, not even ⟦PERSON_9⟧.") == ("Nothing here, not even ⟦PERSON_9⟧.", False)


def test_the_vault_masks_and_restores_json_values():
    vault = Vault()
    value = {"path": "notes.md", "lines": ["Alice Chen called", {"who": "Alice Chen"}], "n": 3}
    spans = {s: [span(s, "Alice Chen", "PERSON")] for s in ("Alice Chen called", "Alice Chen")}
    masked = vault.mask_value(value, lambda s: spans.get(s, []))
    assert masked == {"path": "notes.md", "lines": ["⟦PERSON_1⟧ called", {"who": "⟦PERSON_1⟧"}], "n": 3}
    assert vault.unmask_value(masked) == (value, True)


# ---- the detector ----

class DetectingLLM:
    """Answers the detector's prompt with `reply` (a JSON list, or anything else) and records the models it ran on."""

    def __init__(self, reply):
        self.reply, self.models = reply, []

    async def chat(self, model, messages, tools=None):
        self.models.append(model)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(content=self.reply, tool_calls=None), 0.0


def kinds(spans, text):
    return sorted((s.kind, text[s.start:s.end]) for s in spans)


def test_patterns_find_contact_details_and_secrets():
    found = asyncio.run(Detector(None, "").detect(TEXT))
    assert {("EMAIL_ADDRESS", "alice@example.com"), ("PHONE_NUMBER", "+1 415-555-0123"),
            ("SECRET", "sk-abc123def456ghi789jkl012")} <= set(kinds(found, TEXT))
    assert "PERSON" not in {s.kind for s in found}  # names need the local model


def test_the_local_model_finds_names_and_generalizes_what_matters():
    llm = DetectingLLM(json.dumps([{"text": "Alice Chen", "type": "PERSON"},
                                   {"text": "$100k", "type": "MONEY", "general": "a six-figure amount"},
                                   {"text": "Bob", "type": "PERSON"}]))  # not in the text: ignored
    found = asyncio.run(Detector(llm, "ollama_chat/tiny").detect(TEXT))
    assert ("PERSON", "Alice Chen") in kinds(found, TEXT) and llm.models == ["ollama_chat/tiny"]
    money = next(s for s in found if s.kind == "MONEY")
    assert money.general == "a six-figure amount"
    assert [value for kind, value in kinds(found, TEXT) if kind == "PERSON"] == ["Alice Chen"]  # no "Bob" anywhere


@pytest.mark.parametrize("reply", ["I cannot help with that", ConnectionError("Ollama is down"), '{"not": "a list"}'])
def test_a_detector_that_cannot_answer_returns_nothing_so_callers_refuse(reply):
    assert asyncio.run(Detector(DetectingLLM(reply), "ollama_chat/tiny").detect(TEXT)) is None


def test_the_detector_remembers_what_it_found():
    llm = DetectingLLM("[]")
    detector = Detector(llm, "ollama_chat/tiny")
    asyncio.run(detector.detect(TEXT))
    asyncio.run(detector.detect(TEXT))
    assert len(llm.models) == 1


# ---- across the hub (private_mode=obfuscate, the default) ----

NOTES = '{"path": "notes.md", "text": "Alice Chen earns 100k, write to alice@example.com"}'


class ObfLLM:
    """The local detector marks "Alice Chen" (or answers `detector` instead); any other call is an agent, which
    records what it saw and answers `answer`, by default naming the first placeholder it saw."""

    def __init__(self, detector=None, answer=None):
        self.detector, self.answer, self.seen = detector, answer, []

    async def chat(self, model, messages, tools=None):
        if messages[0]["content"] == DETECT_PROMPT:
            text = messages[1]["content"]
            found = self.detector if self.detector is not None else \
                json.dumps([{"text": "Alice Chen", "type": "PERSON"}] if "Alice Chen" in text else [])
            return SimpleNamespace(content=found, tool_calls=None), 0.0
        prompt = messages[-1]["content"]
        self.seen.append((model, prompt))
        match = PLACEHOLDER.search(prompt)
        return SimpleNamespace(content=self.answer or f"noted {match[0] if match else 'nothing'}",
                               tool_calls=None), 0.0


class RecordingMCP:
    def __init__(self, text=NOTES):
        self.text, self.calls = text, []

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)], is_error=False, structured_content=None)


def call(name, **args):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def obf_hub(tmp_path, ollama, llm=None, **settings):
    settings = {"obfuscate_with": "patterns+model"} | settings  # the fake LLM answers the detector
    return Hub(make_settings(tmp_path, **settings), llm=llm or ObfLLM(), ollama=ollama)  # type: ignore[arg-type]


def test_a_cloud_agent_reads_a_private_file_obfuscated(tmp_path, ollama):
    hub = obf_hub(tmp_path, ollama)
    session = hub.new_session("t", trace=False)  # the CEO runs on gpt-4.1-mini, a cloud model
    out = asyncio.run(runtime.dispatch(hub, session, RecordingMCP(), "ceo", call("read_file", path="notes.md")))
    assert out["text"] == "⟦PERSON_1⟧ earns 100k, write to ⟦EMAIL_1⟧"
    assert session.trace.labels("ceo", session.id) == {OBFUSCATED}  # it holds no private data
    tool = next(e for e in session.trace.events if e["kind"] == "tool")
    assert "Alice" not in json.dumps(tool) and tool["labels"] == [OBFUSCATED]


def test_placeholders_are_restored_for_local_tools_but_not_for_the_web(tmp_path, ollama):
    hub = obf_hub(tmp_path, ollama)
    session = hub.new_session("t", trace=False)
    session.vault.mask("Alice Chen", [Span(0, 10, "PERSON")])
    mcp = RecordingMCP('{"saved": "pet"}')
    asyncio.run(runtime.dispatch(hub, session, mcp, "ceo", call("memory", action="add", key="pet",
                                                                    text="⟦PERSON_1⟧ has a cat")))
    asyncio.run(runtime.dispatch(hub, session, mcp, "ceo", call("deliver_to_user", text="Hi ⟦PERSON_1⟧")))
    asyncio.run(runtime.dispatch(hub, session, mcp, "ceo", call("get_website", url="https://x.example/?q=⟦PERSON_1⟧")))
    assert [args for _, args in mcp.calls] == [{"action": "add", "key": "pet", "text": "Alice Chen has a cat"},
                                               {"text": "Hi Alice Chen"},
                                               {"url": "https://x.example/?q=⟦PERSON_1⟧"}]


def test_messages_cross_to_the_cloud_masked_and_come_back_restored(tmp_path, ollama, monkeypatch):
    hub = obf_hub(tmp_path, ollama)
    session = hub.new_session("t", trace=False)  # a cloud CEO
    session.ledger.add("w1", "worker", "local", "ollama_chat/tiny", "p", "ceo", 1, 0.0)
    session.vault.mask("Alice Chen", [Span(0, 10, "PERSON")])
    got = []

    async def fake_send(hub_, session_, me, name, text):
        got.append(text)
        session.trace.receive("w1", session.id, [PRIVATE])  # w1 read her file while working
        return "Alice Chen earns 100k"
    monkeypatch.setattr(runtime, "a2a_send", fake_send)
    reply = asyncio.run(runtime.ask_agent(hub, session, "ceo", "w1", "summarize ⟦PERSON_1⟧'s notes"))
    assert got == ["summarize Alice Chen's notes"]               # the local worker sees the real name
    assert reply == "⟦PERSON_1⟧ earns 100k"                       # the cloud CEO does not
    assert PRIVATE in session.trace.labels("w1", session.id)
    assert session.trace.labels("ceo", session.id) == {OBFUSCATED}


def test_a_private_hirer_gives_a_cloud_hire_an_obfuscated_prompt(tmp_path, ollama):
    hub = obf_hub(tmp_path, ollama)
    session = hub.new_session("t", trace=False)
    session.trace.receive("ceo", session.id, [PRIVATE])
    out = asyncio.run(hub.tools.hire_agent(ctx(session, "ceo"), "c1", "worker", "gpt-4.1-mini",  # type: ignore[arg-type]
                                           "Draft a reply to Alice Chen", 0.05))
    assert out["hired"] == "c1" and session.ledger.agents["c1"].prompt == "Draft a reply to ⟦PERSON_1⟧"
    assert session.trace.labels("c1", session.id) == {OBFUSCATED}


def test_a_private_chat_keeps_its_cloud_ceo_and_shows_you_the_real_names(tmp_path):
    llm = ObfLLM()

    async def body(hub, base):
        async with httpx.AsyncClient(base_url=f"{base}/v1", timeout=10) as http:
            r = await http.post("/chat/completions", headers={PRIVATE_HEADER: "1"}, json={
                "model": "hashiru", "messages": [{"role": "user", "content": "Write to Alice Chen for me"}]})
            return r.json()["choices"][0]["message"]["content"]

    answer = serve(tmp_path, body, llm, obfuscate_with="patterns+model")
    model, prompt = llm.seen[-1]
    assert model == "gpt-4.1-mini" and "⟦PERSON_1⟧" in prompt and "Alice" not in prompt
    assert answer == "noted Alice Chen"


def test_when_detection_fails_the_crossing_is_refused(tmp_path, ollama):
    hub = obf_hub(tmp_path, ollama, ObfLLM(detector="sorry, I cannot do that"))
    session = hub.new_session("t", trace=False)
    out = asyncio.run(runtime.dispatch(hub, session, RecordingMCP(), "ceo", call("read_file", path="notes.md")))
    assert out["error"].startswith("R3") and "could not be found" in out["error"]
    assert PRIVATE not in session.trace.labels("ceo", session.id)


def test_block_mode_refuses_and_a_hub_without_a_local_detector_blocks(tmp_path, ollama):
    strict = obf_hub(tmp_path, ollama, private_mode="block")
    mcp = RecordingMCP()
    out = asyncio.run(runtime.dispatch(strict, strict.new_session("t", trace=False), mcp, "ceo",
                                       call("read_file", path="notes.md")))
    assert out["error"].startswith("R3") and mcp.calls == []
    assert obf_hub(tmp_path, ollama, models=("gpt-4.1-mini",)).obfuscating() is None
    assert obf_hub(tmp_path, ollama, models=("gpt-4.1-mini",), obfuscate_with="patterns").obfuscating() is not None


def test_recall_reaches_a_cloud_ceo_obfuscated(tmp_path, ollama):
    hub = obf_hub(tmp_path, ollama, memory_autoinject=True)
    session = hub.new_session("t", trace=False)
    session.memory.add("pet", "Alice Chen has a cat named Miso", [])
    text = hub.recall(session, "what is the cat named?")
    assert "Miso" in text and session.trace.labels(USER, session.id) == {PRIVATE}  # masked when sent to the CEO
    assert PRIVATE not in session.trace.labels("ceo", session.id)


def test_obfuscation_settings():
    assert load_settings([]).private_mode == "obfuscate"
    for bad, match in (("private_mode=sometimes", "private_mode"), ("obfuscate_with=magic", "obfuscate_with"),
                       ("obfuscate_model=gpt-4.1-mini", "obfuscate_model")):
        with pytest.raises(ValueError, match=match):
            load_settings([bad])


# ---- follow-ups from the real run ----

def test_detection_never_goes_to_the_network(monkeypatch, tmp_path):
    import socket

    import tldextract

    def offline(*args, **kwargs):
        raise AssertionError("detection tried to reach the network")
    monkeypatch.setattr(socket.socket, "connect", offline)
    # A fresh machine: tldextract's default extractor, with nothing cached, would download the public suffix list.
    monkeypatch.setattr(tldextract.tldextract, "TLD_EXTRACTOR", tldextract.TLDExtract(cache_dir=str(tmp_path)))
    text = "mail bob@example.org or visit https://example.net/x"
    found = asyncio.run(Detector(None, "").detect(text))
    assert ("EMAIL_ADDRESS", "bob@example.org") in kinds(found, text)


def test_local_parties_hold_real_values_not_obfuscated_ones(tmp_path, ollama, monkeypatch):
    hub = obf_hub(tmp_path, ollama)
    session = hub.new_session("t", trace=False)
    session.trace.receive("ceo", session.id, [OBFUSCATED])  # the cloud CEO saw a masked message
    session.ledger.add("w1", "worker", "local", "ollama_chat/tiny", "p", "ceo", 1, 0.0)
    session.vault.mask("Alice Chen", [Span(0, 10, "PERSON")])

    async def fake_send(hub_, session_, me, name, text):
        return "done for ⟦PERSON_1⟧"
    monkeypatch.setattr(runtime, "a2a_send", fake_send)
    asyncio.run(runtime.ask_agent(hub, session, "ceo", "w1", "write to ⟦PERSON_1⟧"))
    assert session.trace.labels("w1", session.id) == {PRIVATE}
    args = {"name": "w2", "role": "worker", "model": "ollama_chat/tiny", "prompt": "greet ⟦PERSON_1⟧",
            "budget_usd": 0.0}
    hire = SimpleNamespace(id="c2", function=SimpleNamespace(name="hire_agent", arguments=json.dumps(args)))
    asyncio.run(runtime.dispatch(hub, session, HiringMCP(hub, session), "ceo", hire))
    assert session.trace.labels("w2", session.id) == {PRIVATE}


class HiringMCP:
    """Runs hire_agent on the hub, as the real MCP endpoint would."""

    def __init__(self, hub, session):
        self.hub, self.session = hub, session

    async def call_tool(self, name, args):
        out = await self.hub.tools.hire_agent(ctx(self.session, "ceo"), **args)
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(out))], is_error=False,
                               structured_content=out)
