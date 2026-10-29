"""Phase 5 pieces that need no server: parsing OpenAI-style chats, finding conversations, activity lines, the public
gate and the serving settings."""
import pytest
from conftest import make_settings

from hashiru.auth import AuthGate, is_open, is_public
from hashiru.config import load_settings
from hashiru.hub import Hub
from hashiru.interop.activity import describe, money, public_event
from hashiru.interop.conversations import Conversation, delivered, fingerprint
from hashiru.interop.openai_api import Problem, parse_messages
from hashiru.trace import Trace


def test_parse_messages_splits_past_turns_the_new_message_and_instructions():
    past, said, instructions = parse_messages([
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": [{"type": "text", "text": "and"}, {"type": "image_url", "image_url": {}}]},
        {"role": "user", "content": "now?"},
    ])
    assert past == [("hi", "hello")]
    assert said == "and\n\nnow?"          # a turn that got no reply joins the next one
    assert instructions == "Be brief."


@pytest.mark.parametrize("messages", [None, [], ["hi"], [{"role": "user", "content": "q"}, {"role": "assistant",
                                                                                           "content": "a"}],
                                      [{"role": "user", "content": "   "}]])
def test_parse_messages_needs_a_final_user_message_with_text(messages):
    with pytest.raises(Problem) as err:
        parse_messages(messages)
    assert err.value.status == 400


def test_fingerprints_ignore_folded_reasoning_and_whitespace():
    plain = [("hi", "hello there")]
    folded = [("hi ", '<details type="reasoning">\nthinking\n</details>\nhello  there'), ]
    assert fingerprint(plain) == fingerprint(folded) == fingerprint([("hi", "<think>x</think>hello there")])
    assert fingerprint(plain) != fingerprint([("hi", "hello")])


def test_a_conversation_is_found_only_by_its_exact_history(hub):
    convs = hub.conversations
    a = Conversation(hub.new_session("t", trace=False))
    b = Conversation(hub.new_session("t", trace=False))
    for conv in (a, b):
        convs.open_[conv.id] = conv
    convs._remember(a, "hi", "hello")
    assert convs.find([("hi", "hello")]) is a
    assert convs.find([("hi", "hello!")]) is None             # an edited reply: a new conversation
    assert convs.find([]) is None                              # a new chat never joins an old one
    convs._remember(b, "hi", "hello")                          # same history: the named one wins
    assert convs.find([("hi", "hello")], cid=a.id) is a
    convs._remember(a, "more", "sure")
    assert convs.find([("hi", "hello")], cid=a.id) is b        # a moved on; its old head is gone


def test_activity_lines_describe_what_agents_did():
    hire = {"kind": "tool", "agent": "ceo", "name": "hire_agent", "seq": 4,
            "args": {"role": "worker", "model": "gpt-4.1-mini", "budget_usd": 0.1, "prompt": "secret prompt"},
            "result": {"hired": "w1"}}
    assert describe(hire) == "ceo hired w1 as a worker on gpt-4.1-mini with $0.10"
    assert describe({"kind": "tool", "agent": "ceo", "name": "fire_agent", "args": {}, "result": {"fired": ["w1"]}}) \
        == "ceo fired w1"
    assert describe({"kind": "tool", "agent": "m", "name": "memory", "args": {}, "result": {"error": "nope"}}) \
        == "m: memory failed: nope"
    assert describe({"kind": "msg", "agent": "ceo", "to": "w1", "text": "go"}) == "ceo → w1: go"
    assert describe({"kind": "msg", "agent": "user", "to": "ceo", "text": "hi"}) is None
    assert describe({"kind": "llm", "agent": "ceo", "text": "thinking", "tool_calls": []}) is None
    assert describe({"kind": "llm", "agent": "ceo", "text": "I will hire", "tool_calls": ["hire_agent"]}) \
        == "ceo: I will hire"
    shown = public_event(hire)
    assert "args" not in shown and "result" not in shown and shown["line"].startswith("ceo hired")
    assert money(0) == "$0.00" and money(0.0042) == "$0.0042" and money(1.5) == "$1.50"


def test_delivered_text_is_attributed_to_its_agent():
    event = {"kind": "tool", "agent": "m1", "name": "deliver_to_user", "args": {"text": "report"},
             "result": {"delivered": True}}
    assert delivered(event) == "**m1:** report"
    assert delivered(event | {"result": {"error": "x"}}) is None


def test_trace_listeners_get_events_until_they_stop():
    trace = Trace(None, task="t")
    got: list[str] = []
    stop = trace.listen(lambda e: got.append(e["kind"]))
    trace.emit("llm")
    stop()
    stop()  # stopping twice is harmless
    trace.emit("tool")
    assert got == ["llm"]


def headers(host="127.0.0.1:9200", origin=None, token=None):
    out = [(b"host", host.encode())]
    if origin:
        out.append((b"origin", origin.encode()))
    if token:
        out.append((b"authorization", f"Bearer {token}".encode()))
    return out


def test_without_a_key_the_public_api_answers_only_loopback_requests(hub):
    gate: AuthGate = hub.asgi
    assert gate.check_public(headers()) is None
    assert gate.check_public(headers("localhost:9200", "http://localhost:9200")) is None
    assert gate.check_public(headers("[::1]:9200")) is None
    assert gate.check_public(headers("evil.example:9200"))[0] == 403          # DNS rebinding
    assert gate.check_public(headers(origin="https://evil.example"))[0] == 403  # a cross-site page
    assert gate.check_public(headers(origin="null"))[0] == 403
    assert gate.check_public(headers("[::1")[0:1])[0] == 403                   # malformed Host


def test_with_a_key_the_public_api_needs_it(tmp_path, ollama):
    hub = Hub(make_settings(tmp_path, api_key="k3y"), ollama=ollama)
    gate: AuthGate = hub.asgi
    assert gate.check_public(headers())[0] == 401
    assert gate.check_public(headers(token="wrong"))[0] == 401
    assert gate.check_public(headers("space.hf.space", "https://space.hf.space", token="k3y")) is None
    session = hub.new_session("t", trace=False)
    assert gate.check_public(headers(token=session.tokens["user"]))[0] == 401  # internal tokens are not keys


def test_route_classes():
    assert all(map(is_open, ["/", "/ui/app.js", "/.well-known/agent-card.json"]))
    assert all(map(is_public, ["/v1/models", "/v1/chat/completions", "/a2a", "/public/mcp"]))
    assert not any(map(is_public, ["/mcp", "/s/x/agents/ceo/", "/v1", "/a2ab", "/public/mcpx"]))
    assert not is_open("/ui")


def test_serving_settings():
    with pytest.raises(ValueError, match="needs an api_key"):
        load_settings(["host=0.0.0.0"])
    settings = load_settings(["host=0.0.0.0", "api_key=k3y", "public_url=https://me.example/"])
    assert settings.base_url == "https://me.example"
    assert "k3y" not in repr(settings)
    assert load_settings(["port=9300"]).base_url == "http://127.0.0.1:9300"
    with pytest.raises(ValueError, match="public_url"):
        load_settings(["public_url=ftp://x"])
    with pytest.raises(ValueError, match="max_sessions"):
        load_settings(["max_sessions=0"])
