"""The chat page in a real browser. Opt-in, since it needs Playwright and an installed Chrome:

    uv run --with playwright pytest tests/test_ui_browser.py
"""
import asyncio
import contextlib
import threading
from pathlib import Path

import pytest
from conftest import FakeOllama, make_settings
from test_interop import PORT, TeamLLM
from test_obfuscation import ObfLLM
from test_privacy_interop import ModelLLM

from hashiru.hub import Hub

playwright = pytest.importorskip("playwright.sync_api")
UI = Path(__file__).resolve().parent.parent / "src" / "hashiru" / "ui"

HOSTILE = {
    "quotes": ">" * 6000 + " deep",
    "lists": "\n".join(" " * (2 * i) + "- item" for i in range(3000)),
    "backticks": "`" * 100_000,
    "stars": "*a " * 30_000,
    "bold": "**" * 4000 + "x" + "**" * 4000,
    "markup": '<img src=x onerror="window.pwned=1"> <script>window.pwned=1</script> [x](javascript:window.pwned=1)',
}


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            chrome = p.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:  # no Chrome installed
            pytest.skip(f"no Chrome: {exc}")
        yield chrome
        chrome.close()


def test_the_markdown_renderer_survives_hostile_replies_and_never_makes_markup(browser):
    page = browser.new_page()
    page.set_content("<html><body></body></html>")
    page.add_script_tag(content=(UI / "markdown.js").read_text())
    result = page.evaluate("""cases => Object.fromEntries(Object.entries(cases).map(([name, text]) => {
        const div = document.createElement("div");
        const started = performance.now();
        div.append(window.renderMarkdown(text));
        document.body.append(div);
        return [name, {ms: performance.now() - started,
                       tags: [...new Set([...div.querySelectorAll("*")].map(e => e.tagName.toLowerCase()))],
                       hrefs: [...div.querySelectorAll("a")].map(a => a.getAttribute("href"))}];
    }))""", HOSTILE)
    assert all(r["ms"] < 1000 for r in result.values())
    assert not {"img", "script"} & {tag for r in result.values() for tag in r["tags"]}
    assert result["markup"]["hrefs"] == [] and page.evaluate("window.pwned") is None
    table = page.evaluate("""() => { const d = document.createElement("div");
        d.append(window.renderMarkdown("| a | b |\\n|---|---|\\n| 1 | **2** |")); return d.innerHTML; }""")
    assert table.startswith("<table><thead><tr><th>a</th>") and "<td><strong>2</strong></td>" in table
    page.close()


@contextlib.contextmanager
def serving(hub):
    """Serve the hub on a thread while the browser drives the page."""
    started, stop = threading.Event(), threading.Event()

    def run():
        async def go():
            async with hub.serving():
                started.set()
                while not stop.is_set():
                    await asyncio.sleep(0.05)
        asyncio.run(go())
    server = threading.Thread(target=run)
    server.start()
    try:
        assert started.wait(10)
        yield
    finally:
        stop.set()
        server.join(15)


def test_the_chat_page_runs_a_turn_and_shows_the_team(browser, tmp_path):
    hub = Hub(make_settings(tmp_path, port=PORT), llm=TeamLLM(), ollama=FakeOllama())  # type: ignore[arg-type]
    with serving(hub):
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(hub.settings.internal_url)
        page.click("#toggle-agents")
        page.fill("#input", "build a team")
        page.keyboard.press("Enter")
        page.wait_for_selector(".msg.ceo .meta", state="attached", timeout=15_000)
        answer = page.inner_text(".msg.ceo .md")
        work = page.text_content(".msg.ceo .work") or ""
        closed = page.evaluate("!document.querySelector('.msg.ceo .work').open")  # open only while working
        page.wait_for_selector(".agent .name:text('w1')", timeout=5_000)
        assert "a note for you" in answer and answer.rstrip().endswith("w1 says hello")
        assert "ceo hired w1" in work and closed and errors == []
        page.close()


def test_the_lock_keeps_a_chat_on_local_models_for_good(browser, tmp_path):
    llm = ModelLLM()
    hub = Hub(make_settings(tmp_path, port=PORT, private_mode="block"), llm=llm, ollama=FakeOllama())  # type: ignore[arg-type]
    with serving(hub):
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(hub.settings.internal_url)
        page.wait_for_selector("#model-sub:has-text('per chat')")
        page.click("#private")
        sub = page.inner_text("#model-sub")
        page.fill("#input", "my salary is 100k")
        page.keyboard.press("Enter")
        page.wait_for_selector(".msg.ceo .meta", state="attached", timeout=15_000)
        locked = page.get_attribute("#private", "aria-pressed"), page.is_disabled("#private")
        fine, listed = page.inner_text("#fine"), page.locator(".chat-row .open .lk").count()
        page.click("#new-chat")
        fresh = page.get_attribute("#private", "aria-pressed"), page.is_disabled("#private")
        page.close()
    assert sub == "CEO tiny · private, local only"
    assert llm.models == ["ollama_chat/tiny"]            # the cloud CEO never saw it
    assert locked == ("true", True) and "only local models" in fine and listed == 1
    assert fresh == ("false", False) and errors == []


def test_a_private_chat_stays_private_across_messages_and_reloads(browser, tmp_path):
    llm = ModelLLM()
    hub = Hub(make_settings(tmp_path, port=PORT, private_mode="block"), llm=llm, ollama=FakeOllama())  # type: ignore[arg-type]
    with serving(hub):
        page = browser.new_page()
        page.goto(hub.settings.internal_url)
        page.wait_for_selector("#model-sub:has-text('per chat')")
        page.click("#private")
        for done, text in enumerate(("first secret", "second secret")):
            page.fill("#input", text)
            page.keyboard.press("Enter")
            page.locator(".msg.ceo .meta").nth(done).wait_for(state="attached", timeout=15_000)
        page.reload()
        page.click("#chats .open")
        reloaded = page.get_attribute("#private", "aria-pressed"), page.is_disabled("#private")
        page.close()
    assert llm.models == ["ollama_chat/tiny", "ollama_chat/tiny"] and reloaded == ("true", True)


def test_the_lock_is_off_when_the_hub_has_no_local_model(browser, tmp_path):
    hub = Hub(make_settings(tmp_path, port=PORT, models=("gpt-4.1-mini",), private_mode="block"), llm=ModelLLM(),  # type: ignore[arg-type]
              ollama=FakeOllama())  # type: ignore[arg-type]
    with serving(hub):
        page = browser.new_page()
        page.goto(hub.settings.internal_url)
        page.wait_for_selector("#model-sub:has-text('per chat')")
        disabled, title = page.is_disabled("#private"), page.get_attribute("#private", "title")
        page.close()
    assert disabled and "no local model" in title


def test_a_lock_clicked_before_the_hub_answers_stays_on(browser, tmp_path):
    hub = Hub(make_settings(tmp_path, port=PORT, private_mode="block"), llm=ModelLLM(), ollama=FakeOllama())  # type: ignore[arg-type]
    with serving(hub):
        page = browser.new_page()

        def slow_info(route):
            page.wait_for_timeout(600)
            route.continue_()
        page.route("**/v1/hashiru/info", slow_info)
        page.goto(hub.settings.internal_url, wait_until="domcontentloaded")
        page.click("#private")
        page.wait_for_selector("#model-sub:has-text('private')", timeout=5_000)
        pressed = page.get_attribute("#private", "aria-pressed")
        page.close()
    assert pressed == "true"


def test_an_obfuscated_private_chat_keeps_the_cloud_ceo_and_shows_real_names(browser, tmp_path):
    llm = ObfLLM()
    hub = Hub(make_settings(tmp_path, port=PORT, obfuscate_with="patterns+model"), llm=llm,  # type: ignore[arg-type]
              ollama=FakeOllama())  # type: ignore[arg-type]
    with serving(hub):
        page = browser.new_page()
        page.goto(hub.settings.internal_url)
        page.wait_for_selector("#model-sub:has-text('per chat')")
        page.click("#private")
        sub, fine = page.inner_text("#model-sub"), page.inner_text("#fine")
        page.fill("#input", "Write to Alice Chen for me")
        page.keyboard.press("Enter")
        page.wait_for_selector(".msg.ceo .meta", state="attached", timeout=15_000)
        answer = page.inner_text(".msg.ceo .md")
        page.close()
    model, prompt = llm.seen[-1]
    assert sub == "CEO gpt-4.1-mini · private, obfuscated" and "hidden from cloud models" in fine
    assert model == "gpt-4.1-mini" and "Alice" not in prompt and answer.strip() == "noted Alice Chen"
