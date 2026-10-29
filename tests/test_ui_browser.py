"""The chat page in a real browser. Opt-in, since it needs Playwright and an installed Chrome:

    uv run --with playwright pytest tests/test_ui_browser.py
"""
import asyncio
import threading
from pathlib import Path

import pytest
from conftest import FakeOllama, make_settings
from test_interop import PORT, TeamLLM

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


def test_the_chat_page_runs_a_turn_and_shows_the_team(browser, tmp_path):
    hub = Hub(make_settings(tmp_path, port=PORT), llm=TeamLLM(), ollama=FakeOllama())  # type: ignore[arg-type]
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
    finally:
        stop.set()
        server.join(15)
