from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pytest

from boss_react import BrowserConfig, BrowserSession, PlaywrightBrowserMiddleware

HTML = """
<!doctype html>
<html>
  <head><title>Browser tools test</title></head>
  <body>
    <main>
      <h1>Agent test page</h1>
      <label>Name <input name="name" placeholder="Your name"></label>
      <button id="send" onclick="document.querySelector('#result').textContent='sent'">Send</button>
      <select id="city"><option value="sh">Shanghai</option><option value="hz">Hangzhou</option></select>
      <input id="file" type="file">
      <button id="upload" onclick="document.querySelector('#file').click()">Upload image</button>
      <p id="result">idle</p>
    </main>
  </body>
</html>
"""

CHAT_HTML = """
<!doctype html>
<html>
  <head><title>Chat test</title></head>
  <body>
    <textarea id="chat-input"></textarea>
    <button id="send" onclick="document.querySelector('#sent').textContent = 'sent'">发送</button>
    <p id="sent">idle</p>
  </body>
</html>
"""


@pytest.fixture
async def session(tmp_path: Path):
    upload = tmp_path / "resume.png"
    upload.write_bytes(b"not-a-real-png")
    config = BrowserConfig(
        user_data_dir=tmp_path / "profile",
        artifacts_dir=tmp_path / "artifacts",
        headless=True,
        browser_channel="chrome",
        start_url="about:blank",
        upload_roots=(tmp_path,),
    )
    browser = BrowserSession(config)
    await browser.start()
    yield browser
    await browser.close()


@pytest.mark.asyncio
async def test_observe_fill_click_select_and_upload(session: BrowserSession, tmp_path: Path) -> None:
    await session.open("data:text/html," + quote(HTML))
    snapshot = await session.observe()

    assert snapshot["title"] == "Browser tools test"
    assert "Agent test page" in snapshot["text"]
    by_name = {element["name"]: element for element in snapshot["elements"] if element["name"]}
    by_text = {element["text"]: element for element in snapshot["elements"] if element["text"]}

    await session.fill(by_name["name"]["ref"], "Feng")
    await session.page.locator("#send").evaluate("el => el.outerHTML = el.outerHTML")
    await session.click(by_text["Send"]["ref"])
    await session.select_option(next(e["ref"] for e in snapshot["elements"] if e["tag"] == "select"), ["hz"])
    result = await session.upload(by_text["Upload image"]["ref"], [str(tmp_path / "resume.png")])

    assert await session.page.locator('input[name="name"]').input_value() == "Feng"
    assert await session.page.locator("#result").text_content() == "sent"
    assert await session.page.locator("#city").input_value() == "hz"
    assert await session.page.locator("#file").evaluate("el => el.files[0].name") == "resume.png"
    assert result["mode"] == "file_chooser"


@pytest.mark.asyncio
async def test_screenshot_returns_multimodal_content(session: BrowserSession) -> None:
    await session.open("data:text/html,<title>shot</title><button>ok</button>")
    content, artifact = await session.screenshot(include_image=True)

    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image"
    assert content[1]["mime_type"] == "image/png"
    assert content[1]["base64"]
    assert Path(artifact["path"]).is_file()


@pytest.mark.asyncio
async def test_upload_rejects_files_outside_roots(session: BrowserSession, tmp_path: Path) -> None:
    await session.open("data:text/html,<input type=file>")
    snapshot = await session.observe()
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    with pytest.raises(PermissionError):
        await session.upload(snapshot["elements"][0]["ref"], [str(outside)])


@pytest.mark.asyncio
async def test_state_gates_greeting_to_chat_page(session: BrowserSession) -> None:
    await session.open("data:text/html,<title>Not chat</title><button>Continue</button>")
    state = await session.state()

    assert state["page_state"]["kind"] == "other"
    assert state["page_state"]["capabilities"]["send_greeting"] is False
    with pytest.raises(ValueError, match="unavailable"):
        await session.send_greeting("Hello")

    await session.open("data:text/html;charset=utf-8," + quote(CHAT_HTML))
    state = await session.state()
    result = await session.send_greeting("老师您好")

    assert state["page_state"]["kind"] == "chat"
    assert state["page_state"]["capabilities"]["send_greeting"] is True
    assert state["page_state"]["available_special_tools"] == ["boss_send_greeting"]
    assert result["ok"] is True
    assert await session.page.locator("#sent").text_content() == "sent"


def test_middleware_registers_expected_tools(tmp_path: Path) -> None:
    middleware = PlaywrightBrowserMiddleware(
        BrowserConfig(
            user_data_dir=tmp_path / "profile",
            artifacts_dir=tmp_path / "artifacts",
            headless=True,
            browser_channel="chrome",
        )
    )
    names = {tool.name for tool in middleware.tools}

    assert {
        "browser_observe",
        "browser_state",
        "browser_click",
        "browser_fill",
        "browser_upload",
        "browser_screenshot",
        "boss_send_greeting",
    } <= names


def test_default_start_url_is_boss_homepage() -> None:
    assert BrowserConfig().start_url == "https://www.zhipin.com/"


@pytest.mark.asyncio
async def test_middleware_exposes_greeting_only_when_chat_is_ready(session: BrowserSession) -> None:
    middleware = PlaywrightBrowserMiddleware(session=session)

    class Request:
        def __init__(self, tools):
            self.tools = tools

        def override(self, **changes):
            return Request(changes.get("tools", self.tools))

    async def tool_names(request):
        return {tool.name for tool in request.tools}

    await session.open("data:text/html,<title>Home</title>")
    home_tools = await middleware.awrap_model_call(Request(middleware.tools), tool_names)

    await session.open("data:text/html;charset=utf-8," + quote(CHAT_HTML))
    chat_tools = await middleware.awrap_model_call(Request(middleware.tools), tool_names)

    assert "boss_send_greeting" not in home_tools
    assert "boss_send_greeting" in chat_tools
