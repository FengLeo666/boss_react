from __future__ import annotations

import asyncio
import base64
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "nodriver_tool_driver.py"
SPEC = importlib.util.spec_from_file_location("nodriver_tool_driver", SCRIPT)
assert SPEC and SPEC.loader
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class FakeTab:
    def __init__(self, target_id: str, visibility: str, *, stuck: bool = False) -> None:
        self.target_id = target_id
        self.visibility = visibility
        self.stuck = stuck
        self.activate = AsyncMock()
        self.bring_to_front = AsyncMock()
        self.close = AsyncMock()
        self.handlers: dict[type, list] = {}

    def add_handler(self, event_type: type, callback) -> None:
        self.handlers.setdefault(event_type, []).append(callback)

    def remove_handler(self, event_type: type, callback) -> None:
        self.handlers[event_type].remove(callback)

    async def send(self, command: object) -> str:
        del command
        if self.stuck:
            await asyncio.Event().wait()
        return base64.b64encode(b"png-data").decode("ascii")

    async def evaluate(self, expression: str) -> str:
        assert expression == "document.visibilityState"
        return self.visibility


def make_session(tmp_path: Path, tabs: list[FakeTab]) -> tuple[object, list[FakeTab]]:
    active = [tabs[0]]
    browser = SimpleNamespace(tabs=tabs, update_targets=AsyncMock())
    backend = SimpleNamespace(
        get_tab=lambda: active[0],
        get_browser=lambda: browser,
        set_tab=lambda tab: active.__setitem__(0, tab),
    )
    session = driver.NodriverToolSession(backend, tmp_path)
    session.state = AsyncMock(return_value={"page_state": {"kind": "chat"}})
    for tab in tabs:
        async def close_tab(tab_to_close: FakeTab = tab) -> None:
            tabs.remove(tab_to_close)
        tab.close.side_effect = close_tab
    return session, active


@pytest.mark.asyncio
async def test_state_uses_live_page_url_when_target_metadata_is_stale(tmp_path: Path) -> None:
    session, _ = make_session(tmp_path, [FakeTab("home", "visible")])
    session._evaluate = AsyncMock(  # type: ignore[method-assign]
        return_value={"title": "BOSS直聘", "url": "https://www.zhipin.com/", "chatInputReady": False}
    )
    session.state = driver.NodriverToolSession.state.__get__(session)  # type: ignore[method-assign]

    result = await session.state()

    assert result["url"] == "https://www.zhipin.com/"
    assert result["page_state"]["kind"] == "boss_home"


@pytest.mark.asyncio
async def test_screenshot_captures_current_target_without_nodriver_sleep(tmp_path: Path) -> None:
    session, active = make_session(tmp_path, [FakeTab("chat", "visible")])

    result = await session.screenshot()

    assert result["target_id"] == "chat"
    assert Path(result["path"]).read_bytes() == b"png-data"
    assert active[0].target_id == "chat"


@pytest.mark.asyncio
async def test_settle_waits_and_returns_post_action_screenshot(tmp_path: Path) -> None:
    tab = FakeTab("chat", "visible")
    session, _ = make_session(tmp_path, [tab])
    started = asyncio.get_running_loop().time()

    result = await session.settle_and_screenshot()

    assert asyncio.get_running_loop().time() - started >= 1
    assert result["network_idle"] is True
    assert Path(result["path"]).read_bytes() == b"png-data"
    assert all(not callbacks for callbacks in tab.handlers.values())


@pytest.mark.asyncio
async def test_screenshot_recovers_from_stuck_hidden_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "SCREENSHOT_TIMEOUT_SECONDS", 0.01)
    hidden = FakeTab("old-detail", "hidden", stuck=True)
    visible = FakeTab("chat", "visible")
    session, active = make_session(tmp_path, [hidden, visible])

    result = await session.screenshot()

    assert result["target_id"] == "chat"
    assert Path(result["path"]).read_bytes() == b"png-data"
    assert active[0] is visible
    visible.activate.assert_awaited_once_with()
    visible.bring_to_front.assert_awaited_once_with()
    hidden.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_new_tab_replaces_old_tab(tmp_path: Path) -> None:
    old = FakeTab("list", "visible")
    new = FakeTab("detail", "visible")
    tabs = [old]
    session, active = make_session(tmp_path, tabs)
    tabs.append(new)
    new.close.side_effect = lambda: tabs.remove(new)

    assert await session._adopt_new_tab({"list"}) is True

    assert active[0] is new
    assert tabs == [new]
    old.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_screenshot_adopts_tab_opened_after_click_returns(tmp_path: Path) -> None:
    old = FakeTab("list", "visible")
    new = FakeTab("detail", "visible")
    session, active = make_session(tmp_path, [old, new])

    result = await session.screenshot()

    assert result["target_id"] == "detail"
    assert active[0] is new
    old.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_closed_tab_is_removed_before_next_screenshot(tmp_path: Path) -> None:
    old = FakeTab("list", "hidden")
    new = FakeTab("detail", "visible")
    tabs = [old, new]
    session, active = make_session(tmp_path, tabs)
    old.close.side_effect = None

    async def refresh_targets() -> None:
        if old.close.await_count and old in tabs:
            tabs.remove(old)

    session.browser.update_targets.side_effect = refresh_targets

    first = await session.screenshot()
    second = await session.screenshot()

    assert first["target_id"] == second["target_id"] == "detail"
    assert active[0] is new
    assert tabs == [new]
    old.close.assert_awaited_once_with()
    old.activate.assert_not_awaited()


@pytest.mark.asyncio
async def test_late_new_tab_recovers_hidden_page_screenshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "SCREENSHOT_TIMEOUT_SECONDS", 0.01)
    old = FakeTab("list", "hidden")
    new = FakeTab("detail", "hidden")
    tabs = [old]
    session, active = make_session(tmp_path, tabs)

    async def open_late_tab(_command: object) -> str:
        if new not in tabs:
            tabs.append(new)
        await asyncio.Event().wait()
        return ""

    old.send = open_late_tab  # type: ignore[method-assign]

    result = await session.screenshot()

    assert result["target_id"] == "detail"
    assert active[0] is new
    assert tabs == [new]


@pytest.mark.asyncio
async def test_failed_activation_keeps_current_page(tmp_path: Path) -> None:
    old = FakeTab("list", "visible")
    new = FakeTab("detail", "visible")
    session, active = make_session(tmp_path, [old, new])
    new.activate.side_effect = RuntimeError("Session with given id not found")

    with pytest.raises(driver.BrowserToolTimeout, match="Browser page session is gone"):
        await session.screenshot()

    assert active[0] is old
    old.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_reset_discards_closed_target_before_screenshot(tmp_path: Path) -> None:
    old = FakeTab("detail", "visible")
    home = FakeTab("home", "visible")
    tabs = [old]
    session, active = make_session(tmp_path, tabs)
    old.close.side_effect = RuntimeError("Session with given id not found")

    async def open_home(_url: str, *, new_tab: bool) -> FakeTab:
        assert new_tab is True
        tabs.append(home)
        return home

    async def refresh_targets() -> None:
        if old.close.await_count and old in tabs:
            tabs.remove(old)

    session.browser.get = AsyncMock(side_effect=open_home)
    session.browser.update_targets.side_effect = refresh_targets

    await session.reset()
    result = await session.screenshot()

    assert result["target_id"] == "home"
    assert active[0] is home
    assert tabs == [home]
    old.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_back_without_history_returns_tool_error(tmp_path: Path) -> None:
    tab = FakeTab("detail", "visible")
    tab.send = AsyncMock(return_value=(0, []))  # type: ignore[method-assign]
    session, _ = make_session(tmp_path, [tab])

    result = await driver._execute_tool_request(
        session, {"browser_back": session.back},
        {"request_id": 43, "tool": "browser_back", "args": {}}, timeout=5,
    )

    assert result["ok"] is False
    assert result["error_type"] == "RuntimeError"
    assert "No previous page" in result["error"]


@pytest.mark.asyncio
async def test_driver_timeout_returns_error_with_matching_request_id(tmp_path: Path) -> None:
    session, _ = make_session(tmp_path, [FakeTab("chat", "visible")])

    async def blocked() -> dict[str, object]:
        await asyncio.Event().wait()
        return {}

    result = await driver._execute_tool_request(
        session, {"browser_wait": blocked},
        {"request_id": 42, "tool": "browser_wait", "args": {}},
        timeout=0.01,
    )

    assert result["request_id"] == 42
    assert result["tool"] == "browser_wait"
    assert result["error_type"] == "BrowserToolTimeout"


@pytest.mark.asyncio
async def test_visible_screenshot_timeout_requires_browser_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "SCREENSHOT_TIMEOUT_SECONDS", 0.01)
    session, _ = make_session(tmp_path, [FakeTab("chat", "visible", stuck=True)])

    with pytest.raises(driver.BrowserToolTimeout, match="target chat"):
        await session.screenshot()
