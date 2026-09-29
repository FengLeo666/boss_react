from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from boss_react import nodriver_backend


class FakeTab:
    def __init__(self) -> None:
        self.activate = AsyncMock()
        self.bring_to_front = AsyncMock()


class FakeBrowser:
    def __init__(self, tab: FakeTab) -> None:
        self.tab = tab
        self.get = AsyncMock(return_value=tab)
        self.stop = Mock()


@pytest.mark.asyncio
async def test_open_browser_reuses_initial_page_instead_of_creating_window(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tab = FakeTab()
    browser = FakeBrowser(tab)
    monkeypatch.setattr(nodriver_backend.uc, "start", AsyncMock(return_value=browser))
    monkeypatch.setattr(nodriver_backend.asyncio, "sleep", AsyncMock())

    await nodriver_backend.open_browser("https://www.zhipin.com/", tmp_path)

    browser.get.assert_awaited_once_with("https://www.zhipin.com/")
    tab.activate.assert_awaited_once_with()
    tab.bring_to_front.assert_awaited_once_with()
    await nodriver_backend.shutdown()
