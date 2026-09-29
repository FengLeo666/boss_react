from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from boss_react.nodriver import NodriverBrowserSession, NodriverToolError


@pytest.mark.asyncio
async def test_read_response_discards_late_response_from_timed_out_request() -> None:
    session = NodriverBrowserSession()
    session._read_json_line = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {"ok": True, "request_id": 4, "tool": "browser_screenshot"},
            {"ok": True, "request_id": 5, "tool": "browser_wait"},
        ]
    )

    response = await session._read_response(5)

    assert response["request_id"] == 5
    assert response["tool"] == "browser_wait"
    assert session._read_json_line.await_count == 2


@pytest.mark.asyncio
async def test_tool_timeout_restarts_browser_and_tells_model_to_observe() -> None:
    session = NodriverBrowserSession()
    writer = SimpleNamespace(write=lambda data: None, drain=AsyncMock())
    session.process = SimpleNamespace(pid=123, stdin=writer)
    session.start = AsyncMock(return_value={})  # type: ignore[method-assign]
    session.close = AsyncMock()  # type: ignore[method-assign]
    session._read_response = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "ok": False, "request_id": 1, "tool": "browser_wait",
            "error_type": "BrowserToolTimeout", "error": "browser_wait exceeded 5s",
        }
    )

    with pytest.raises(NodriverToolError, match="请先调用 browser_state"):
        await session.call("browser_wait", milliseconds=10_000)

    session.close.assert_awaited_once_with(force=True)
    assert session.start.await_count == 2


@pytest.mark.asyncio
async def test_unresponsive_driver_also_restarts_browser() -> None:
    session = NodriverBrowserSession()
    writer = SimpleNamespace(write=lambda data: None, drain=AsyncMock())
    session.process = SimpleNamespace(pid=123, stdin=writer)
    session.start = AsyncMock(return_value={})  # type: ignore[method-assign]
    session.close = AsyncMock()  # type: ignore[method-assign]
    session._read_response = AsyncMock(side_effect=asyncio.TimeoutError)  # type: ignore[method-assign]

    with pytest.raises(NodriverToolError, match="已强制重启"):
        await session.call("browser_wait")

    session.close.assert_awaited_once_with(force=True)
    assert session.start.await_count == 2
