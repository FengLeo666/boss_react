from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from boss_react import NodriverBrowserConfig, BossReactMiddleware
from boss_react.nodriver import NodriverToolError
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from boss_react.nodriver_middleware import _normalize_patterns


class FakeNodriverSession:
    def __init__(self, screenshot_path: Path) -> None:
        self.screenshot_path = screenshot_path
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.started = False
        self.closed = False

    async def start(self) -> dict[str, Any]:
        self.started = True
        return {"event": "ready"}

    async def close(self) -> None:
        self.closed = True

    async def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        self.calls.append((name, arguments))
        if name == "browser_ensure_login":
            return {
                "ok": True,
                "logged_in": False,
                "redirected": True,
                "reasons": ["headerLoginVisible"],
                "page_state": {"kind": "login"},
            }
        if name in {"browser_screenshot", "browser_settle_and_screenshot"}:
            return {
                "ok": True,
                "path": str(self.screenshot_path),
                "url": "https://www.zhipin.com/job_detail/secret.html",
                "page_state": {"kind": "job_detail"},
                "network_idle": True,
            }
        if name == "browser_observe":
            return {
                "ok": True,
                "url": "https://www.zhipin.com/jobs",
                "elements": [{"text": "立即沟通", "href": "https://www.zhipin.com/chat"}],
                "text": "详情见 https://www.zhipin.com/job/1",
            }
        return {"ok": True, "tool": name, "arguments": arguments}


@pytest.fixture
def fake_middleware(tmp_path: Path) -> tuple[BossReactMiddleware, FakeNodriverSession]:
    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"png-bytes")
    session = FakeNodriverSession(screenshot)
    return BossReactMiddleware(session=session), session


def test_nodriver_config_resolves_local_project_defaults() -> None:
    config = NodriverBrowserConfig()
    project_root = Path(__file__).resolve().parents[1]

    assert config.python_executable == project_root / ".venv" / "Scripts" / "python.exe"
    assert config.profile_dir == project_root / "browser-profile" / "nodriver"
    assert config.start_url == "https://www.zhipin.com/"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["搜索", "职位"], ["搜索", "职位"]),
        ('["搜索", "职位"]', ["搜索", "职位"]),
        ("搜索，职位", ["搜索", "职位"]),
        ("", None),
        (None, None),
    ],
)
def test_normalize_patterns_accepts_model_compatible_shapes(raw, expected) -> None:
    assert _normalize_patterns(raw) == expected


def test_middleware_registers_all_unrestricted_tools(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, _ = fake_middleware
    names = {item.name for item in middleware.tools}

    assert names == {
        "browser_state", "browser_observe", "browser_click_text",
        "browser_hover_text", "browser_input_text", "browser_scroll", "browser_scroll_to_text",
        "browser_press", "browser_back", "browser_reset", "browser_wait", "browser_eval_js",
        "browser_upload_text", "boss_send_greeting",
    }


@pytest.mark.asyncio
async def test_text_click_and_javascript_are_forwarded_without_filtering(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    tools = {item.name: item for item in middleware.tools}
    script = "document.body.innerHTML = ''; return location.href"

    await tools["browser_click_text"].ainvoke({
        "text": "立即沟通", "match": "contains", "scope_text": "百度", "occurrence": 2,
    })
    await tools["browser_eval_js"].ainvoke({"name": "clear_page", "script": script})

    assert session.calls[0] == (
        "browser_click_text",
        {
            "text": "立即沟通", "match": "contains", "role": None, "scope_text": "百度",
            "occurrence": 2, "wait_after_ms": 750,
        },
    )
    assert session.calls[1] == ("browser_eval_js", {"script": script})


@pytest.mark.asyncio
async def test_greeting_tool_remains_available_and_forwards_calls(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    tools = {item.name: item for item in middleware.tools}

    await tools["boss_send_greeting"].ainvoke({"message": "老师您好"})

    assert session.calls == [
        ("boss_send_greeting", {"message": "老师您好"}),
    ]


async def test_configured_resume_image_exposes_zero_argument_send_tool(tmp_path: Path) -> None:
    resume_image = tmp_path / "resume.png"
    resume_image.write_bytes(b"png")
    session = FakeNodriverSession(tmp_path / "shot.png")
    middleware = BossReactMiddleware(
        session=session,
        resume_image_path=resume_image,
    )
    tools = {item.name: item for item in middleware.tools}

    assert tools["boss_send_image"].args == {}
    await tools["boss_send_image"].ainvoke({})

    assert session.calls == [("boss_send_image", {"path": str(resume_image)})]


def test_missing_configured_resume_image_disables_send_tool(tmp_path: Path) -> None:
    middleware = BossReactMiddleware(
        session=FakeNodriverSession(tmp_path / "shot.png"),
        resume_image_path=tmp_path / "missing.png",
    )

    assert middleware.resume_image_path is None
    assert "boss_send_image" not in {tool.name for tool in middleware.tools}


@pytest.mark.asyncio
async def test_state_returns_state_and_screenshot_as_multimodal_content(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    state_tool = next(item for item in middleware.tools if item.name == "browser_state")

    content, artifact = await state_tool.coroutine()

    assert session.calls == [("browser_screenshot", {})]
    assert content[0]["type"] == "text"
    assert "Browser state" in content[0]["text"]
    assert content[1]["type"] == "image"
    assert content[1]["mime_type"] == "image/png"
    assert content[1]["base64"] == "cG5nLWJ5dGVz"
    assert artifact["ok"] is True
    assert "url" not in artifact
    assert "zhipin.com" not in content[0]["text"]


@pytest.mark.asyncio
async def test_observe_forwards_pattern_matching_options(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    observe_tool = next(item for item in middleware.tools if item.name == "browser_observe")

    result = await observe_tool.ainvoke({
        "patterns": '["立即沟通", "职位描述"]',
        "match": "contains",
        "max_results": 12,
        "context_chars": 400,
    })

    assert session.calls == [("browser_observe", {
        "patterns": ["立即沟通", "职位描述"],
        "match": "contains",
        "max_results": 12,
        "context_chars": 400,
    })]
    assert "url" not in result
    assert "href" not in result["elements"][0]
    assert result["text"] == "详情见 [URL hidden]"


@pytest.mark.asyncio
async def test_lifecycle_starts_and_closes_the_retained_session(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware

    await middleware.abefore_agent(None, None)
    await middleware.aclose()

    assert session.started is True
    assert session.calls == [("browser_ensure_login", {})]
    assert session.closed is True


async def test_manual_compaction_does_not_start_browser(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware

    await middleware.abefore_agent({"manual_compact": True, "messages": []}, None)

    assert session.calls == []
    assert session.started is False


@pytest.mark.asyncio
async def test_every_tool_result_gets_one_post_action_screenshot(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    request = SimpleNamespace(tool_call={"id": "call-1", "name": "browser_state", "args": {}})
    original = ToolMessage(
        content=[{"type": "text", "text": "Original state"}, {"type": "image", "base64": "old", "mime_type": "image/png"}],
        tool_call_id="call-1",
    )

    async def handler(_request):
        return original

    result = await middleware.awrap_tool_call(request, handler)

    assert session.calls == [("browser_settle_and_screenshot", {})]
    assert len([block for block in result.content if block["type"] == "image"]) == 1
    assert result.content[-1]["base64"] == "cG5nLWJ5dGVz"
    assert "network_idle" in result.content[-2]["text"]
    assert "zhipin.com" not in str(result.content)


@pytest.mark.asyncio
async def test_failed_tool_also_returns_post_action_screenshot(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    request = SimpleNamespace(tool_call={"id": "call-2", "name": "browser_click_text", "args": {}})

    async def handler(_request):
        raise NodriverToolError("No element matched")

    result = await middleware.awrap_tool_call(request, handler)

    assert result.status == "error"
    assert session.calls == [("browser_settle_and_screenshot", {})]
    assert result.content[-1]["type"] == "image"
    assert "No element matched" in result.content[0]["text"]


@pytest.mark.asyncio
async def test_named_javascript_is_checkpointed_and_reused_after_middleware_restart(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    seen: list[dict[str, Any]] = []

    async def handler(request):
        seen.append(request.tool_call["args"])
        return ToolMessage(content='{"ok": true}', tool_call_id=request.tool_call["id"])

    request = ToolCallRequest(
        tool_call={"name": "browser_eval_js", "args": {"name": "find_jobs", "script": "return 42"}, "id": "first"},
        tool=None, state={"messages": []}, runtime=None,
    )
    saved = await middleware.awrap_tool_call(request, handler)
    assert isinstance(saved, Command)
    assert saved.update["js_cache"] == {"find_jobs": "return 42"}
    assert saved.update["messages"][0].content[-1]["type"] == "image"

    resumed = BossReactMiddleware(session=session)
    replay = ToolCallRequest(
        tool_call={"name": "browser_eval_js", "args": {"name": "find_jobs"}, "id": "second"},
        tool=None, state={"messages": [], "js_cache": saved.update["js_cache"]}, runtime=None,
    )
    result = await resumed.awrap_tool_call(replay, handler)

    assert isinstance(result, ToolMessage)
    assert seen == [
        {"name": "find_jobs", "script": "return 42"},
        {"name": "find_jobs", "script": "return 42"},
    ]
    assert session.calls == [
        ("browser_settle_and_screenshot", {}),
        ("browser_settle_and_screenshot", {}),
    ]


@pytest.mark.asyncio
async def test_unknown_javascript_name_fails_without_running_script(
    fake_middleware: tuple[BossReactMiddleware, FakeNodriverSession],
) -> None:
    middleware, session = fake_middleware
    request = ToolCallRequest(
        tool_call={"name": "browser_eval_js", "args": {"name": "missing"}, "id": "lookup"},
        tool=None, state={"messages": [], "js_cache": {}}, runtime=None,
    )

    async def handler(_request):
        pytest.fail("browser_eval_js should not run without cached code")

    result = await middleware.awrap_tool_call(request, handler)

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "No cached JavaScript" in result.content[0]["text"]
    assert session.calls == [("browser_settle_and_screenshot", {})]


async def test_javascript_cache_survives_sqlite_checkpoint_restart(tmp_path: Path) -> None:
    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    screenshot = tmp_path / "shot.png"
    screenshot.write_bytes(b"png-bytes")
    session = FakeNodriverSession(screenshot)
    config = {"configurable": {"thread_id": "js-cache-test"}}
    database = tmp_path / "checkpoints.sqlite"

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        first_model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "browser_eval_js", "args": {"name": "count_jobs", "script": "return 42"}, "id": "js-1"}]),
            AIMessage(content="saved"),
        ])
        first_agent = create_agent(first_model, middleware=[BossReactMiddleware(session=session)], checkpointer=saver)
        first = await first_agent.ainvoke({"messages": [HumanMessage(content="run named JS")]}, config=config)
        assert first["js_cache"] == {"count_jobs": "return 42"}

        second_model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "browser_eval_js", "args": {"name": "count_jobs"}, "id": "js-2"}]),
            AIMessage(content="reused"),
        ])
        second_agent = create_agent(second_model, middleware=[BossReactMiddleware(session=session)], checkpointer=saver)
        second = await second_agent.ainvoke({"messages": [HumanMessage(content="reuse named JS")]}, config=config)
        assert second["js_cache"] == {"count_jobs": "return 42"}

        update_model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "browser_eval_js", "args": {"name": "count_jobs", "script": "return 99"}, "id": "js-3"}]),
            AIMessage(content="updated"),
        ])
        update_agent = create_agent(update_model, middleware=[BossReactMiddleware(session=session)], checkpointer=saver)
        updated = await update_agent.ainvoke({"messages": [HumanMessage(content="replace named JS")]}, config=config)
        assert updated["js_cache"] == {"count_jobs": "return 99"}

        replay_model = ToolCallingModel(responses=[
            AIMessage(content="", tool_calls=[{"name": "browser_eval_js", "args": {"name": "count_jobs"}, "id": "js-4"}]),
            AIMessage(content="replayed update"),
        ])
        replay_agent = create_agent(replay_model, middleware=[BossReactMiddleware(session=session)], checkpointer=saver)
        final = await replay_agent.ainvoke({"messages": [HumanMessage(content="reuse updated JS")]}, config=config)

    assert final["js_cache"] == {"count_jobs": "return 99"}
    assert [args["script"] for name, args in session.calls if name == "browser_eval_js"] == [
        "return 42", "return 42", "return 99", "return 99",
    ]
