from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import RunControl

from boss_react import cli as cli_module
from boss_react.agent import build_agent, build_initial_messages, build_model
from boss_react.agent_config import load_agent_settings
from boss_react.cli import display_checkpoint_history, resolve_task, run
from boss_react.context_compaction import (
    COMPACTION_PROMPT,
    AgentNodeCompactionMiddleware,
    resume_message,
    task_message,
)
from boss_react.nodriver_middleware import BossReactMiddleware


def _write_config(tmp_path: Path, *, task: str = "") -> Path:
    (tmp_path / "resume.txt").write_text("Agent platform experience", encoding="utf-8")
    (tmp_path / "system.txt").write_text("Use browser tools carefully.", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "LLM_API_KEY=test-key\nLLM_BASE_URL=https://example.invalid/v1\nLLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    config = tmp_path / "agent.toml"
    config.write_text(
        f"""
[agent]
task = {task!r}
system_prompt_path = "system.txt"
resume_text_path = "resume.txt"

[model]
env_file = ".env"
request_timeout_seconds = 300

[browser]
profile_dir = "browser-profile/nodriver"
action_timeout_seconds = 90

[context]
summary_trigger_tokens = 1000
summary_trigger_images = 255

[checkpoint]
database_path = "data/checkpoints.sqlite"
thread_id = "test-thread"

[runtime]
recursion_limit = 25
debug = false
log_level = "DEBUG"
log_file = "logs/test.log"
""",
        encoding="utf-8",
    )
    return config


def test_settings_resolve_paths_and_context_limits(tmp_path: Path) -> None:
    settings = load_agent_settings(_write_config(tmp_path, task="Inspect the page"))

    assert settings.task == "Inspect the page"
    assert settings.resume_text_path == tmp_path / "resume.txt"
    assert settings.resume_image_path is None
    assert settings.model_env_path == tmp_path / ".env"
    assert settings.browser_profile_path == tmp_path / "browser-profile" / "nodriver"
    assert settings.browser_action_timeout_seconds == 90
    assert settings.request_timeout_seconds == 300
    assert settings.summary_trigger_tokens == 1000
    assert settings.summary_trigger_images == 255
    assert settings.checkpoint_database_path == tmp_path / "data" / "checkpoints.sqlite"
    assert settings.checkpoint_thread_id == "test-thread"
    assert settings.recursion_limit == 25
    assert settings.log_level == "DEBUG"
    assert settings.log_file == tmp_path / "logs" / "test.log"


def test_missing_resume_image_path_is_treated_as_disabled(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    content = config.read_text(encoding="utf-8")
    config.write_text(
        content.replace(
            'resume_text_path = "resume.txt"',
            'resume_text_path = "resume.txt"\nresume_image_path = "resume.jpg"',
        ),
        encoding="utf-8",
    )

    settings = load_agent_settings(config)

    assert settings.resume_image_path is None


def test_existing_resume_image_path_remains_enabled(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    image = tmp_path / "resume.jpg"
    image.write_bytes(b"image")
    content = config.read_text(encoding="utf-8")
    config.write_text(
        content.replace(
            'resume_text_path = "resume.txt"',
            'resume_text_path = "resume.txt"\nresume_image_path = "resume.jpg"',
        ),
        encoding="utf-8",
    )

    settings = load_agent_settings(config)

    assert settings.resume_image_path == image


def test_initial_messages_put_resume_first_and_task_second(tmp_path: Path) -> None:
    settings = load_agent_settings(_write_config(tmp_path))
    messages = build_initial_messages(settings, "Find one matching role")

    assert len(messages) == 2
    assert all(isinstance(message, HumanMessage) for message in messages)
    assert "Agent platform experience" in str(messages[0].content)
    assert messages[1].content == "Find one matching role"


async def test_empty_configured_task_prompts_for_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_agent_settings(_write_config(tmp_path))

    async def fake_prompt() -> str:
        return "  Apply to one suitable role  "

    monkeypatch.setattr(
        "boss_react.cli._read_enhanced_task",
        fake_prompt,
    )

    assert await resolve_task(settings) == "Apply to one suitable role"


async def test_interactive_task_accepts_multiple_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    settings = load_agent_settings(_write_config(tmp_path))

    async def fake_prompt() -> str:
        return "Find Agent roles\r\nOnly campus recruitment"

    monkeypatch.setattr(
        "boss_react.cli._read_enhanced_task",
        fake_prompt,
    )

    assert await resolve_task(settings) == "Find Agent roles\nOnly campus recruitment"
    assert "[输入] 已收到。" in capsys.readouterr().out


async def test_cli_override_wins_over_configured_task(tmp_path: Path) -> None:
    settings = load_agent_settings(_write_config(tmp_path, task="Configured task"))

    assert await resolve_task(settings, "CLI task") == "CLI task"


def test_checkpoint_history_is_displayed_without_tool_payloads(capsys: pytest.CaptureFixture) -> None:
    checkpoint = SimpleNamespace(
        checkpoint={
            "channel_values": {
                "messages": [
                    HumanMessage(
                        content="private resume body",
                        additional_kwargs={"boss_react_role": "resume"},
                    ),
                    HumanMessage(
                        content="Find a suitable role",
                        additional_kwargs={"boss_react_role": "task"},
                    ),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "browser_state",
                                "args": {},
                                "id": "tool-1",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="large screenshot payload",
                        tool_call_id="tool-1",
                    ),
                    AIMessage(content="Completed one action"),
                ]
            }
        }
    )

    display_checkpoint_history(checkpoint)
    output = capsys.readouterr().out

    assert "[已加载简历资料" in output
    assert "private resume body" not in output
    assert "Find a suitable role" in output
    assert "Completed one action" in output
    assert "large screenshot payload" not in output
    assert "browser_state" not in output
    assert "工具" not in output


def test_checkpoint_history_renders_markdown_in_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TerminalBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalBuffer()
    monkeypatch.setattr(sys, "stdout", output)
    checkpoint = SimpleNamespace(checkpoint={"channel_values": {"messages": [
        HumanMessage(content="## 目标\n\n**校招**岗位"),
        AIMessage(content="找到 `Agent` 职位"),
        ToolMessage(content="screenshot payload", tool_call_id="tool-1"),
    ]}})

    display_checkpoint_history(checkpoint)

    rendered = output.getvalue()
    assert "[用户]" in rendered and "[助手]" in rendered
    assert "目标" in rendered and "校招" in rendered
    assert "**校招**" not in rendered
    assert "screenshot payload" not in rendered


def test_empty_checkpoint_history_prints_nothing(capsys: pytest.CaptureFixture) -> None:
    display_checkpoint_history(None)

    assert capsys.readouterr().out == ""


def test_model_uses_local_openai_compatible_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_agent_settings(_write_config(tmp_path))
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "BOSS_LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)

    model = build_model(settings)

    assert model.model_name == "test-model"
    assert str(model.openai_api_base) == "https://example.invalid/v1"
    assert model.request_timeout == 300


def test_build_agent_installs_compactor_before_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_agent_settings(_write_config(tmp_path))
    model = build_model(settings)

    class Session:
        async def call(self, name: str, **arguments):
            return {"ok": True, "name": name, "arguments": arguments}

        async def close(self):
            return None

        async def start(self):
            return {"event": "ready"}

    browser = BossReactMiddleware(session=Session())
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "compiled-agent"

    monkeypatch.setattr("boss_react.agent.create_agent", fake_create_agent)
    checkpointer = object()
    graph, returned_browser = build_agent(
        settings,
        model=model,
        browser_middleware=browser,
        checkpointer=checkpointer,
    )

    assert graph == "compiled-agent"
    assert returned_browser is browser
    assert isinstance(captured["middleware"][0], AgentNodeCompactionMiddleware)
    assert captured["middleware"][0].trigger_tokens == 1000
    assert captured["middleware"][0].trigger_images == 255
    assert captured["middleware"][1] is browser
    assert captured["checkpointer"] is checkpointer


async def test_compactor_uses_agent_model_node_without_tools_and_rewrites_state() -> None:
    middleware = AgentNodeCompactionMiddleware(trigger_tokens=1)
    resume = resume_message("resume facts")
    prior = AIMessage(content="previous work")
    task = task_message("continue the task")
    original = [resume, prior, task]

    before_update = middleware.before_agent({"messages": original}, None)

    assert before_update is not None
    compaction_request = before_update["messages"][0]
    assert isinstance(compaction_request, HumanMessage)
    assert compaction_request.content == COMPACTION_PROMPT

    seen_request = None

    async def handler(request):
        nonlocal seen_request
        seen_request = request
        return ModelResponse(result=[AIMessage(content="compressed history")])

    request = ModelRequest(
        model=object(),
        messages=[*original, compaction_request],
        tools=[{"type": "function", "function": {"name": "browser_state"}}],
        tool_choice="auto",
        state={"messages": [*original, compaction_request]},
    )
    response = await middleware.awrap_model_call(request, handler)

    assert response.result[0].content == "compressed history"
    assert seen_request.tools == []
    assert seen_request.tool_choice is None

    after_update = middleware.after_model(
        {"messages": [*original, compaction_request, response.result[0]]}, None
    )
    assert after_update is not None
    assert after_update["jump_to"] == "model"
    rewritten = after_update["messages"]
    assert isinstance(rewritten[0], RemoveMessage)
    assert rewritten[0].id == REMOVE_ALL_MESSAGES
    assert rewritten[1] is resume
    assert rewritten[2] is task
    assert rewritten[3].content == "以下是压缩后的历史上下文：\n\ncompressed history"


def test_compactor_triggers_at_255_images_without_token_limit() -> None:
    middleware = AgentNodeCompactionMiddleware(trigger_tokens=1_000_000, trigger_images=255)
    images = [ToolMessage(content=[{"type": "image", "base64": "a"}], tool_call_id=str(i)) for i in range(255)]
    messages = [resume_message("resume"), task_message("task"), *images]

    assert middleware.before_agent({"messages": messages[:-1]}, None) is None
    update = middleware.before_agent({"messages": messages}, None)
    assert update is not None
    assert update["messages"][0].content == COMPACTION_PROMPT
    assert middleware.before_agent({"messages": [*messages, *update["messages"]]}, None) is None


async def test_compactor_rejects_tool_calls_from_compression_model_node() -> None:
    middleware = AgentNodeCompactionMiddleware(trigger_tokens=1)
    compaction_request = middleware.before_agent(
        {"messages": [resume_message("resume"), task_message("task")]}, None
    )["messages"][0]

    async def handler(request):
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "browser_state",
                            "args": {},
                            "id": "tool-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )

    request = ModelRequest(
        model=object(),
        messages=[resume_message("resume"), task_message("task"), compaction_request],
        tools=[],
        state={"messages": []},
    )
    with pytest.raises(RuntimeError, match="禁止工具调用"):
        await middleware.awrap_model_call(request, handler)


async def test_compactor_returns_to_agent_model_after_rewriting_context() -> None:
    model = FakeMessagesListChatModel(
        responses=[
            AIMessage(content="compressed history"),
            AIMessage(content="finished current task"),
        ]
    )
    agent = create_agent(
        model=model,
        middleware=[AgentNodeCompactionMiddleware(trigger_tokens=1)],
    )

    result = await agent.ainvoke(
        {"messages": [resume_message("resume facts"), task_message("current task")]}
    )

    assert len(result["messages"]) == 4
    assert result["messages"][0].content == "resume facts"
    assert result["messages"][1].content == "current task"
    assert "compressed history" in result["messages"][2].content
    assert result["messages"][3].content == "finished current task"


async def test_manual_compaction_stops_before_normal_agent_work_and_can_resume(tmp_path: Path) -> None:
    config = {"configurable": {"thread_id": "manual-compact-test"}}
    database = tmp_path / "checkpoints.sqlite"
    middleware = AgentNodeCompactionMiddleware(trigger_tokens=1_000_000)

    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        await saver.setup()
        first = create_agent(
            FakeMessagesListChatModel(responses=[AIMessage(content="first run complete")]),
            middleware=[middleware], checkpointer=saver,
        )
        await first.ainvoke(
            {"messages": [resume_message("resume"), task_message("find work")], "manual_compact": False},
            config=config,
        )

        compression_model = FakeMessagesListChatModel(responses=[
            AIMessage(content="compressed history"), AIMessage(content="should not run"),
        ])
        compact = create_agent(compression_model, middleware=[middleware], checkpointer=saver)
        result = await compact.ainvoke({"manual_compact": True}, config=config)
        assert compression_model.i == 1
        assert result["manual_compact"] is False
        assert [message.content for message in result["messages"]] == [
            "resume", "find work", "以下是压缩后的历史上下文：\n\ncompressed history",
        ]

        resumed = create_agent(
            FakeMessagesListChatModel(responses=[AIMessage(content="continued")]),
            middleware=[middleware], checkpointer=saver,
        )
        continuation = await resumed.ainvoke({"messages": [task_message("new human input")]}, config=config)
        assert continuation["messages"][-1].content == "continued"


async def test_cli_loops_after_agent_end_and_manual_compaction_waits_for_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, task="first task")
    calls: list[dict] = []
    inputs = iter(["/compact", "second task", "/exit"])

    class Saver:
        def __init__(self):
            self.messages = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def setup(self):
            return None

        async def aget_tuple(self, _config):
            if not self.messages:
                return None
            return SimpleNamespace(
                checkpoint={"channel_values": {"messages": self.messages}},
                config={"configurable": {"checkpoint_id": "test"}},
            )

    saver = Saver()

    class Graph:
        async def ainvoke(self, update, config, **_kwargs):
            calls.append(update)
            if update.get("manual_compact"):
                saver.messages = [*saver.messages[:2], HumanMessage(content="summary")]
            else:
                saver.messages = [*saver.messages, *update["messages"], AIMessage(content="done")]
            return {"messages": saver.messages}

        async def astream(self, update, config, **_kwargs):
            result = await self.ainvoke(update, config)
            yield {"type": "updates", "data": {"model": {"messages": [result["messages"][-1]]}}}

    class Browser:
        closed = False

        async def aclose(self):
            self.closed = True

    browser = Browser()

    async def next_input():
        return next(inputs)

    monkeypatch.setattr("boss_react.cli.configure_logging", lambda *_args: None)
    monkeypatch.setattr(
        cli_module, "_load_active_chat", AsyncMock(side_effect=lambda _saver, default: default)
    )
    monkeypatch.setattr("boss_react.cli.AsyncSqliteSaver.from_conn_string", lambda *_args: saver)
    monkeypatch.setattr("boss_react.cli.build_agent", lambda *_args, **_kwargs: (Graph(), browser))
    monkeypatch.setattr("boss_react.cli._read_next_input", next_input)

    code = await run(argparse.Namespace(config=config_path, task=None))

    assert code == 0
    assert len(calls) == 3
    assert calls[0]["messages"][1].content == "first task"
    assert calls[1] == {"manual_compact": True}
    assert calls[2]["messages"][0].content == "second task"
    assert browser.closed is True


async def test_escape_key_requests_graph_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    import msvcrt

    monkeypatch.setattr(
        cli_module, "sys",
        SimpleNamespace(platform="win32", stdin=SimpleNamespace(isatty=lambda: True)),
    )
    monkeypatch.setattr(msvcrt, "kbhit", lambda: True)
    monkeypatch.setattr(msvcrt, "getwch", lambda: "\x1b")
    control = RunControl()

    await cli_module._watch_escape(control)

    assert control.drain_requested
    assert control.drain_reason == "user_escape"


async def test_graph_drain_saves_checkpoint_and_can_resume() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first(_state):
        entered.set()
        await release.wait()
        return {"value": 1}

    async def second(_state):
        return {"value": 2}

    builder = StateGraph(dict)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "escape-test"}}
    control = RunControl()
    run_task = asyncio.create_task(graph.ainvoke({"value": 0}, config, control=control))

    await entered.wait()
    control.request_drain("user_escape")
    release.set()
    with pytest.raises(GraphDrained):
        await run_task

    state = await graph.aget_state(config)
    assert state.values["value"] == 1
    assert state.next == ("second",)
    assert (await graph.ainvoke(None, config))["value"] == 2


async def test_new_cli_input_after_drain_does_not_execute_pending_tool() -> None:
    control = RunControl()
    tool_calls = []

    @tool
    def pending_action() -> str:
        """A pending action that should not run after user interruption."""
        tool_calls.append(True)
        return "done"

    class ToolModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, *args, **kwargs):
            if self.i == 0:
                control.request_drain("user_escape")
            return super()._generate(*args, **kwargs)

    model = ToolModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "pending_action", "args": {}, "id": "call-1"}]),
        AIMessage(content="new task acknowledged"),
    ])
    graph = create_agent(model, tools=[pending_action], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "pending-action-test"}}

    with pytest.raises(GraphDrained):
        await graph.ainvoke({"messages": [HumanMessage(content="old task")]}, config, control=control)
    assert (await graph.aget_state(config)).next == ("tools",)

    await cli_module._close_pending_tool_calls(graph, config)
    saved = await graph.aget_state(config)
    assert saved.next == ("model",)
    assert isinstance(saved.values["messages"][-1], ToolMessage)
    assert saved.values["messages"][-1].tool_call_id == "call-1"

    result = await graph.ainvoke({"messages": [HumanMessage(content="new task")]}, config)
    assert result["messages"][-1].content == "new task acknowledged"
    assert not tool_calls


async def test_cli_escape_returns_to_input_and_stops_forever_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    config_path = _write_config(tmp_path, task="/run forever")
    calls: list[dict] = []
    inputs = iter(["second task", "/exit"])

    class Saver:
        messages = [resume_message("resume"), task_message("original task")]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def setup(self):
            return None

        async def aget_tuple(self, _config):
            return SimpleNamespace(
                checkpoint={"channel_values": {"messages": self.messages}},
                config={"configurable": {"checkpoint_id": "test"}},
            )

    saver = Saver()

    class Graph:
        async def aget_state(self, config):
            return SimpleNamespace(next=(), values={"messages": saver.messages})

        async def astream(self, update, config, *, control, **_kwargs):
            calls.append(update)
            saver.messages = [*saver.messages, *update["messages"]]
            if len(calls) == 1:
                while not control.drain_requested:
                    await asyncio.sleep(0.01)
                raise GraphDrained("user_escape")
            saver.messages.append(AIMessage(content="done"))
            yield {"type": "updates", "data": {"model": {"messages": [saver.messages[-1]]}}}

    class Browser:
        async def aclose(self):
            return None

    watch_count = 0

    async def watch_escape(control):
        nonlocal watch_count
        watch_count += 1
        if watch_count == 1:
            control.request_drain("user_escape")
        else:
            await asyncio.Event().wait()

    async def next_input():
        return next(inputs)

    monkeypatch.setattr(cli_module, "configure_logging", lambda *_args: None)
    monkeypatch.setattr(
        cli_module, "_load_active_chat", AsyncMock(side_effect=lambda _saver, default: default)
    )
    monkeypatch.setattr(cli_module.AsyncSqliteSaver, "from_conn_string", lambda *_args: saver)
    monkeypatch.setattr(cli_module, "build_agent", lambda *_args, **_kwargs: (Graph(), Browser()))
    monkeypatch.setattr(cli_module, "_watch_escape", watch_escape)
    monkeypatch.setattr(cli_module, "_read_next_input", next_input)

    assert await run(argparse.Namespace(config=config_path, task=None)) == 0
    assert len(calls) == 2
    assert calls[0]["messages"][0].content.startswith("继续执行")
    assert calls[1]["messages"][0].content == "second task"
    assert "已暂停，返回输入。" in capsys.readouterr().out


async def test_cli_run_forever_reinvokes_without_more_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, task="/run forever")
    calls: list[dict] = []

    class Saver:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def setup(self):
            return None

        async def aget_tuple(self, _config):
            return SimpleNamespace(
                checkpoint={"channel_values": {"messages": [resume_message("resume"), task_message("original task")]}},
                config={"configurable": {"checkpoint_id": "test"}},
            )

    class Graph:
        async def ainvoke(self, update, config):
            calls.append(update)
            if len(calls) == 2:
                raise asyncio.CancelledError
            return {"messages": [AIMessage(content="done")]}

        async def astream(self, update, config, **_kwargs):
            result = await self.ainvoke(update, config)
            yield {"type": "updates", "data": {"model": {"messages": result["messages"]}}}

    class Browser:
        closed = False

        async def aclose(self):
            self.closed = True

    browser = Browser()

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr("boss_react.cli.configure_logging", lambda *_args: None)
    monkeypatch.setattr(
        cli_module, "_load_active_chat", AsyncMock(side_effect=lambda _saver, default: default)
    )
    monkeypatch.setattr("boss_react.cli.AsyncSqliteSaver.from_conn_string", lambda *_args: Saver())
    monkeypatch.setattr("boss_react.cli.build_agent", lambda *_args, **_kwargs: (Graph(), browser))
    monkeypatch.setattr("boss_react.cli.asyncio.sleep", no_delay)

    with pytest.raises(asyncio.CancelledError):
        await run(argparse.Namespace(config=config_path, task=None))

    assert len(calls) == 2
    assert all(call["messages"][0].content.startswith("继续执行") for call in calls)
    assert browser.closed is True


async def test_cli_new_switch_and_list_chats_use_separate_sqlite_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture,
) -> None:
    config_path = _write_config(tmp_path, task="/new chat abc")
    inputs = iter([
        "/chats", "first task", "/new chat xyz", "/chats",
        "/new chat abc", "/switch chat missing", "second task",
        "/switch chat abc", "third task", "/switch chat xyz", "/exit",
    ])

    class ToolCallingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class Session:
        async def start(self):
            return {"event": "ready"}

        async def call(self, name, **kwargs):
            return {"ok": True, "logged_in": True, "page_state": {"kind": "boss_home"}}

        async def close(self):
            return None

    browser = BossReactMiddleware(session=Session())

    def build_fake_agent(_settings, *, checkpointer):
        graph = create_agent(
            ToolCallingModel(responses=[AIMessage(content="done")]),
            middleware=[AgentNodeCompactionMiddleware(trigger_tokens=1000), browser],
            checkpointer=checkpointer,
        )
        return graph, browser

    async def next_input():
        return next(inputs)

    monkeypatch.setattr("boss_react.cli.configure_logging", lambda *_args: None)
    monkeypatch.setattr("boss_react.cli.build_agent", build_fake_agent)
    monkeypatch.setattr("boss_react.cli._read_next_input", next_input)

    assert await run(argparse.Namespace(config=config_path, task=None)) == 0
    output = capsys.readouterr().out
    assert "当前聊天: abc" in output
    assert "当前聊天: xyz" in output
    assert "* abc" in output
    assert "* xyz" in output
    assert "  abc" in output
    assert "聊天 abc 已存在" in output
    assert "聊天 missing 不存在" in output

    settings = load_agent_settings(config_path)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_database_path)) as saver:
        await saver.setup()
        abc = await saver.aget_tuple({"configurable": {"thread_id": "abc"}})
        xyz = await saver.aget_tuple({"configurable": {"thread_id": "xyz"}})
    abc_text = [m.content for m in abc.checkpoint["channel_values"]["messages"] if isinstance(m, HumanMessage)]
    xyz_text = [m.content for m in xyz.checkpoint["channel_values"]["messages"] if isinstance(m, HumanMessage)]
    assert "first task" in abc_text and "third task" in abc_text
    assert "second task" not in abc_text
    assert "second task" in xyz_text
    assert "first task" not in xyz_text

    assert await run(argparse.Namespace(config=config_path, task="/exit")) == 0
    assert "会话  xyz" in capsys.readouterr().out

    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_database_path)) as saver:
        await saver.setup()
        await saver.conn.execute(
            "UPDATE boss_react_cli_state SET value = ? WHERE key = ?",
            ("deleted-chat", "active_thread_id"),
        )
        await saver.conn.commit()

    assert await run(argparse.Namespace(config=config_path, task="/exit")) == 0
    assert "会话  test-thread" in capsys.readouterr().out
