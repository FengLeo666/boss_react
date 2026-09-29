from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from boss_react.agent import build_agent, build_initial_messages, build_model
from boss_react.agent_config import load_agent_settings
from boss_react.cli import display_checkpoint_history, resolve_task
from boss_react.context_compaction import (
    COMPACTION_PROMPT,
    AgentNodeCompactionMiddleware,
    resume_message,
    task_message,
)
from boss_react.nodriver_middleware import NodriverBrowserMiddleware


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = load_agent_settings(_write_config(tmp_path))

    async def fake_prompt() -> str:
        return "Find Agent roles\r\nOnly campus recruitment"

    monkeypatch.setattr(
        "boss_react.cli._read_enhanced_task",
        fake_prompt,
    )

    assert await resolve_task(settings) == "Find Agent roles\nOnly campus recruitment"


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

    browser = NodriverBrowserMiddleware(session=Session())
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
