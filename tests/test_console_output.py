from __future__ import annotations

import io
import logging
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langgraph.runtime import RunControl

from boss_react.cli import _stream_agent
from boss_react.console_output import finish_model_text, stream_model_text, tool_finished, tool_started
from boss_react.logging_config import configure_logging


class TerminalBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_interactive_model_output_renders_markdown_and_finishes_before_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = TerminalBuffer()
    monkeypatch.setattr(sys, "stdout", output)

    stream_model_text("## 结论\n\n**推")
    stream_model_text("荐**这个岗位")
    tool_started("browser_click_text", {"text": "职位"})

    rendered = output.getvalue()
    assert "[模型]" in rendered
    assert "结论" in rendered
    assert "推荐" in rendered
    assert "**推荐**" not in rendered
    assert "[工具] 点击  '职位'" in rendered
    assert rendered.index("推荐") < rendered.index("[工具] 点击")


def test_console_shows_tool_actions_and_bounded_results_without_payloads(capsys) -> None:
    script = "return 'private JavaScript source';"
    tool_started("browser_eval_js", {"name": "find_jobs", "script": script})
    result = ToolMessage(
        content='{"ok": true, "result": "' + "job " * 100 + '"}',
        tool_call_id="js-1",
    )
    tool_finished(
        "browser_eval_js", result,
        {"path": "C:/shots/shot.png", "page_state": {"kind": "job_detail"}, "network_idle": True},
        1.7,
    )

    output = capsys.readouterr().out
    assert "[工具] 运行 JavaScript" not in output
    assert "[工具结果]" in output
    assert "职位详情" in output
    assert "shot.png" in output
    assert script not in output
    assert len(output) < 400


def test_console_shows_failed_tool_without_image_payload(capsys) -> None:
    result = ToolMessage(content="No element matched", tool_call_id="click-1", status="error")

    tool_finished("browser_click_text", result, None, 0.2)

    output = capsys.readouterr().out
    assert "失败: No element matched" in output
    assert "截图不可用" in output


def test_detailed_logging_goes_to_file_not_terminal(tmp_path: Path, capsys) -> None:
    root = logging.getLogger()
    previous_handlers = list(root.handlers)
    previous_level = root.level
    path = tmp_path / "agent.log"
    try:
        configure_logging(path)
        logging.getLogger("boss_react.test").info("detailed diagnostic")
        assert "detailed diagnostic" in path.read_text(encoding="utf-8")
        assert "detailed diagnostic" not in capsys.readouterr().err
    finally:
        for handler in list(root.handlers):
            if handler not in previous_handlers:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(previous_level)


async def test_agent_text_streams_without_tool_arguments_or_duplicate_final(capsys) -> None:
    class Graph:
        async def astream(self, update, config, **kwargs):
            assert kwargs == {
                "stream_mode": ["messages", "updates"], "version": "v2", "control": control,
            }
            yield {"type": "messages", "data": (
                AIMessageChunk(content="你好"), {"langgraph_node": "model"},
            )}
            yield {"type": "messages", "data": (
                AIMessageChunk(content=[{"type": "tool_call_chunk", "args": "secret"}]),
                {"langgraph_node": "model"},
            )}
            yield {"type": "messages", "data": (
                AIMessageChunk(content="，老师"), {"langgraph_node": "model"},
            )}
            yield {"type": "updates", "data": {"model": {
                "messages": [AIMessage(content="你好，老师")],
            }}}

    control = RunControl()
    assert await _stream_agent(Graph(), {}, {}, control) == "你好，老师"
    output = capsys.readouterr().out
    assert output == "[模型] 你好，老师\n"
    assert "secret" not in output


def test_tool_output_starts_on_new_line_after_streamed_text(capsys) -> None:
    stream_model_text("准备点击")
    tool_started("browser_click_text", {"text": "职位"})
    finish_model_text()

    assert capsys.readouterr().out == "[模型] 准备点击\n[工具] 点击  '职位'\n"
