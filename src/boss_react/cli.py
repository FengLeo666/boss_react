"""Command-line entry point for the BOSS ReAct agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from prompt_toolkit import PromptSession

from .agent import build_agent, build_initial_messages
from .agent_config import AgentSettings, load_agent_settings
from .context_compaction import task_message
from .logging_config import configure_logging

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "agent.toml"
logger = logging.getLogger(__name__)
_MAX_HISTORY_TEXT_CHARS = 4_000


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LangChain BOSS browser agent")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", help="Override [agent].task for this run")
    return parser.parse_args()


async def _read_enhanced_task() -> str:
    session: PromptSession[str] = PromptSession()
    return await session.prompt_async(
        "你需要找什么工作？",
        multiline=False,
        wrap_lines=True,
    )


async def resolve_task(settings: AgentSettings, override: str | None = None) -> str:
    task = (override or settings.task).strip()
    if task:
        return task
    try:
        task = await _read_enhanced_task()
    except EOFError:
        task = ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("当前控制台不支持增强粘贴输入，回退为空行提交模式: %s", exc)
        task = _read_task_until_blank_line()
    task = task.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not task:
        raise ValueError("任务不能为空")
    return task


def _read_task_until_blank_line() -> str:
    lines: list[str] = []
    input_prompt = "你需要找什么工作？"
    while True:
        try:
            line = input(input_prompt if not lines else "")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _final_text(result: dict[str, Any]) -> str:
    for message in reversed(result.get("messages", [])):
        if isinstance(message, AIMessage) and message.content:
            if isinstance(message.content, str):
                return message.content
            return str(message.content)
    return "Agent 已结束，但没有返回最终文本。"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""


def display_checkpoint_history(checkpoint: Any | None) -> None:
    if checkpoint is None:
        return

    messages = checkpoint.checkpoint.get("channel_values", {}).get("messages", [])
    if not messages:
        return
    visible: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = str(message.additional_kwargs.get("boss_react_role", ""))
            text = _message_text(message)
            if role == "resume":
                visible.append(("简历", f"[已加载简历资料，共 {len(text)} 字符]"))
            elif role == "compaction_request":
                continue
            elif role == "compacted_context":
                visible.append(("历史摘要", text))
            else:
                visible.append(("用户", text))
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                continue
            text = _message_text(message)
            if text:
                visible.append(("助手", text))

    print("\n=== 历史消息 ===")
    for label, text in visible:
        if len(text) > _MAX_HISTORY_TEXT_CHARS:
            text = text[:_MAX_HISTORY_TEXT_CHARS] + "\n[内容过长，已截断]"
        print(f"\n[{label}]\n{text}")
    print("=== 历史结束 ===\n", flush=True)


def _checkpoint_message_count(checkpoint: Any | None) -> int:
    if checkpoint is None:
        return 0
    values = checkpoint.checkpoint.get("channel_values", {})
    messages = values.get("messages", []) if isinstance(values, dict) else []
    return len(messages) if isinstance(messages, list) else 0


def _checkpoint_id(checkpoint: Any | None) -> str:
    if checkpoint is None:
        return "none"
    configurable = checkpoint.config.get("configurable", {})
    return str(configurable.get("checkpoint_id", "unknown"))


async def run(args: argparse.Namespace) -> int:
    settings = load_agent_settings(args.config)
    configure_logging(settings.log_file, settings.log_level)
    settings.checkpoint_database_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = {
        "recursion_limit": settings.recursion_limit,
        "configurable": {"thread_id": settings.checkpoint_thread_id},
    }
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_database_path)
    ) as checkpointer:
        await checkpointer.setup()
        checkpoint = await checkpointer.aget_tuple(run_config)
        display_checkpoint_history(checkpoint)
        task = await resolve_task(settings, args.task)
        logger.info(
            "启动 Agent: config=%s task_source=%s task_chars=%d recursion_limit=%d",
            settings.config_path,
            "cli" if args.task else "config_or_input",
            len(task),
            settings.recursion_limit,
        )
        started = time.perf_counter()
        restored_messages = _checkpoint_message_count(checkpoint)
        is_resume = restored_messages > 0
        logger.info(
            "检查点已加载: mode=%s database=%s thread_id=%s checkpoint_id=%s "
            "restored_messages=%d",
            "resume" if is_resume else "new",
            settings.checkpoint_database_path,
            settings.checkpoint_thread_id,
            _checkpoint_id(checkpoint),
            restored_messages,
        )
        graph, browser = build_agent(settings, checkpointer=checkpointer)
        new_messages = (
            [task_message(task)]
            if is_resume
            else build_initial_messages(settings, task)
        )
        try:
            result = await graph.ainvoke(
                {"messages": new_messages},
                config=run_config,
            )
            saved_checkpoint = await checkpointer.aget_tuple(run_config)
            logger.info(
                "Agent 正常结束: elapsed=%.2fs messages=%d checkpoint_id=%s",
                time.perf_counter() - started,
                len(result.get("messages", [])),
                _checkpoint_id(saved_checkpoint),
            )
            print(_final_text(result))
            return 0
        except Exception:
            logger.exception("Agent 运行失败: elapsed=%.2fs", time.perf_counter() - started)
            raise
        finally:
            logger.info("正在关闭浏览器会话")
            await browser.aclose()
            logger.info("浏览器会话已关闭")


def main() -> None:
    try:
        raise SystemExit(asyncio.run(run(_parse_args())))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
