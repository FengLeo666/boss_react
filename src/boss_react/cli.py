"""Command-line entry point for the BOSS ReAct agent."""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphDrained
from langgraph.runtime import RunControl
from prompt_toolkit import PromptSession

from .agent import build_agent, build_initial_messages
from .agent_config import AgentSettings, load_agent_settings
from .context_compaction import task_message
from .console_output import finish_model_text, print_markdown, show_banner, stream_model_text
from .logging_config import configure_logging
from .shallow_sqlite import AsyncShallowSqliteSaver as AsyncSqliteSaver

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "agent.toml"
logger = logging.getLogger(__name__)
_MAX_HISTORY_TEXT_CHARS = 4_000
_CONTINUE_TASK = "继续执行此前的用户求职任务，从当前状态继续。"
_CHAT_COMMAND = re.compile(r"^/(new|switch) chat (\S+)$", re.IGNORECASE)


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


def _announce_input(task: str) -> None:
    if not task:
        return
    if task.startswith("/"):
        command = task.partition("\n")[0][:80]
        print(f"[命令] 已收到 {command}，开始执行。", flush=True)
    else:
        print("[输入] 已收到。开始处理。", flush=True)


async def resolve_task(settings: AgentSettings, override: str | None = None) -> str:
    task = (override or settings.task).strip()
    if task:
        _announce_input(task)
        return task
    task = await _read_next_input()
    if not task:
        raise ValueError("任务不能为空")
    return task


async def _read_next_input() -> str:
    try:
        task = await _read_enhanced_task()
    except EOFError:
        _announce_input("/exit")
        return "/exit"
    except Exception as exc:  # noqa: BLE001
        logger.warning("当前控制台不支持增强粘贴输入，回退为空行提交模式: %s", exc)
        task = _read_task_until_blank_line()
    task = task.replace("\r\n", "\n").replace("\r", "\n").strip()
    _announce_input(task)
    return task


def _read_task_until_blank_line() -> str:
    lines: list[str] = []
    input_prompt = "你需要找什么工作？"
    while True:
        try:
            line = input(input_prompt if not lines else "")
        except EOFError:
            return "/exit" if not lines else "\n".join(lines).strip()
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


def _text_delta(message: Any) -> str:
    if not isinstance(message, AIMessage):
        return ""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


async def _stream_agent(
    graph: Any, update: dict[str, Any], config: dict[str, Any], control: RunControl
) -> str:
    """Stream visible model text while retaining the last final answer for fallback."""
    model_text = ""
    final_text = ""
    try:
        async for part in graph.astream(
            update, config=config, stream_mode=["messages", "updates"], version="v2",
            control=control,
        ):
            if part.get("type") == "messages":
                message, metadata = part["data"]
                if metadata.get("langgraph_node") != "model":
                    continue
                delta = _text_delta(message)
                if delta:
                    stream_model_text(delta)
                    model_text += delta
            elif part.get("type") == "updates":
                model_update = part.get("data", {}).get("model", {})
                for message in model_update.get("messages", []):
                    if isinstance(message, AIMessage):
                        if not message.tool_calls:
                            final_text = _text_delta(message)
                            if final_text and final_text != model_text:
                                if final_text.startswith(model_text):
                                    stream_model_text(final_text[len(model_text):])
                                else:
                                    finish_model_text()
                                    stream_model_text(final_text)
                        model_text = ""
                        finish_model_text()
    finally:
        finish_model_text()
    return final_text


async def _watch_escape(control: RunControl) -> None:
    if sys.platform != "win32" or not sys.stdin.isatty():
        logger.warning("当前控制台无法监听 Esc 键")
        return
    import msvcrt

    while not control.drain_requested:
        try:
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == "\x1b":
                    control.request_drain("user_escape")
                    print("\n[中断] 已请求暂停，等待当前步骤完成...", flush=True)
                    return
                if key in {"\x00", "\xe0"} and msvcrt.kbhit():
                    msvcrt.getwch()
        except OSError as exc:
            logger.warning("Esc 监听失败: %s", exc)
            return
        await asyncio.sleep(0.05)


@asynccontextmanager
async def _escape_monitor() -> AsyncIterator[RunControl]:
    control = RunControl()
    listener = asyncio.create_task(_watch_escape(control))
    try:
        yield control
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


async def _close_pending_tool_calls(graph: Any, config: dict[str, Any]) -> None:
    snapshot = await graph.aget_state(config)
    if "tools" not in snapshot.next:
        return
    messages = snapshot.values.get("messages", [])
    last_ai = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
    if last_ai is None or not last_ai.tool_calls:
        return
    replies = [
        ToolMessage(
            content="用户按 Esc 中断，工具未执行。",
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )
        for call in last_ai.tool_calls
    ]
    await graph.aupdate_state(config, {"messages": replies}, as_node="tools")
    logger.info("已取消待执行工具调用: count=%d", len(replies))


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
        print(f"\n[{label}]", flush=True)
        print_markdown(text)
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


async def _list_chat_ids(checkpointer: AsyncSqliteSaver) -> list[str]:
    async with checkpointer.conn.execute(
        "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
    ) as cursor:
        rows = await cursor.fetchall()
    return [str(row[0]) for row in rows]


async def _load_active_chat(checkpointer: AsyncSqliteSaver, default_thread_id: str) -> str:
    await checkpointer.conn.execute(
        "CREATE TABLE IF NOT EXISTS boss_react_cli_state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    await checkpointer.conn.commit()
    async with checkpointer.conn.execute(
        "SELECT value FROM boss_react_cli_state WHERE key = ?", ("active_thread_id",)
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return default_thread_id
    selected = str(row[0])
    if selected in await _list_chat_ids(checkpointer):
        return selected
    logger.warning("已保存的聊天不存在，回退到默认会话: thread_id=%s", selected)
    return default_thread_id


async def _save_active_chat(checkpointer: AsyncSqliteSaver, thread_id: str) -> None:
    await checkpointer.conn.execute(
        "INSERT INTO boss_react_cli_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        ("active_thread_id", thread_id),
    )
    await checkpointer.conn.commit()


async def run(args: argparse.Namespace) -> int:
    settings = load_agent_settings(args.config)
    configure_logging(settings.log_file, settings.log_level)
    settings.checkpoint_database_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(
        str(settings.checkpoint_database_path)
    ) as checkpointer:
        await checkpointer.setup()
        current_thread_id = await _load_active_chat(checkpointer, settings.checkpoint_thread_id)
        show_banner(current_thread_id, settings.log_file)
        run_config = {
            "recursion_limit": settings.recursion_limit,
            "configurable": {"thread_id": current_thread_id},
        }
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
        restored_messages = _checkpoint_message_count(checkpoint)
        logger.info(
            "检查点已加载: mode=%s database=%s thread_id=%s checkpoint_id=%s "
            "restored_messages=%d",
            "resume" if restored_messages else "new",
            settings.checkpoint_database_path,
            current_thread_id,
            _checkpoint_id(checkpoint),
            restored_messages,
        )
        graph, browser = build_agent(settings, checkpointer=checkpointer)
        forever = False
        try:
            while True:
                command = task.lower()
                if command == "/exit":
                    return 0
                if not task:
                    task = await _read_next_input()
                    continue
                if command == "/chats":
                    chat_ids = await _list_chat_ids(checkpointer)
                    if current_thread_id not in chat_ids:
                        chat_ids.append(current_thread_id)
                        chat_ids.sort()
                    print("\n".join(
                        f"{'* ' if chat_id == current_thread_id else '  '}{chat_id}"
                        for chat_id in chat_ids
                    ), flush=True)
                    task = await _read_next_input()
                    continue
                chat_command = _CHAT_COMMAND.fullmatch(task)
                if chat_command:
                    action, chat_id = chat_command.group(1).lower(), chat_command.group(2)
                    chat_ids = await _list_chat_ids(checkpointer)
                    if action == "new" and chat_id in chat_ids:
                        print(f"聊天 {chat_id} 已存在；请用 /switch chat {chat_id}。", flush=True)
                    elif action == "switch" and chat_id not in chat_ids:
                        print(f"聊天 {chat_id} 不存在；请用 /new chat {chat_id}。", flush=True)
                    else:
                        next_config = {
                            "recursion_limit": settings.recursion_limit,
                            "configurable": {"thread_id": chat_id},
                        }
                        if action == "new":
                            await graph.aupdate_state(next_config, {"messages": []})
                        await _save_active_chat(checkpointer, chat_id)
                        current_thread_id = chat_id
                        run_config = next_config
                        checkpoint = await checkpointer.aget_tuple(run_config)
                        restored_messages = _checkpoint_message_count(checkpoint)
                        logger.info("切换聊天: action=%s thread_id=%s messages=%d", action, chat_id, restored_messages)
                        print(f"当前聊天: {chat_id}", flush=True)
                        display_checkpoint_history(checkpoint)
                    task = await _read_next_input()
                    continue
                if command.startswith(("/new chat", "/switch chat")):
                    print("用法：/new chat <thread_id> 或 /switch chat <thread_id>", flush=True)
                    task = await _read_next_input()
                    continue
                if command == "/compact":
                    if not restored_messages:
                        print("当前没有可压缩的历史消息。", flush=True)
                    else:
                        started = time.perf_counter()
                        interrupted = False
                        async with _escape_monitor() as control:
                            try:
                                result = await graph.ainvoke(
                                    {"manual_compact": True}, config=run_config, control=control
                                )
                            except GraphDrained:
                                if not control.drain_requested:
                                    raise
                                interrupted = True
                            else:
                                interrupted = control.drain_requested
                        saved_checkpoint = await checkpointer.aget_tuple(run_config)
                        restored_messages = _checkpoint_message_count(saved_checkpoint)
                        if interrupted:
                            await _close_pending_tool_calls(graph, run_config)
                            saved_checkpoint = await checkpointer.aget_tuple(run_config)
                            restored_messages = _checkpoint_message_count(saved_checkpoint)
                            logger.info("手动压缩已中断: checkpoint_id=%s", _checkpoint_id(saved_checkpoint))
                            print("已暂停，返回输入。", flush=True)
                        else:
                            logger.info(
                                "手动压缩完成: elapsed=%.2fs messages=%d checkpoint_id=%s",
                                time.perf_counter() - started,
                                len(result.get("messages", [])),
                                _checkpoint_id(saved_checkpoint),
                            )
                            print("上下文已压缩。", flush=True)
                    task = await _read_next_input()
                    continue
                if command == "/run forever":
                    if not restored_messages:
                        print("请先输入一次具体任务，再启用 /run forever。", flush=True)
                        task = await _read_next_input()
                        continue
                    forever = True
                    task = _CONTINUE_TASK
                    print("已开启持续运行；按 Esc 返回输入。", flush=True)

                started = time.perf_counter()
                messages = (
                    build_initial_messages(settings, task)
                    if not restored_messages
                    else [HumanMessage(content=task)] if forever else [task_message(task)]
                )
                interrupted = False
                async with _escape_monitor() as control:
                    try:
                        streamed_final = await _stream_agent(
                            graph, {"messages": messages, "manual_compact": False},
                            run_config, control,
                        )
                    except GraphDrained:
                        if not control.drain_requested:
                            raise
                        interrupted = True
                    else:
                        interrupted = control.drain_requested
                saved_checkpoint = await checkpointer.aget_tuple(run_config)
                restored_messages = _checkpoint_message_count(saved_checkpoint)
                if interrupted:
                    forever = False
                    await _close_pending_tool_calls(graph, run_config)
                    saved_checkpoint = await checkpointer.aget_tuple(run_config)
                    restored_messages = _checkpoint_message_count(saved_checkpoint)
                    logger.info(
                        "Agent 已中断: elapsed=%.2fs messages=%d checkpoint_id=%s",
                        time.perf_counter() - started, restored_messages,
                        _checkpoint_id(saved_checkpoint),
                    )
                    print("已暂停，返回输入。", flush=True)
                    task = await _read_next_input()
                    continue
                result = saved_checkpoint.checkpoint.get("channel_values", {}) if saved_checkpoint else {}
                logger.info(
                    "Agent 正常结束: elapsed=%.2fs messages=%d checkpoint_id=%s forever=%s",
                    time.perf_counter() - started,
                    len(result.get("messages", [])),
                    _checkpoint_id(saved_checkpoint),
                    forever,
                )
                final_text = _final_text(result)
                if final_text != streamed_final:
                    print(f"[结果] {final_text}", flush=True)
                if forever:
                    await asyncio.sleep(1)
                    task = _CONTINUE_TASK
                else:
                    task = await _read_next_input()
        except Exception as exc:
            logger.exception("Agent 运行失败")
            print(f"运行失败: {type(exc).__name__}: {exc}\n详情见 {settings.log_file}", flush=True)
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
    except Exception:
        raise SystemExit(1) from None
