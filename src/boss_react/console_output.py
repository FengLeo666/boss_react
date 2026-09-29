"""Concise terminal output for interactive agent runs."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


_TOOL_LABELS = {
    "browser_state": "查看页面",
    "browser_observe": "查找页面元素",
    "browser_click_text": "点击",
    "browser_hover_text": "悬停",
    "browser_input_text": "输入",
    "browser_scroll": "滚动页面",
    "browser_scroll_to_text": "定位文字",
    "browser_press": "按键",
    "browser_back": "返回",
    "browser_reset": "重置浏览器",
    "browser_wait": "等待页面",
    "browser_eval_js": "运行 JavaScript",
    "browser_upload_text": "上传文件",
    "boss_send_greeting": "发送招呼",
    "boss_send_image": "发送简历图片",
}

_PAGE_LABELS = {
    "boss_home": "首页",
    "boss_page": "页面",
    "job_search": "职位列表",
    "job_detail": "职位详情",
    "company_detail": "公司详情",
    "chat": "聊天",
    "login": "登录",
}

_model_line_open = False

_LOGO = (
    r" ____   ___  ____ ____      ____  _____    _    ____ _____",
    r"| __ ) / _ \/ ___/ ___|    |  _ \| ____|  / \  / ___|_   _|",
    r"|  _ \| | | \___ \___ \    | |_) |  _|   / _ \| |     | |",
    r"| |_) | |_| |___) |__) |   |  _ <| |___ / ___ \ |___  | |",
    r"|____/ \___/|____/____/    |_| \_\_____/_/   \_\____| |_|",
)


def show_banner(thread_id: str, log_file: Path) -> None:
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    cyan, gold, muted, reset = (
        ("\033[1;36m", "\033[1;33m", "\033[2m", "\033[0m")
        if color else ("", "", "", "")
    )
    width = shutil.get_terminal_size((80, 24)).columns
    if width >= max(map(len, _LOGO)) + 2:
        print(f"\n{cyan}" + "\n".join(f"  {line}" for line in _LOGO) + reset)
    else:
        print(f"\n{cyan}  BOSS REACT{reset}")
    print(f"{gold}  浏览职位 · 判断匹配 · 自主沟通{reset}")
    print(f"{muted}  {'─' * max(0, min(width - 4, 62))}{reset}")
    print(f"  会话  {thread_id}")
    print("  命令  /chats   /new chat <名称>   /compact")
    print("        /switch chat <名称>   /run forever   /exit")
    print(f"  日志  {log_file}\n", flush=True)


def stream_model_text(delta: str) -> None:
    global _model_line_open
    if not delta:
        return
    if not _model_line_open:
        print("[模型] ", end="", flush=True)
        _model_line_open = True
    print(delta, end="", flush=True)


def finish_model_text() -> None:
    global _model_line_open
    if _model_line_open:
        print(flush=True)
        _model_line_open = False


def _short(value: Any, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def tool_started(name: str, args: dict[str, Any]) -> None:
    finish_model_text()
    if name == "browser_eval_js":
        return
    label = _TOOL_LABELS.get(name, name)
    if name in {"browser_click_text", "browser_hover_text", "browser_scroll_to_text"}:
        detail = repr(args.get("text", ""))
    elif name == "browser_observe":
        detail = _short(args.get("patterns") or "页面概览", 80)
    elif name == "browser_input_text":
        detail = f"{args.get('target_text', '')!r}: {_short(args.get('value', ''), 80)}"
    elif name == "browser_press":
        detail = str(args.get("key", ""))
    elif name == "browser_scroll":
        detail = f"{args.get('direction', 'down')} {args.get('amount', 800)}"
    elif name == "browser_wait":
        detail = f"{args.get('milliseconds', 1000)} ms"
    elif name == "boss_send_greeting":
        detail = _short(args.get("message", ""), 100)
    elif name == "browser_upload_text":
        detail = f"{len(args.get('paths', []))} 个文件"
    else:
        detail = ""
    print(f"[工具] {label}{'  ' + detail if detail else ''}", flush=True)


def _result_data(result: Any) -> dict[str, Any] | None:
    artifact = getattr(result, "artifact", None)
    if isinstance(artifact, dict):
        return artifact
    content = getattr(result, "content", "")
    if isinstance(content, list):
        content = next(
            (block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"),
            "",
        )
    if isinstance(content, str):
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None
    return None


def tool_finished(name: str, result: Any, screenshot: dict[str, Any] | None, elapsed: float) -> None:
    if getattr(result, "status", "success") == "error":
        content = getattr(result, "content", "")
        if isinstance(content, list):
            content = next((b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"), "")
        summary = f"失败: {_short(content, 180)}"
    else:
        data = _result_data(result) or {}
        if name == "browser_eval_js" and "result" in data:
            summary = _short(data["result"], 180)
        elif name == "browser_observe" and "element_count" in data:
            summary = f"找到 {data['element_count']} 个元素"
        elif name == "boss_send_greeting":
            summary = "已发送"
        elif name == "boss_send_image":
            summary = "已发送图片"
        else:
            summary = "完成"

    if screenshot:
        kind = screenshot.get("page_state", {}).get("kind", "unknown")
        page = _PAGE_LABELS.get(kind, kind)
        idle = "" if screenshot.get("network_idle", True) else " · 网络未确认空闲"
        shot = Path(str(screenshot.get("path", ""))).name
        summary += f" · {page} · 截图 {shot}{idle}"
    else:
        summary += " · 截图不可用"
    prefix = "[工具结果] " if name == "browser_eval_js" else "  "
    print(f"{prefix}{summary} · {elapsed:.1f}s", flush=True)
