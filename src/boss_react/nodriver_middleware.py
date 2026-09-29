"""LangChain middleware for the nodriver-backed BOSS browser tools."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, Literal

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.tools import ToolException, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing_extensions import NotRequired

from .nodriver import NodriverBrowserConfig, NodriverBrowserSession, NodriverToolError
from .console_output import tool_finished, tool_started

MatchMode = Literal["exact", "contains", "regex"]
logger = logging.getLogger(__name__)
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def _merge_js_cache(current: dict[str, str] | None, update: dict[str, str] | None) -> dict[str, str]:
    return {**(current or {}), **(update or {})}


class BrowserAgentState(AgentState):
    js_cache: NotRequired[Annotated[dict[str, str], _merge_js_cache]]


def _hide_url_information(value: Any) -> Any:
    """Remove browser URLs from values before they enter model-visible messages."""
    if isinstance(value, dict):
        return {
            key: _hide_url_information(item)
            for key, item in value.items()
            if "url" not in key.lower() and key.lower() not in {"href", "src"}
        }
    if isinstance(value, list):
        return [_hide_url_information(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_hide_url_information(item) for item in value)
    if isinstance(value, str):
        return URL_PATTERN.sub("[URL hidden]", value)
    return value


def _normalize_patterns(patterns: list[str] | str | None) -> list[str] | None:
    """Accept native arrays and JSON-array strings emitted by compatible models."""
    if patterns is None:
        return None
    values: Any = patterns
    if isinstance(patterns, str):
        stripped = patterns.strip()
        if not stripped:
            return None
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            decoded = re.split(r"[,，\n]+", stripped)
        values = decoded if isinstance(decoded, list) else [decoded]
    normalized = [str(item).strip() for item in values if str(item).strip()]
    return normalized or None


def _argument_summary(arguments: Any, limit: int = 600) -> str:
    """Format bounded tool arguments without emitting large scripts or payloads."""
    if not isinstance(arguments, dict):
        return type(arguments).__name__
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "script" and isinstance(value, str):
            safe[key] = f"<{len(value)} chars>"
        elif isinstance(value, str) and len(value) > 200:
            safe[key] = value[:200] + "..."
        else:
            safe[key] = value
    rendered = json.dumps(safe, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


class BossReactMiddleware(AgentMiddleware):
    """Expose an unrestricted, persistent nodriver session as LangChain tools."""

    name = "boss_react_browser"
    state_schema = BrowserAgentState

    def __init__(
        self,
        config: NodriverBrowserConfig | None = None,
        *,
        session: Any | None = None,
        resume_image_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.session = session or NodriverBrowserSession(config)
        image_path = Path(resume_image_path).expanduser().resolve() if resume_image_path else None
        self.resume_image_path = image_path if image_path and image_path.is_file() else None
        self._tool_lock = asyncio.Lock()
        self.tools = self._build_tools()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        del runtime
        if isinstance(state, dict) and state.get("manual_compact", False):
            logger.info("手动压缩模式跳过浏览器启动")
            return
        logger.info("正在启动 nodriver 浏览器会话")
        await self.session.start()
        login = await self.session.call("browser_ensure_login")
        logger.info(
            "登录状态检查完成: logged_in=%s redirected=%s reasons=%s kind=%s",
            login.get("logged_in"),
            login.get("redirected", False),
            login.get("reasons", []),
            login.get("page_state", {}).get("kind", "unknown"),
        )
        logger.info("nodriver 浏览器会话已就绪")

    async def awrap_model_call(
        self, request: Any, handler: Callable[[Any], Awaitable[Any]]
    ) -> Any:
        state = getattr(request, "state", {}) or {}
        messages = state.get("messages", []) if isinstance(state, dict) else []
        tools = getattr(request, "tools", []) or []
        started = time.perf_counter()
        logger.info("开始模型调用: messages=%d tools=%d", len(messages), len(tools))
        print("[模型] 思考中...", flush=True)
        try:
            response = await handler(request)
        except Exception:
            logger.exception("模型调用失败: elapsed=%.2fs", time.perf_counter() - started)
            raise
        logger.info("模型调用完成: elapsed=%.2fs", time.perf_counter() - started)
        return response

    async def awrap_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[ToolMessage]]
    ) -> ToolMessage | Command[Any]:
        async with self._tool_lock:
            return await self._execute_tool_call(request, handler)

    async def _execute_tool_call(
        self, request: Any, handler: Callable[[Any], Awaitable[ToolMessage]]
    ) -> ToolMessage | Command[Any]:
        tool_call = request.tool_call
        name = tool_call.get("name", "unknown")
        arguments = tool_call.get("args", {})
        started = time.perf_counter()
        logger.info("开始工具调用: tool=%s args=%s", name, _argument_summary(arguments))
        tool_started(name, arguments if isinstance(arguments, dict) else {})
        cache_entry: dict[str, str] | None = None
        try:
            if name == "browser_eval_js":
                cache_name = arguments.get("name")
                script = arguments.get("script")
                if not isinstance(cache_name, str) or not cache_name.strip():
                    raise ValueError("browser_eval_js requires a non-empty name")
                cache_name = cache_name.strip()
                if script is None:
                    state = getattr(request, "state", {}) or {}
                    cached = state.get("js_cache", {}) if isinstance(state, dict) else {}
                    script = cached.get(cache_name)
                    if script is None:
                        raise ValueError(f"No cached JavaScript named {cache_name!r}")
                elif not isinstance(script, str) or not script.strip():
                    raise ValueError("browser_eval_js script must be non-empty when provided")
                else:
                    cache_entry = {cache_name: script}
                request = request.override(
                    tool_call={**tool_call, "args": {"name": cache_name, "script": script}}
                )
            result = await handler(request)
        except (NodriverToolError, TimeoutError, ValueError, FileNotFoundError, ToolException) as exc:
            logger.warning(
                "工具调用失败: tool=%s elapsed=%.2fs error=%s: %s",
                name,
                time.perf_counter() - started,
                type(exc).__name__,
                exc,
            )
            result = ToolMessage(
                content=f"Browser tool failed: {type(exc).__name__}: {exc}",
                tool_call_id=tool_call["id"],
                name=name,
                status="error",
            )
        except Exception:
            logger.exception("工具调用异常: tool=%s elapsed=%.2fs", name, time.perf_counter() - started)
            raise
        shot: dict[str, Any] | None = None
        try:
            shot = _hide_url_information(await self.session.call("browser_settle_and_screenshot"))
            path = Path(shot["path"])
            content = result.content
            blocks = (
                [block for block in content if not (isinstance(block, dict) and block.get("type") in {"image", "image_url"})]
                if isinstance(content, list) else [{"type": "text", "text": str(content)}]
            )
            blocks.extend([
                {"type": "text", "text": f"Post-tool page state: {json.dumps({key: value for key, value in shot.items() if key != 'path'}, ensure_ascii=False)}\nScreenshot: {path}"},
                {"type": "image", "base64": base64.b64encode(path.read_bytes()).decode("ascii"), "mime_type": "image/png"},
            ])
            result.content = blocks
        except Exception as exc:
            logger.exception("工具后截图失败: tool=%s", name)
            shot = None
            warning = f"Post-tool screenshot unavailable: {type(exc).__name__}: {exc}"
            if isinstance(result.content, str):
                result.content += "\n" + warning
            else:
                result.content = [*result.content, {"type": "text", "text": warning}]
        logger.info(
            "工具调用完成: tool=%s elapsed=%.2fs status=%s",
            name,
            time.perf_counter() - started,
            getattr(result, "status", "success"),
        )
        if getattr(result, "status", None) == "error":
            logger.warning("工具返回错误状态: tool=%s detail=%s", name, str(result.content)[:1000])
        tool_finished(name, result, shot, time.perf_counter() - started)
        if cache_entry and result.status != "error":
            logger.info("JavaScript 已写入 checkpoint 状态: name=%s chars=%d", cache_name, len(script))
            return Command(update={"js_cache": cache_entry, "messages": [result]})
        return result

    async def aclose(self) -> None:
        logger.info("请求关闭 nodriver 浏览器会话")
        await self.session.close()

    def _build_tools(self) -> list[Any]:
        session = self.session

        async def model_safe_call(tool_name: str, **arguments: Any) -> dict[str, Any]:
            result = await session.call(tool_name, **arguments)
            return _hide_url_information(result)

        @tool("browser_state", response_format="content_and_artifact")
        async def browser_state() -> tuple[list[dict[str, Any]], dict[str, Any]]:
            """Return the active page state together with a current viewport screenshot."""
            result = await model_safe_call("browser_screenshot")
            path = Path(result["path"])
            logger.info(
                "页面状态截图完成: kind=%s target_id=%s url=%s path=%s",
                result.get("page_state", {}).get("kind", "unknown"),
                result.get("target_id", ""),
                result.get("url", ""),
                path,
            )
            state = {key: value for key, value in result.items() if key != "path"}
            content = [
                {
                    "type": "text",
                    "text": f"Browser state: {json.dumps(state, ensure_ascii=False)}\nScreenshot: {path}",
                },
                {
                    "type": "image",
                    "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
                    "mime_type": "image/png",
                },
            ]
            return content, result

        @tool("browser_observe")
        async def browser_observe(
            patterns: list[str] | str | None = None,
            match: MatchMode = "contains",
            max_results: int = 20,
            context_chars: int = 320,
        ) -> dict[str, Any]:
            """Return a compact page outline, or elements matching requested text patterns."""
            normalized_patterns = _normalize_patterns(patterns)
            result = await model_safe_call(
                "browser_observe",
                patterns=normalized_patterns,
                match=match,
                max_results=max_results,
                context_chars=context_chars,
            )
            logger.info(
                "页面观察完成: mode=%s patterns=%s matches=%s kind=%s url=%s",
                result.get("mode", "unknown"),
                normalized_patterns or [],
                result.get("element_count", 0),
                result.get("page_state", {}).get("kind", "unknown"),
                result.get("url", ""),
            )
            return result

        @tool("browser_click_text")
        async def browser_click_text(
            text: str, match: MatchMode = "exact", role: str | None = None,
            scope_text: str | None = None, occurrence: int = 0, wait_after_ms: int = 750,
        ) -> dict[str, Any]:
            """Click an element located by visible/accessibility text, optional role, and optional surrounding text."""
            return await model_safe_call(
                "browser_click_text", text=text, match=match, role=role, scope_text=scope_text,
                occurrence=occurrence, wait_after_ms=wait_after_ms,
            )

        @tool("browser_hover_text")
        async def browser_hover_text(
            text: str, match: MatchMode = "exact", role: str | None = None,
            scope_text: str | None = None, occurrence: int = 0, wait_after_ms: int = 300,
        ) -> dict[str, Any]:
            """Hover an element located by text to reveal menus, tooltips, or hidden controls."""
            return await model_safe_call(
                "browser_hover_text", text=text, match=match, role=role, scope_text=scope_text,
                occurrence=occurrence, wait_after_ms=wait_after_ms,
            )

        @tool("browser_input_text")
        async def browser_input_text(
            target_text: str, value: str, match: MatchMode = "exact", scope_text: str | None = None,
            occurrence: int = 0, clear: bool = True, press_enter: bool = False,
        ) -> dict[str, Any]:
            """Locate an input by label, placeholder, name, or nearby text and enter a value."""
            return await model_safe_call(
                "browser_input_text", target_text=target_text, value=value, match=match,
                scope_text=scope_text, occurrence=occurrence, clear=clear, press_enter=press_enter,
            )

        @tool("browser_scroll")
        async def browser_scroll(
            direction: Literal["up", "down", "left", "right"] = "down", amount: int = 800,
            target_text: str | None = None, match: MatchMode = "exact", occurrence: int = 0,
        ) -> dict[str, Any]:
            """Scroll the page or the scrollable container surrounding a text-located element."""
            return await model_safe_call(
                "browser_scroll", direction=direction, amount=amount, target_text=target_text,
                match=match, occurrence=occurrence,
            )

        @tool("browser_scroll_to_text")
        async def browser_scroll_to_text(
            text: str, match: MatchMode = "exact", occurrence: int = 0
        ) -> dict[str, Any]:
            """Bring an element located by text into the center of the viewport."""
            return await model_safe_call("browser_scroll_to_text", text=text, match=match, occurrence=occurrence)

        @tool("browser_press")
        async def browser_press(
            key: str, target_text: str | None = None, match: MatchMode = "exact", occurrence: int = 0
        ) -> dict[str, Any]:
            """Dispatch a keyboard key to the focused element or an element located by text."""
            return await model_safe_call(
                "browser_press", key=key, target_text=target_text, match=match, occurrence=occurrence
            )

        @tool("browser_back")
        async def browser_back() -> dict[str, Any]:
            """Go back in the active page's history; report an error if no previous page exists."""
            return await model_safe_call("browser_back")

        @tool("browser_reset")
        async def browser_reset() -> dict[str, Any]:
            """Open a new BOSS homepage and close prior tabs."""
            return await model_safe_call("browser_reset")

        @tool("browser_wait")
        async def browser_wait(milliseconds: int = 1000) -> dict[str, Any]:
            """Wait for a page transition, animation, or asynchronous content."""
            return await model_safe_call("browser_wait", milliseconds=milliseconds)

        @tool("browser_eval_js")
        async def browser_eval_js(name: str, script: str | None = None) -> dict[str, Any]:
            """Run JavaScript in the active page. Give a reusable name and script to cache it in the checkpoint; later pass only name to rerun it. Use return for JSON-serializable results."""
            if script is None:
                raise ValueError("Cached JavaScript must be resolved by the agent middleware")
            return await model_safe_call("browser_eval_js", script=script)

        @tool("browser_upload_text")
        async def browser_upload_text(
            target_text: str, paths: list[str], match: MatchMode = "exact", occurrence: int = 0
        ) -> dict[str, Any]:
            """Click a text-located upload control and attach local files without a system chooser."""
            return await model_safe_call(
                "browser_upload_text", target_text=target_text, paths=paths,
                match=match, occurrence=occurrence,
            )

        @tool("boss_send_greeting")
        async def boss_send_greeting(message: str) -> dict[str, Any]:
            """Enter a greeting in the current BOSS chat and explicitly click its send button."""
            return await model_safe_call("boss_send_greeting", message=message)

        tools = [
            browser_state, browser_observe, browser_click_text, browser_hover_text,
            browser_input_text, browser_scroll, browser_scroll_to_text, browser_press, browser_back, browser_reset,
            browser_wait, browser_eval_js, browser_upload_text, boss_send_greeting,
        ]
        if self.resume_image_path is not None:
            resume_image_path = self.resume_image_path

            @tool("boss_send_image")
            async def boss_send_image() -> dict[str, Any]:
                """Send the user's configured resume image in the current BOSS chat."""
                return await model_safe_call("boss_send_image", path=str(resume_image_path))

            tools.append(boss_send_image)
        return tools
