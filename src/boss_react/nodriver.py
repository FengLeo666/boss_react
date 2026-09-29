"""Persistent async client for the nodriver JSON-lines browser process."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _call_summary(tool_name: str, arguments: dict[str, Any]) -> str:
    safe = dict(arguments)
    if tool_name == "browser_eval_js" and isinstance(safe.get("script"), str):
        safe["script"] = f"<{len(safe['script'])} chars>"
    rendered = json.dumps(safe, ensure_ascii=False, default=str)
    return rendered if len(rendered) <= 500 else rendered[:500] + "..."

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class NodriverBrowserConfig:
    """Paths and timeouts used to launch the local BOSS browser backend."""

    python_executable: Path | None = None
    profile_dir: Path | None = None
    artifacts_dir: Path = field(default_factory=lambda: Path.cwd() / "artifacts")
    driver_script: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "scripts" / "nodriver_tool_driver.py"
    )
    start_url: str = "https://www.zhipin.com/"
    startup_timeout: float = 45.0
    action_timeout: float = 120.0

    def __post_init__(self) -> None:
        self.artifacts_dir = Path(self.artifacts_dir).expanduser().resolve()
        self.driver_script = Path(self.driver_script).expanduser().resolve()
        if self.python_executable is None:
            self.python_executable = _project_root() / ".venv" / "Scripts" / "python.exe"
        self.python_executable = Path(self.python_executable).expanduser().resolve()
        if self.profile_dir is None:
            self.profile_dir = _project_root() / "browser-profile" / "nodriver"
        self.profile_dir = Path(self.profile_dir).expanduser().resolve()


class NodriverToolError(RuntimeError):
    """A browser tool failed in the nodriver subprocess."""


class NodriverBrowserSession:
    """Own one nodriver subprocess and serialize all tool calls to it."""

    def __init__(self, config: NodriverBrowserConfig | None = None) -> None:
        self.config = config or NodriverBrowserConfig()
        self.process: asyncio.subprocess.Process | None = None
        self.ready: dict[str, Any] | None = None
        self._lock = asyncio.Lock()
        self._request_counter = 0
        self.diagnostics: list[str] = []

    async def start(self) -> dict[str, Any]:
        if self.process and self.process.returncode is None:
            logger.debug("复用 nodriver 子进程: pid=%s", self.process.pid)
            return self.ready or {}
        config = self.config
        for path, label in (
            (config.python_executable, "python executable"),
            (config.driver_script, "nodriver driver script"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")
        config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        config.profile_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "启动 nodriver 子进程: python=%s driver=%s profile=%s url=%s",
            config.python_executable,
            config.driver_script,
            config.profile_dir,
            config.start_url,
        )
        creationflags = getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        child_env = os.environ.copy()
        child_env["PYTHONUTF8"] = "1"
        child_env["PYTHONIOENCODING"] = "utf-8"
        driver_timeout = min(90.0, max(5.0, config.action_timeout - 20.0))
        self.process = await asyncio.create_subprocess_exec(
            str(config.python_executable),
            str(config.driver_script),
            "--profile", str(config.profile_dir),
            "--artifacts", str(config.artifacts_dir),
            "--url", config.start_url,
            "--tool-timeout", str(driver_timeout),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=child_env,
            creationflags=creationflags,
        )
        logger.info("nodriver 子进程已创建: pid=%s", self.process.pid)
        try:
            while True:
                message = await asyncio.wait_for(self._read_json_line(), timeout=config.startup_timeout)
                if message.get("event") == "ready":
                    self.ready = message
                    logger.info("nodriver 子进程已就绪: pid=%s", self.process.pid)
                    return message
        except BaseException:
            logger.exception("nodriver 子进程启动失败")
            await self.close(force=True)
            raise

    async def _read_json_line(self) -> dict[str, Any]:
        if not self.process or not self.process.stdout:
            raise NodriverToolError("nodriver process is not running")
        while True:
            line = await self.process.stdout.readline()
            if not line:
                code = await self.process.wait()
                tail = "\n".join(self.diagnostics[-20:])
                raise NodriverToolError(f"nodriver process exited with code {code}\n{tail}")
            decoded = line.decode("utf-8", errors="replace").strip()
            if not decoded:
                continue
            try:
                value = json.loads(decoded)
            except json.JSONDecodeError:
                self.diagnostics.append(decoded)
                self.diagnostics = self.diagnostics[-200:]
                logger.info("nodriver> %s", decoded[:1000])
                continue
            if isinstance(value, dict):
                return value

    async def _read_response(self, request_id: int) -> dict[str, Any]:
        while True:
            response = await self._read_json_line()
            response_id = response.get("request_id")
            if response_id == request_id:
                return response
            logger.warning(
                "忽略非当前浏览器响应: expected_request_id=%s actual_request_id=%s tool=%s",
                request_id,
                response_id,
                response.get("tool", "unknown"),
            )

    async def call(self, tool_name: str, **arguments: Any) -> dict[str, Any]:
        await self.start()
        async with self._lock:
            if not self.process or not self.process.stdin:
                raise NodriverToolError("nodriver process is not running")
            self._request_counter += 1
            request_id = self._request_counter
            request = json.dumps(
                {"request_id": request_id, "tool": tool_name, "args": arguments},
                ensure_ascii=False,
            ) + "\n"
            started = time.perf_counter()
            logger.info(
                "发送浏览器请求: request_id=%s tool=%s timeout=%.1fs args=%s",
                request_id,
                tool_name,
                self.config.action_timeout,
                _call_summary(tool_name, arguments),
            )
            self.process.stdin.write(request.encode("utf-8"))
            try:
                await asyncio.wait_for(self.process.stdin.drain(), timeout=5)
                response = await asyncio.wait_for(
                    self._read_response(request_id),
                    timeout=self.config.action_timeout,
                )
            except TimeoutError as exc:
                logger.exception(
                    "浏览器请求超时: request_id=%s tool=%s elapsed=%.2fs pid=%s args=%s diagnostics=%s",
                    request_id, tool_name, time.perf_counter() - started,
                    self.process.pid, _call_summary(tool_name, arguments), self.diagnostics[-20:],
                )
                message = await self._hard_reset_after_timeout(request_id, tool_name, str(exc))
                raise NodriverToolError(message) from exc
            except Exception:
                logger.exception(
                    "浏览器请求失败: request_id=%s tool=%s elapsed=%.2fs",
                    request_id,
                    tool_name,
                    time.perf_counter() - started,
                )
                raise
            if not response.get("ok"):
                error_type = response.get("error_type", "ToolError")
                logger.error(
                    "浏览器返回错误: request_id=%s tool=%s elapsed=%.2fs error_type=%s error=%s",
                    request_id,
                    tool_name,
                    time.perf_counter() - started,
                    error_type,
                    response.get("error", "unknown browser error"),
                )
                if error_type in {"BrowserToolTimeout", "TimeoutError"}:
                    message = await self._hard_reset_after_timeout(
                        request_id, tool_name, str(response.get("error", "browser tool timed out"))
                    )
                    raise NodriverToolError(message)
                raise NodriverToolError(f"{error_type}: {response.get('error', 'unknown browser error')}")
            result = response.get("result")
            if not isinstance(result, dict):
                raise NodriverToolError(f"invalid tool result: {result!r}")
            logger.info(
                "收到浏览器结果: request_id=%s tool=%s elapsed=%.2fs keys=%s kind=%s target_id=%s url=%s",
                request_id,
                tool_name,
                time.perf_counter() - started,
                sorted(result),
                result.get("page_state", {}).get("kind", "unknown"),
                result.get("target_id", ""),
                result.get("url", ""),
            )
            return result

    async def _hard_reset_after_timeout(self, request_id: int, tool_name: str, reason: str) -> str:
        logger.error(
            "强制重启浏览器: request_id=%s tool=%s reason=%s pid=%s",
            request_id, tool_name, reason, self.process.pid if self.process else None,
        )
        try:
            await self.close(force=True)
            ready = await self.start()
        except Exception as exc:
            logger.exception("浏览器强制重启失败: request_id=%s tool=%s", request_id, tool_name)
            return (
                f"浏览器工具 {tool_name} 超时（request_id={request_id}）：{reason}。"
                f"强制重启失败：{type(exc).__name__}: {exc}。请停止自动操作并请求人工处理。"
            )
        logger.warning(
            "浏览器强制重启完成: request_id=%s tool=%s new_pid=%s kind=%s",
            request_id, tool_name, self.process.pid if self.process else None,
            ready.get("page_state", {}).get("kind", "unknown"),
        )
        return (
            f"浏览器工具 {tool_name} 超时（request_id={request_id}）：{reason}。"
            "浏览器已强制重启并回到 BOSS 首页；此前页面和 tab 状态已清空。"
            "请先调用 browser_state 重新观察页面，再决定下一步，不要假设上一次操作成功。"
        )

    async def close(self, *, force: bool = False) -> None:
        process, self.process = self.process, None
        self.ready = None
        if not process or process.returncode is not None:
            logger.debug("nodriver 子进程无需关闭")
            return
        logger.info("关闭 nodriver 子进程: pid=%s force=%s", process.pid, force)
        if force and os.name == "nt":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/PID", str(process.pid), "/T", "/F",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    creationflags=getattr(__import__("subprocess"), "CREATE_NO_WINDOW", 0),
                )
                output, _ = await asyncio.wait_for(killer.communicate(), timeout=10)
                logger.warning(
                    "结束驱动进程树: pid=%s exit=%s output=%s",
                    process.pid, killer.returncode, output.decode("mbcs", errors="replace").strip(),
                )
            except Exception:
                logger.exception("结束驱动进程树失败: pid=%s", process.pid)
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
                return
            except TimeoutError:
                logger.warning("驱动进程树结束后进程仍存活: pid=%s", process.pid)
        if not force and process.stdin:
            try:
                process.stdin.write(b'{"command":"close"}\n')
                await process.stdin.drain()
                await asyncio.wait_for(process.wait(), timeout=10)
                logger.info("nodriver 子进程正常退出: pid=%s code=%s", process.pid, process.returncode)
                return
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            logger.warning("nodriver 子进程未及时退出，执行 kill: pid=%s", process.pid)
            process.kill()
            await process.wait()
        logger.info("nodriver 子进程已结束: pid=%s code=%s", process.pid, process.returncode)

    async def state(self) -> dict[str, Any]:
        return await self.call("browser_state")

    async def screenshot(self) -> dict[str, Any]:
        return await self.call("browser_screenshot")
