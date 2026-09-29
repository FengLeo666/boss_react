"""JSON-lines browser tool driver backed by the local nodriver runtime."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

HOME_URL = "https://www.zhipin.com/"
LOGIN_URL = "https://www.zhipin.com/web/user/?ka=header-login"
MAX_TABS = 10
SCREENSHOT_TIMEOUT_SECONDS = 12
NETWORK_IDLE_TIMEOUT_SECONDS = 6
NETWORK_QUIET_SECONDS = 0.5
logger = logging.getLogger(__name__)


class BrowserToolTimeout(TimeoutError):
    """A browser operation exceeded its deadline and requires a fresh browser."""


def _configure_driver_logging(project_root: Path) -> None:
    log_path = project_root / "logs" / "nodriver-driver.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    logger.info("driver_started pid=%s log_file=%s", os.getpid(), log_path)


def _browser_snapshot(session: Any) -> dict[str, Any]:
    try:
        current = session._tab_id(session.tab)
        tabs = [
            {"target_id": session._tab_id(tab), "url": str(tab.url)}
            for tab in session.browser.tabs
        ]
        return {"current_target_id": current, "tabs": tabs}
    except Exception as exc:
        return {"snapshot_error": f"{type(exc).__name__}: {exc}"}


async def _execute_tool_request(
    session: Any, tools: dict[str, Any], request: dict[str, Any], timeout: float
) -> dict[str, Any]:
    request_id = request.get("request_id")
    name = request["tool"]
    if name not in tools:
        raise ValueError(f"Unknown tool: {name}")
    arguments = request.get("args", {})
    started = time.perf_counter()
    logger.info(
        "request_start id=%s tool=%s timeout=%.1fs browser=%s args=%s",
        request_id, name, timeout, json.dumps(_browser_snapshot(session), ensure_ascii=False),
        json.dumps(arguments, ensure_ascii=False, default=str),
    )
    task = asyncio.create_task(tools[name](**arguments))
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            pending_stack = "".join(
                "".join(traceback.format_stack(frame)) for frame in task.get_stack()
            )
            logger.error(
                "request_timeout id=%s tool=%s elapsed=%.2fs browser=%s pending_stack=\n%s",
                request_id, name, time.perf_counter() - started,
                json.dumps(_browser_snapshot(session), ensure_ascii=False), pending_stack,
            )
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=1)
            except (asyncio.CancelledError, TimeoutError):
                pass
            raise BrowserToolTimeout(f"{name} exceeded {timeout:.1f}s")
        result = await task
    except TimeoutError as exc:
        logger.exception(
            "request_timeout id=%s tool=%s elapsed=%.2fs browser=%s",
            request_id, name, time.perf_counter() - started,
            json.dumps(_browser_snapshot(session), ensure_ascii=False),
        )
        return {
            "ok": False, "request_id": request_id, "tool": name,
            "error_type": "BrowserToolTimeout", "error": str(exc),
        }
    except Exception as exc:
        logger.exception(
            "request_error id=%s tool=%s elapsed=%.2fs browser=%s",
            request_id, name, time.perf_counter() - started,
            json.dumps(_browser_snapshot(session), ensure_ascii=False),
        )
        return {
            "ok": False, "request_id": request_id, "tool": name,
            "error_type": type(exc).__name__, "error": str(exc),
        }
    logger.info(
        "request_done id=%s tool=%s elapsed=%.2fs browser=%s result=%s",
        request_id, name, time.perf_counter() - started,
        json.dumps(_browser_snapshot(session), ensure_ascii=False),
        json.dumps(result, ensure_ascii=False, default=str),
    )
    return {"ok": True, "request_id": request_id, "tool": name, "result": result}


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="strict")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--url", default="https://www.zhipin.com/")
    parser.add_argument("--tool-timeout", type=float, default=100.0)
    return parser.parse_args()


class NodriverToolSession:
    def __init__(self, backend: Any, artifacts: Path) -> None:
        self.backend = backend
        self.artifacts = artifacts.resolve()
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.screenshot_counter = 0
        self.return_stack: list[str] = []
        self.tab_order: list[str] = []

    @property
    def tab(self):
        tab = self.backend.get_tab()
        if tab is None:
            raise RuntimeError("nodriver tab is unavailable")
        return tab

    @property
    def browser(self):
        browser = self.backend.get_browser()
        if browser is None:
            raise RuntimeError("nodriver browser is unavailable")
        return browser

    @staticmethod
    def _tab_id(tab: Any) -> str:
        return str(tab.target_id)

    def _tabs_by_id(self) -> dict[str, Any]:
        return {self._tab_id(tab): tab for tab in self.browser.tabs}

    async def initialize_tabs(self) -> None:
        """Register existing page targets without exposing them to the agent."""
        current_id = self._tab_id(self.tab)
        self.tab_order = [self._tab_id(tab) for tab in self.browser.tabs]
        if current_id in self.tab_order:
            self.tab_order.remove(current_id)
        self.tab_order.append(current_id)
        await self._enforce_tab_limit()

    async def _activate(self, tab: Any) -> None:
        self.backend.set_tab(tab)
        await tab.activate()
        await tab.bring_to_front()
        tab_id = self._tab_id(tab)
        if tab_id in self.tab_order:
            self.tab_order.remove(tab_id)
        self.tab_order.append(tab_id)

    async def _close_tab(self, tab: Any) -> None:
        tab_id = self._tab_id(tab)
        await tab.close()
        self.tab_order = [item for item in self.tab_order if item != tab_id]
        self.return_stack = [item for item in self.return_stack if item != tab_id]

    async def _enforce_tab_limit(self) -> None:
        """Keep at most MAX_TABS page targets, closing the oldest background page."""
        while len(self.browser.tabs) > MAX_TABS:
            tabs = self._tabs_by_id()
            current_id = self._tab_id(self.tab)
            candidate_id = next((item for item in self.tab_order if item in tabs and item != current_id), None)
            if candidate_id is None:
                candidate_id = next((item for item in tabs if item != current_id), None)
            if candidate_id is None:
                break
            await self._close_tab(tabs[candidate_id])
            await asyncio.sleep(0.1)

    async def _adopt_new_tab(self, before_ids: set[str]) -> bool:
        """Switch to a page target created by the preceding action."""
        for _ in range(12):
            new_tabs = [tab for tab in self.browser.tabs if self._tab_id(tab) not in before_ids]
            if new_tabs:
                previous_id = self._tab_id(self.tab)
                if previous_id not in self.return_stack:
                    self.return_stack.append(previous_id)
                for tab in new_tabs:
                    tab_id = self._tab_id(tab)
                    if tab_id not in self.tab_order:
                        self.tab_order.append(tab_id)
                await self._activate(new_tabs[-1])
                await self._enforce_tab_limit()
                return True
            await asyncio.sleep(0.1)
        return False

    async def _evaluate(self, expression: str) -> dict[str, Any]:
        result = await self.backend.safe_evaluate(expression, timeout=15)
        if not isinstance(result, dict):
            raise TypeError(f"page evaluation returned an invalid result: {result!r}")
        return result

    @staticmethod
    def _target_script(
        text: str,
        match: str = "exact",
        role: str | None = None,
        scope_text: str | None = None,
        occurrence: int = 0,
    ) -> str:
        if match not in {"exact", "contains", "regex"}:
            raise ValueError("match must be exact, contains, or regex")
        if occurrence < 0:
            raise ValueError("occurrence must be non-negative")
        return f"""
          const wanted = {json.dumps(text)};
          const matchMode = {json.dumps(match)};
          const wantedRole = {json.dumps(role)};
          const scopeText = {json.dumps(scope_text)};
          const norm = (v) => String(v || '').replace(/\\s+/g, ' ').trim();
          const visible = (el) => {{
            if (!el) return false;
            const s = getComputedStyle(el), r = el.getBoundingClientRect();
            return s.display !== 'none' && s.visibility !== 'hidden'
              && Number(s.opacity) !== 0 && r.width > 0 && r.height > 0;
          }};
          const nativeRole = (el) => {{
            if (el.getAttribute('role')) return el.getAttribute('role').toLowerCase();
            const tag = el.tagName.toLowerCase();
            if (tag === 'button') return 'button';
            if (tag === 'textarea') return 'textbox';
            if (tag === 'select') return 'combobox';
            if (tag === 'input') return ['button','submit','reset'].includes(el.type) ? 'button' : 'textbox';
            const className = String(el.className || '');
            const looksLikeButton = el.hasAttribute('onclick') || el.hasAttribute('tabindex')
              || getComputedStyle(el).cursor === 'pointer'
              || /(^|[-_ ])(btn|button|search)([-_ ]|$)/i.test(className);
            if (looksLikeButton) return 'button';
            if (tag === 'a' && el.hasAttribute('href')) return 'link';
            return '';
          }};
          const accessibleText = (el) => norm(
            el.getAttribute('aria-label')
            || el.getAttribute('title')
            || el.getAttribute('placeholder')
            || el.getAttribute('alt')
            || (el.labels ? Array.from(el.labels).map((x) => x.innerText).join(' ') : '')
            || el.innerText
            || el.textContent
            || el.value
            || el.getAttribute('name')
          );
          const matches = (value) => {{
            const actual = norm(value), expected = norm(wanted);
            if (matchMode === 'contains') return actual.toLowerCase().includes(expected.toLowerCase());
            if (matchMode === 'regex') return new RegExp(wanted, 'i').test(actual);
            return actual.toLowerCase() === expected.toLowerCase();
          }};
          const inScope = (el) => {{
            if (!scopeText) return true;
            for (let node = el, depth = 0; node && depth < 10; node = node.parentElement, depth += 1) {{
              if (norm(node.innerText).toLowerCase().includes(norm(scopeText).toLowerCase())) return true;
            }}
            return false;
          }};
          const selector = [
            'button','a','input','textarea','select','option','label','summary','[role]',
            '[contenteditable="true"]','[onclick]','[tabindex]','.job-card-box','div','span'
          ].join(',');
          const candidates = Array.from(document.querySelectorAll(selector))
            .filter((el) => visible(el) && inScope(el))
            .filter((el) => !wantedRole || nativeRole(el) === String(wantedRole).toLowerCase())
            .map((el) => ({{
              el,
              label: accessibleText(el),
              priority: el.matches(
                'button,a[href],input,textarea,select,option,label,summary,[role],'
                + '[contenteditable="true"],[onclick],[tabindex],.job-card-box'
              ) ? 0 : 1
            }}))
            .filter((x) => matches(x.label))
            .sort((a, b) => a.priority - b.priority || a.label.length - b.label.length);
          let target = candidates[{occurrence}]?.el || null;
        """

    async def state(self) -> dict[str, Any]:
        detail = await self._evaluate(
            """JSON.stringify((() => {
              const input = document.querySelector('#chat-input');
              const visible = (el) => {
                if (!el) return false;
                const s = getComputedStyle(el), r = el.getBoundingClientRect();
                return s.display !== 'none' && s.visibility !== 'hidden' && r.width > 0 && r.height > 0;
              };
              return {title: document.title, url: location.href, chatInputReady: visible(input)};
            })())"""
        )
        url = detail.get("url") or self.tab.url
        parsed = urlparse(url)
        host, path = (parsed.hostname or "").lower(), parsed.path.rstrip("/") or "/"
        if "security" in path or "verify" in path or "captcha" in path:
            kind = "security_verification"
        elif "passport" in host or "/web/user/" in path or "/login" in path:
            kind = "login"
        elif detail.get("chatInputReady") or "/web/geek/chat" in path:
            kind = "chat"
        elif path.startswith("/web/geek/jobs"):
            kind = "job_search"
        elif path.startswith("/job_detail/"):
            kind = "job_detail"
        elif path.startswith("/gongsi/") and path != "/gongsi":
            kind = "company_detail"
        elif host == "zhipin.com" or host.endswith(".zhipin.com"):
            kind = "boss_home" if path == "/" or path.count("/") == 2 else "boss_page"
        else:
            kind = "other"
        return {
            "url": url,
            "title": detail.get("title", ""),
            "page_state": {"kind": kind, "capabilities": {
                "general_browser_actions": True,
                "send_greeting": kind == "chat" and bool(detail.get("chatInputReady")),
            }},
        }

    async def login_status(self) -> dict[str, Any]:
        detail = await self._evaluate(
            """JSON.stringify((() => {
              const visible = (el) => {
                if (!el) return false;
                const s = getComputedStyle(el), r = el.getBoundingClientRect();
                return s.display !== 'none' && s.visibility !== 'hidden'
                  && Number(s.opacity) !== 0 && r.width > 0 && r.height > 0;
              };
              const loginWall = document.querySelector(
                '[class*="login-dialog"], [class*="boss-login"], '
                + '[class*="loginDialog"], [class*="login-wrap"]'
              );
              const headerLogin = document.querySelector(
                '.header-login-btn, a[ka="header-login"], '
                + '[ka="guide_login_btn_click"], .guide-login-btn, '
                + '.zp-job-list-login-card'
              );
              const bodyText = document.body?.innerText || '';
              return {
                loginWallVisible: visible(loginWall),
                headerLoginVisible: visible(headerLogin),
                loginRequiredVisible: bodyText.includes('登录查看完整内容')
                  || bodyText.includes('登录账号，查看更多好职位')
              };
            })())"""
        )
        parsed = urlparse(self.tab.url)
        host, path = (parsed.hostname or "").lower(), parsed.path.lower()
        login_page = "passport" in host or "/web/user/" in path or "/login" in path
        reasons = [name for name, value in detail.items() if value]
        if login_page:
            reasons.insert(0, "login_page")
        return {
            "logged_in": not login_page and not reasons,
            "reasons": reasons,
            **await self.state(),
        }

    async def ensure_login_page(self) -> dict[str, Any]:
        status = await self.login_status()
        if status["logged_in"]:
            return {"ok": True, "redirected": False, **status}
        redirected = status.get("page_state", {}).get("kind") != "login"
        if redirected:
            await self.tab.get(LOGIN_URL)
            await asyncio.sleep(1)
        return {
            "ok": True,
            "redirected": redirected,
            **await self.login_status(),
        }

    async def observe(
        self,
        patterns: list[str] | None = None,
        match: str = "contains",
        max_results: int = 20,
        context_chars: int = 320,
    ) -> dict[str, Any]:
        if match not in {"exact", "contains", "regex"}:
            raise ValueError("match must be exact, contains, or regex")
        cleaned_patterns = [str(item).strip() for item in (patterns or []) if str(item).strip()]
        if len(cleaned_patterns) > 10:
            raise ValueError("patterns accepts at most 10 values")
        max_results = max(1, min(max_results, 50))
        context_chars = max(0, min(context_chars, 1_000))
        snapshot = await self._evaluate(
            f"""JSON.stringify((() => {{
              const patterns = {json.dumps(cleaned_patterns, ensure_ascii=False)};
              const matchMode = {json.dumps(match)};
              const maxResults = {max_results};
              const contextChars = {context_chars};
              const norm = (value) => String(value || '').replace(/\\s+/g, ' ').trim();
              const visible = (el) => {{
                const s = getComputedStyle(el), r = el.getBoundingClientRect();
                return s.display !== 'none' && s.visibility !== 'hidden'
                  && Number(s.opacity) !== 0 && r.width > 0 && r.height > 0;
              }};
              const inViewport = (el) => {{
                const r = el.getBoundingClientRect();
                return r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
              }};
              const roleOf = (el) => {{
                const explicit = el.getAttribute('role');
                if (explicit) return explicit.toLowerCase();
                const tag = el.tagName.toLowerCase();
                if (tag === 'button') return 'button';
                if (tag === 'textarea') return 'textbox';
                if (tag === 'select') return 'combobox';
                if (tag === 'input') return ['button','submit','reset'].includes(el.type) ? 'button' : 'textbox';
                if (/^h[1-6]$/.test(tag)) return 'heading';
                if (el.classList.contains('job-card-box')) return 'job-card';
                const className = String(el.className || '');
                const looksLikeButton = el.hasAttribute('onclick') || el.hasAttribute('tabindex')
                  || getComputedStyle(el).cursor === 'pointer'
                  || /(^|[-_ ])(btn|button|search)([-_ ]|$)/i.test(className);
                if (looksLikeButton) return 'button';
                if (tag === 'a') return 'link';
                return '';
              }};
              const labelOf = (el) => norm(
                el.getAttribute('aria-label') || el.getAttribute('title')
                || el.getAttribute('placeholder') || el.getAttribute('alt')
                || (el.labels ? Array.from(el.labels).map((item) => item.innerText).join(' ') : '')
                || el.innerText || el.textContent || el.value || el.getAttribute('name')
              );
              const matches = (value, pattern) => {{
                const actual = norm(value), wanted = norm(pattern);
                if (matchMode === 'regex') {{
                  try {{ return new RegExp(wanted, 'i').test(actual); }} catch (_) {{ return false; }}
                }}
                if (matchMode === 'exact') return actual.toLowerCase() === wanted.toLowerCase();
                return actual.toLowerCase().includes(wanted.toLowerCase());
              }};
              const contextOf = (el) => {{
                if (!contextChars) return '';
                const container = el.closest(
                  '.job-card-box,article,li,form,section,[role="dialog"],.job-detail,.job-detail-box'
                ) || el.parentElement;
                return norm(container?.innerText).slice(0, contextChars);
              }};
              const selector = [
                'h1','h2','h3','button','input','textarea','select','a[href]','label',
                '[role="button"]','[role="tab"]','[role="textbox"]','[role="combobox"]',
                '[contenteditable="true"]','[onclick]','[tabindex]','.job-card-box',
                '[class*="btn"]','[class*="button"]','[class*="search"]'
              ].join(',');
              const candidates = Array.from(document.querySelectorAll(selector))
                .filter(visible)
                .map((el) => {{
                  const text = labelOf(el).slice(0, 240);
                  const context = contextOf(el);
                  const role = roleOf(el);
                  const main = Boolean(el.closest('main,[role="main"],.job-list-box,.job-detail-box'));
                  const priority = (main ? 0 : 20)
                    + (['button','textbox','combobox','tab'].includes(role) ? 0 : 5)
                    + (role === 'heading' ? 1 : 0) + Math.min(text.length / 80, 5);
                  return {{el, text, context, role, priority}};
                }})
                .filter((item) => item.text);
              let selected = candidates;
              if (patterns.length) {{
                selected = candidates.filter((item) =>
                  patterns.some((pattern) => matches(item.text, pattern) || matches(item.context, pattern))
                );
              }} else {{
                selected = candidates.filter((item) => inViewport(item.el));
              }}
              selected.sort((a, b) => a.priority - b.priority || a.text.length - b.text.length);
              const seen = new Set();
              const elements = [];
              for (const item of selected) {{
                const href = item.el.href ? String(item.el.href).slice(0, 300) : '';
                const key = `${{item.role}}|${{item.text.toLowerCase()}}|${{href}}`;
                if (seen.has(key)) continue;
                seen.add(key);
                const value = {{
                  tag: item.el.tagName.toLowerCase(), role: item.role, text: item.text,
                  disabled: Boolean(item.el.disabled) || item.el.getAttribute('aria-disabled') === 'true'
                }};
                const placeholder = item.el.getAttribute('placeholder') || '';
                if (placeholder && placeholder !== item.text) value.placeholder = placeholder;
                if (href) value.href = href;
                if (patterns.length && item.context && item.context !== item.text) value.context = item.context;
                elements.push(value);
                if (elements.length >= maxResults) break;
              }}
              const shortUniqueText = (selector) => {{
                const values = Array.from(document.querySelectorAll(selector))
                  .filter((el) => visible(el) && inViewport(el))
                  .map(labelOf).filter((value) => value && value.length <= 100);
                return [...new Set(values)].slice(0, 12);
              }};
              return {{
                mode: patterns.length ? 'matches' : 'outline',
                patterns,
                match: matchMode,
                headings: patterns.length ? [] : shortUniqueText('h1,h2,h3'),
                tags: patterns.length ? [] : shortUniqueText('[role="tab"],.tag-list span,.job-tags span'),
                elements,
                element_count: elements.length,
                hint: patterns.length
                  ? 'Refine patterns or call browser_state for a current screenshot if details remain unclear.'
                  : 'Use browser_observe with patterns for exact text or nearby context.'
              }};
            }})())"""
        )
        return {**await self.state(), **snapshot}

    async def _text_action(
        self,
        action: str,
        text: str,
        match: str,
        role: str | None,
        scope_text: str | None,
        occurrence: int,
        wait_after_ms: int,
    ) -> dict[str, Any]:
        action_js = "target.click();" if action == "click" else """
          for (const type of ['pointerover','mouseover','mouseenter','pointerenter'])
            target.dispatchEvent(new MouseEvent(type, {bubbles: true, cancelable: true, view: window}));
        """
        before_ids = set(self._tabs_by_id()) if action == "click" else set()
        result = await self._evaluate(
            f"""JSON.stringify((() => {{
              {self._target_script(text, match, role, scope_text, occurrence)}
              if (!target) return {{ok: false, reason: 'text_not_found'}};
              target.scrollIntoView({{block: 'center', inline: 'center'}});
              {action_js}
              return {{ok: true, tag: target.tagName.toLowerCase(), text: accessibleText(target)}};
            }})())"""
        )
        if not result.get("ok"):
            raise ValueError(f"No element matched text {text!r}")
        await asyncio.sleep(max(0, wait_after_ms) / 1000)
        opened_new_page = await self._adopt_new_tab(before_ids) if action == "click" else False
        return {**result, "opened_new_page": opened_new_page, **await self.state()}

    async def click_text(self, text: str, match: str = "exact", role: str | None = None,
                         scope_text: str | None = None, occurrence: int = 0,
                         wait_after_ms: int = 750) -> dict[str, Any]:
        return await self._text_action("click", text, match, role, scope_text, occurrence, wait_after_ms)

    async def hover_text(self, text: str, match: str = "exact", role: str | None = None,
                         scope_text: str | None = None, occurrence: int = 0,
                         wait_after_ms: int = 300) -> dict[str, Any]:
        return await self._text_action("hover", text, match, role, scope_text, occurrence, wait_after_ms)

    async def input_text(self, target_text: str, value: str, match: str = "exact",
                         scope_text: str | None = None, occurrence: int = 0,
                         clear: bool = True, press_enter: bool = False) -> dict[str, Any]:
        result = await self._evaluate(
            f"""JSON.stringify((() => {{
              {self._target_script(target_text, match, None, scope_text, occurrence)}
              if (!target) return {{ok: false, reason: 'text_not_found'}};
              if (target.tagName === 'LABEL' && target.control) target = target.control;
              if (!target.matches('input,textarea,[contenteditable="true"]'))
                target = target.querySelector('input,textarea,[contenteditable="true"]');
              if (!target) return {{ok: false, reason: 'editable_not_found'}};
              target.focus();
              const value = {json.dumps(value)}, clear = {json.dumps(clear)};
              if ('value' in target) target.value = clear ? value : String(target.value || '') + value;
              else target.textContent = clear ? value : String(target.textContent || '') + value;
              target.dispatchEvent(new InputEvent('input', {{bubbles: true, inputType: 'insertText', data: value}}));
              target.dispatchEvent(new Event('change', {{bubbles: true}}));
              if ({json.dumps(press_enter)}) {{
                target.dispatchEvent(new KeyboardEvent('keydown', {{key: 'Enter', code: 'Enter', bubbles: true}}));
                target.dispatchEvent(new KeyboardEvent('keyup', {{key: 'Enter', code: 'Enter', bubbles: true}}));
              }}
              return {{ok: true, tag: target.tagName.toLowerCase(), characters: value.length}};
            }})())"""
        )
        if not result.get("ok"):
            raise ValueError(f"Unable to input text into {target_text!r}: {result.get('reason')}")
        return {**result, **await self.state()}

    async def scroll(self, direction: str = "down", amount: int = 800,
                     target_text: str | None = None, match: str = "exact",
                     occurrence: int = 0) -> dict[str, Any]:
        if direction not in {"up", "down", "left", "right"}:
            raise ValueError("direction must be up, down, left, or right")
        setup = self._target_script(target_text, match, None, None, occurrence) if target_text else "let target = null;"
        dx = abs(amount) if direction == "right" else -abs(amount) if direction == "left" else 0
        dy = abs(amount) if direction == "down" else -abs(amount) if direction == "up" else 0
        result = await self._evaluate(
            f"""JSON.stringify((() => {{
              {setup}
              let scroller = target;
              while (scroller && scroller !== document.body) {{
                const s = getComputedStyle(scroller);
                if (/(auto|scroll)/.test(s.overflow + s.overflowY + s.overflowX)) break;
                scroller = scroller.parentElement;
              }}
              if (scroller && scroller !== document.body) scroller.scrollBy({{left:{dx}, top:{dy}, behavior:'instant'}});
              else window.scrollBy({{left:{dx}, top:{dy}, behavior:'instant'}});
              return {{ok:true, direction:{json.dumps(direction)}, amount:{abs(amount)}}};
            }})())"""
        )
        return {**result, **await self.state()}

    async def scroll_to_text(self, text: str, match: str = "exact", occurrence: int = 0) -> dict[str, Any]:
        result = await self._evaluate(
            f"""JSON.stringify((() => {{
              {self._target_script(text, match, None, None, occurrence)}
              if (!target) return {{ok:false, reason:'text_not_found'}};
              target.scrollIntoView({{block:'center', inline:'nearest', behavior:'instant'}});
              return {{ok:true, tag:target.tagName.toLowerCase(), text:accessibleText(target)}};
            }})())"""
        )
        if not result.get("ok"):
            raise ValueError(f"No element matched text {text!r}")
        return {**result, **await self.state()}

    async def press(self, key: str, target_text: str | None = None,
                    match: str = "exact", occurrence: int = 0) -> dict[str, Any]:
        setup = self._target_script(target_text, match, None, None, occurrence) if target_text else "let target = document.activeElement || document.body;"
        result = await self._evaluate(
            f"""JSON.stringify((() => {{
              {setup}
              if (!target) return {{ok:false, reason:'text_not_found'}};
              target.focus?.();
              for (const type of ['keydown','keypress','keyup'])
                target.dispatchEvent(new KeyboardEvent(type, {{key:{json.dumps(key)}, code:{json.dumps(key)}, bubbles:true}}));
              return {{ok:true, key:{json.dumps(key)}}};
            }})())"""
        )
        if not result.get("ok"):
            raise ValueError(f"No element matched text {target_text!r}")
        return {**result, **await self.state()}

    async def back(self) -> dict[str, Any]:
        from nodriver import cdp

        current = self.tab
        current_id = self._tab_id(current)
        current_index, entries = await current.send(cdp.page.get_navigation_history())
        if current_index > 0:
            await current.send(cdp.page.navigate_to_history_entry(entries[current_index - 1].id_))
            await asyncio.sleep(1)
            return {"ok": True, "back_mode": "history", **await self.state()}

        tabs = self._tabs_by_id()
        previous = None
        while self.return_stack and previous is None:
            candidate_id = self.return_stack.pop()
            if candidate_id != current_id:
                previous = tabs.get(candidate_id)
        if previous is None:
            previous_id = next(
                (item for item in reversed(self.tab_order) if item != current_id and item in tabs),
                None,
            )
            previous = tabs.get(previous_id) if previous_id else None

        if previous is not None:
            await self._activate(previous)
            await self._close_tab(current)
            await asyncio.sleep(0.5)
            return {"ok": True, "back_mode": "previous_tab", **await self.state()}

        await current.get(HOME_URL)
        await asyncio.sleep(1)
        return {"ok": True, "back_mode": "homepage", **await self.state()}

    async def reset(self) -> dict[str, Any]:
        """Replace every page target with one clean BOSS homepage target."""
        homepage = await self.browser.get(HOME_URL, new_tab=True)
        await self._activate(homepage)
        for tab in list(self.browser.tabs):
            if self._tab_id(tab) != self._tab_id(homepage):
                await self._close_tab(tab)
        self.return_stack.clear()
        self.tab_order = [self._tab_id(homepage)]
        await asyncio.sleep(1)
        return {"ok": True, "reset": True, **await self.state()}

    async def wait(self, milliseconds: int = 1000) -> dict[str, Any]:
        await asyncio.sleep(max(0, milliseconds) / 1000)
        return await self.state()

    async def eval_js(self, script: str) -> dict[str, Any]:
        result = await self._evaluate(f"""JSON.stringify((() => {{
              try {{
                const result = (() => {{ {script} }})();
                return {{ok:true, result:result === undefined ? null : result}};
              }} catch (error) {{ return {{ok:false, error:String(error), stack:error?.stack || ''}}; }}
            }})())""")
        if not result.get("ok"):
            raise RuntimeError(f"JavaScript failed: {result.get('error')}")
        return {**result, **await self.state()}

    async def upload_text(self, target_text: str, paths: list[str], match: str = "exact",
                          occurrence: int = 0) -> dict[str, Any]:
        from nodriver import cdp

        resolved = [str(Path(path).expanduser().resolve()) for path in paths]
        missing = [path for path in resolved if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Upload files do not exist: {missing}")
        await self.tab.send(cdp.page.set_intercept_file_chooser_dialog(True, cancel=True))
        try:
            await self.click_text(target_text, match=match, occurrence=occurrence, wait_after_ms=250)
        finally:
            await self.tab.send(cdp.page.set_intercept_file_chooser_dialog(False))
        document = await self.tab.send(cdp.dom.get_document(depth=1, pierce=True))
        node_id = await self.tab.send(cdp.dom.query_selector(document.node_id, "input[type='file']"))
        if not node_id:
            raise RuntimeError("No input[type=file] appeared after clicking the upload control")
        await self.tab.send(cdp.dom.set_file_input_files(resolved, node_id=node_id))
        await asyncio.sleep(1)
        return {"ok": True, "paths": resolved, **await self.state()}

    async def send_greeting(self, message: str) -> dict[str, Any]:
        await self.backend.send_chat_message(message)
        return {"ok": True, "characters": len(message), **await self.state()}

    async def send_image(self, path: str) -> dict[str, Any]:
        await self.backend.send_chat_image(path)
        return {"ok": True, "path": str(Path(path).expanduser().resolve()), **await self.state()}

    async def screenshot(self) -> dict[str, Any]:
        from nodriver import cdp

        self.screenshot_counter += 1
        path = self.artifacts / f"nodriver-screenshot-{self.screenshot_counter:04d}.png"
        tab = self.tab
        target_id = self._tab_id(tab)

        async def capture(target: Any) -> bytes:
            data = await asyncio.wait_for(
                target.send(cdp.page.capture_screenshot(format_="png", capture_beyond_viewport=False)),
                timeout=SCREENSHOT_TIMEOUT_SECONDS,
            )
            if not data:
                raise RuntimeError(f"Screenshot returned no data for target {self._tab_id(target)}")
            return base64.b64decode(data)

        try:
            image = await capture(tab)
        except TimeoutError as exc:
            logger.warning("截图超时: target_id=%s", target_id)
            try:
                visibility = await asyncio.wait_for(tab.evaluate("document.visibilityState"), timeout=2)
            except Exception:
                visibility = None
            if visibility != "hidden":
                raise BrowserToolTimeout(f"Screenshot timed out for target {target_id}") from exc

            await asyncio.wait_for(self.browser.update_targets(), timeout=3)
            replacement = None
            for candidate in self.browser.tabs:
                if self._tab_id(candidate) == target_id:
                    continue
                try:
                    state = await asyncio.wait_for(candidate.evaluate("document.visibilityState"), timeout=2)
                except Exception:
                    continue
                if state == "visible":
                    replacement = candidate
                    break
            if replacement is None:
                raise BrowserToolTimeout(f"Screenshot timed out for hidden target {target_id}; no visible page") from exc
            await asyncio.wait_for(self._activate(replacement), timeout=3)
            tab = replacement
            logger.warning("截图切换到可见页面: old_target_id=%s target_id=%s", target_id, self._tab_id(tab))
            image = await capture(tab)

        path.write_bytes(image)
        return {"ok": True, "path": str(path), "target_id": self._tab_id(tab), **await self.state()}

    async def settle_and_screenshot(self) -> dict[str, Any]:
        from nodriver import cdp

        tab = self.tab
        pending: set[str] = set()
        last_activity = time.monotonic()

        def started(event: Any) -> None:
            nonlocal last_activity
            if event.type_ not in {cdp.network.ResourceType.WEB_SOCKET, cdp.network.ResourceType.EVENT_SOURCE}:
                pending.add(str(event.request_id))
                last_activity = time.monotonic()

        def finished(event: Any) -> None:
            nonlocal last_activity
            pending.discard(str(event.request_id))
            last_activity = time.monotonic()

        idle = False
        listening = False
        try:
            tab.add_handler(cdp.network.RequestWillBeSent, started)
            tab.add_handler(cdp.network.LoadingFinished, finished)
            tab.add_handler(cdp.network.LoadingFailed, finished)
            listening = True
            await tab.send(cdp.network.enable())
            await asyncio.sleep(1)
            deadline = time.monotonic() + NETWORK_IDLE_TIMEOUT_SECONDS
            while time.monotonic() < deadline:
                if not pending and time.monotonic() - last_activity >= NETWORK_QUIET_SECONDS:
                    idle = True
                    break
                await asyncio.sleep(0.1)
        except Exception:
            logger.exception("网络空闲监听失败，继续截图")
            await asyncio.sleep(1)
        finally:
            if listening:
                tab.remove_handler(cdp.network.RequestWillBeSent, started)
                tab.remove_handler(cdp.network.LoadingFinished, finished)
                tab.remove_handler(cdp.network.LoadingFailed, finished)
        result = await self.screenshot()
        logger.info("post_tool_screenshot idle=%s pending=%d path=%s", idle, len(pending), result["path"])
        return {**result, "network_idle": idle}


async def _main() -> None:
    _configure_utf8_stdio()
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    _configure_driver_logging(project_root)
    sys.path.insert(0, str(project_root / "src"))
    from boss_react import nodriver_backend

    session = NodriverToolSession(nodriver_backend, args.artifacts)
    logger.info("driver_config profile=%s start_url=%s tool_timeout=%.1fs", args.profile, args.url, args.tool_timeout)
    tools = {
        "browser_state": session.state, "browser_observe": session.observe,
        "browser_click_text": session.click_text, "browser_hover_text": session.hover_text,
        "browser_input_text": session.input_text, "browser_scroll": session.scroll,
        "browser_scroll_to_text": session.scroll_to_text, "browser_press": session.press,
        "browser_back": session.back, "browser_wait": session.wait,
        "browser_reset": session.reset,
        "browser_eval_js": session.eval_js, "browser_upload_text": session.upload_text,
        "browser_screenshot": session.screenshot, "boss_send_greeting": session.send_greeting,
        "boss_send_image": session.send_image,
        "browser_settle_and_screenshot": session.settle_and_screenshot,
        "browser_ensure_login": session.ensure_login_page,
    }
    try:
        await nodriver_backend.open_browser(args.url, args.profile)
        await session.initialize_tabs()
        print(json.dumps({"event": "ready", "tools": sorted(tools), **await session.state()}, ensure_ascii=False), flush=True)
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            request: dict[str, Any] | None = None
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    print(json.dumps({"ok": True, "closing": True}), flush=True)
                    break
                payload = await _execute_tool_request(session, tools, request, args.tool_timeout)
            except Exception as exc:  # noqa: BLE001
                logger.exception("request_parse_error request=%s", request)
                payload = {
                    "ok": False,
                    "request_id": request.get("request_id") if request else None,
                    "tool": request.get("tool") if request else None,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)
    finally:
        await nodriver_backend.shutdown()
        await asyncio.sleep(0.25)


if __name__ == "__main__":
    asyncio.run(_main())
