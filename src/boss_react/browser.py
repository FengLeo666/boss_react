"""Long-lived Playwright session used by the LangChain middleware."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

_REF_RE = re.compile(r"^e[1-9][0-9]*$")
_ELEMENT_SCRIPT = r"""
({maxElements, includeOffscreen}) => {
  const selector = [
    'button', 'input', 'textarea', 'select', 'a[href]',
    '[role="button"]', '[role="link"]', '[role="textbox"]',
    '[role="checkbox"]', '[role="radio"]', '[role="combobox"]',
    '[role="menuitem"]', '[contenteditable="true"]', '[tabindex]'
  ].join(',');

  const textOf = (el) => {
    const raw = el.innerText || el.textContent || '';
    return raw.replace(/\s+/g, ' ').trim().slice(0, 240);
  };

  const visible = (el) => {
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    if (style.visibility === 'hidden' || style.display === 'none') return false;
    if (Number(style.opacity) === 0 || rect.width <= 0 || rect.height <= 0) return false;
    if (includeOffscreen) return true;
    return rect.bottom >= 0 && rect.right >= 0 && rect.top <= innerHeight && rect.left <= innerWidth;
  };

  const elements = Array.from(document.querySelectorAll(selector)).filter(visible);
  document.querySelectorAll('[data-boss-react-ref]').forEach((el) => {
    el.removeAttribute('data-boss-react-ref');
  });
  return elements.slice(0, maxElements).map((el, index) => {
    const ref = `e${index + 1}`;
    el.setAttribute('data-boss-react-ref', ref);
    const rect = el.getBoundingClientRect();
    const type = el.getAttribute('type') || '';
    const value = type.toLowerCase() === 'password' ? '<redacted>' : (el.value || '');
    return {
      ref,
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      role: el.getAttribute('role') || '',
      type,
      text: textOf(el),
      aria_label: el.getAttribute('aria-label') || '',
      title: el.getAttribute('title') || '',
      placeholder: el.getAttribute('placeholder') || '',
      name: el.getAttribute('name') || '',
      value: String(value).slice(0, 240),
      disabled: Boolean(el.disabled) || el.getAttribute('aria-disabled') === 'true',
      checked: typeof el.checked === 'boolean' ? el.checked : null,
      href: el.href || '',
      x: Math.round(rect.x),
      y: Math.round(rect.y),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    };
  });
}
"""

_TEXT_SCRIPT = r"""
({maxChars}) => {
  const root = document.querySelector('main, [role="main"]') || document.body;
  if (!root) return '';
  const text = (root.innerText || '').replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  return text.slice(0, maxChars);
}
"""

_OVERLAY_SCRIPT = r"""
() => {
  document.querySelectorAll('[data-boss-react-overlay]').forEach((node) => node.remove());
  for (const el of document.querySelectorAll('[data-boss-react-ref]')) {
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) continue;
    const marker = document.createElement('div');
    marker.setAttribute('data-boss-react-overlay', 'true');
    marker.textContent = el.getAttribute('data-boss-react-ref');
    Object.assign(marker.style, {
      position: 'absolute',
      left: `${Math.max(0, rect.left + window.scrollX)}px`,
      top: `${Math.max(0, rect.top + window.scrollY)}px`,
      zIndex: '2147483647',
      pointerEvents: 'none',
      color: '#ffffff',
      background: '#d40000',
      border: '1px solid #ffffff',
      borderRadius: '2px',
      padding: '1px 3px',
      font: 'bold 12px/16px monospace',
      boxShadow: '0 1px 3px rgba(0,0,0,.65)',
    });
    document.documentElement.appendChild(marker);
  }
}
"""

_REMOVE_OVERLAY_SCRIPT = "document.querySelectorAll('[data-boss-react-overlay]').forEach((node) => node.remove())"

_BOSS_DEVTOOLS_COMPAT_SCRIPT = r"""
(() => {
  if (!(location.hostname === 'zhipin.com' || location.hostname.endsWith('.zhipin.com'))) return;
  if (window.__bossReactDevtoolsCompat) return;
  Object.defineProperty(window, '__bossReactDevtoolsCompat', {value: true});

  const fromDestructiveDevtoolsHandler = () => {
    const stack = String(new Error().stack || '');
    return /\bat Bm\b/.test(stack) || /main\.js:\d+:66\d{4}/.test(stack);
  };

  const originalClose = window.close.bind(window);
  window.close = (...args) => {
    if (fromDestructiveDevtoolsHandler()) return undefined;
    return originalClose(...args);
  };

  const originalBack = history.back.bind(history);
  history.back = (...args) => {
    if (fromDestructiveDevtoolsHandler()) return undefined;
    return originalBack(...args);
  };

  const originalSetTimeout = window.setTimeout.bind(window);
  window.setTimeout = (callback, delay, ...args) => {
    if (fromDestructiveDevtoolsHandler()) return 0;
    return originalSetTimeout(callback, delay, ...args);
  };
})();
"""

_SEND_GREETING_SCRIPT = r"""
() => {
  const input = document.querySelector('#chat-input');
  if (!input) return {ok: false, reason: 'chat_input_not_found'};
  const visible = (el) => {
    const style = getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== 'none' && style.visibility !== 'hidden'
      && Number(style.opacity) !== 0 && rect.width > 0 && rect.height > 0;
  };
  const inputRect = input.getBoundingClientRect();
  const candidates = Array.from(document.querySelectorAll(
    'button, a, div, span, [role="button"]'
  )).map((el) => {
    if (!visible(el) || el.disabled || el.getAttribute('aria-disabled') === 'true') return null;
    const text = (el.innerText || el.textContent || '').trim().replace(/\s+/g, '');
    const marker = [
      el.id || '',
      String(el.className || ''),
      el.getAttribute('aria-label') || '',
      el.getAttribute('title') || '',
    ].join(' ').toLowerCase();
    const rect = el.getBoundingClientRect();
    const nearInput = Math.abs(
      (rect.top + rect.bottom) / 2 - (inputRect.top + inputRect.bottom) / 2
    ) < 160;
    const isUploadControl = /image|img|picture|photo|upload|file|emoji|face/.test(marker);
    const exactText = text === '发送' || text.toLowerCase() === 'send';
    const sendClass = /(^|[-_\s])send($|[-_\s])|btn-send|send-btn|chat-send/.test(marker);
    if (!nearInput || isUploadControl || (!exactText && !sendClass)) return null;
    return {el, score: exactText ? 0 : 1, text};
  }).filter(Boolean).sort((left, right) => left.score - right.score);
  if (!candidates.length) return {ok: false, reason: 'send_button_not_found'};
  candidates[0].el.click();
  return {ok: true, clicked_text: candidates[0].text};
}
"""


@dataclass(slots=True)
class BrowserConfig:
    """Runtime configuration for one persistent browser session."""

    user_data_dir: Path = Path("browser-profile")
    artifacts_dir: Path = Path("artifacts")
    headless: bool = False
    browser_channel: str | None = "chrome"
    cdp_url: str | None = None
    start_url: str = "https://www.zhipin.com/"
    viewport: tuple[int, int] | None = (1440, 1000)
    navigation_timeout_ms: int = 45_000
    action_timeout_ms: int = 15_000
    allowed_hosts: tuple[str, ...] = ()
    upload_roots: tuple[Path, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.user_data_dir = Path(self.user_data_dir).resolve()
        self.artifacts_dir = Path(self.artifacts_dir).resolve()
        self.upload_roots = tuple(Path(path).resolve() for path in self.upload_roots)


class BrowserSession:
    """Own a Playwright process, persistent context, and active page.

    All operations are serialized because Playwright page state is inherently
    sequential and LangChain models may request multiple tools in parallel.
    """

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self.playwright: Playwright | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._start_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._started = False
        self._screenshot_counter = 0
        self._references_page: Page | None = None
        self._references: dict[str, dict[str, Any]] = {}

    @property
    def started(self) -> bool:
        return self._started and self.context is not None

    async def start(self) -> None:
        async with self._start_lock:
            if self.started:
                return
            self.config.user_data_dir.mkdir(parents=True, exist_ok=True)
            self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.playwright = await async_playwright().start()
            if self.config.cdp_url:
                try:
                    self.browser = await self.playwright.chromium.connect_over_cdp(self.config.cdp_url)
                    if not self.browser.contexts:
                        raise RuntimeError("CDP browser has no browser context")
                    self.context = self.browser.contexts[0]
                    self.context.set_default_timeout(self.config.action_timeout_ms)
                    self.context.set_default_navigation_timeout(self.config.navigation_timeout_ms)
                    await self.context.add_init_script(_BOSS_DEVTOOLS_COMPAT_SCRIPT)
                    pages = [page for page in self.context.pages if not page.is_closed()]
                    self.page = pages[-1] if pages else await self.context.new_page()
                    self._started = True
                    return
                except Exception:
                    await self._close_unlocked()
                    raise

            launch_options: dict[str, Any] = {
                "headless": self.config.headless,
            }
            if self.config.browser_channel:
                launch_options["channel"] = self.config.browser_channel
            if self.config.viewport is None:
                launch_options["no_viewport"] = True
            else:
                launch_options["viewport"] = {
                    "width": self.config.viewport[0],
                    "height": self.config.viewport[1],
                }
            try:
                self.context = await self.playwright.chromium.launch_persistent_context(
                    str(self.config.user_data_dir),
                    **launch_options,
                )
                self.context.set_default_timeout(self.config.action_timeout_ms)
                self.context.set_default_navigation_timeout(self.config.navigation_timeout_ms)
                await self.context.add_init_script(_BOSS_DEVTOOLS_COMPAT_SCRIPT)
                self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
                if self.config.start_url != "about:blank" or self.page.url == "about:blank":
                    await self.page.goto(self.config.start_url, wait_until="domcontentloaded")
                self._started = True
            except Exception:
                await self._close_unlocked()
                raise

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def _close_unlocked(self) -> None:
        if self.context is not None and self.config.cdp_url is None:
            await self.context.close()
        if self.playwright is not None:
            await self.playwright.stop()
        self.page = None
        self.context = None
        self.browser = None
        self.playwright = None
        self._started = False
        self._references_page = None
        self._references = {}

    async def _active_page(self) -> Page:
        if not self.started:
            await self.start()
        assert self.context is not None
        if self.page is None or self.page.is_closed():
            pages = [page for page in self.context.pages if not page.is_closed()]
            self.page = pages[-1] if pages else await self.context.new_page()
        return self.page

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https", "about", "data", "file"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme or '<missing>'}")
        if self.config.allowed_hosts and parsed.scheme in {"http", "https"}:
            host = (parsed.hostname or "").lower()
            allowed = any(host == item or host.endswith(f".{item}") for item in self.config.allowed_hosts)
            if not allowed:
                raise ValueError(f"Host is not allowed: {host}")

    async def open(self, url: str) -> dict[str, Any]:
        self._validate_url(url)
        async with self._lock:
            page = await self._active_page()
            await page.goto(url, wait_until="domcontentloaded")
            return await self._page_summary(page)

    async def observe(
        self,
        *,
        max_elements: int = 250,
        max_text_chars: int = 16_000,
        include_offscreen: bool = True,
    ) -> dict[str, Any]:
        max_elements = max(1, min(max_elements, 500))
        max_text_chars = max(0, min(max_text_chars, 50_000))
        async with self._lock:
            page = await self._active_page()
            await page.wait_for_load_state("domcontentloaded")
            elements = await page.evaluate(
                _ELEMENT_SCRIPT,
                {"maxElements": max_elements, "includeOffscreen": include_offscreen},
            )
            self._remember_references(page, elements)
            text = await page.evaluate(_TEXT_SCRIPT, {"maxChars": max_text_chars}) if max_text_chars else ""
            result = await self._page_summary(page)
            result.update(
                {
                    "text": text,
                    "elements": elements,
                    "element_count": len(elements),
                    "truncated_elements": len(elements) >= max_elements,
                    "truncated_text": len(text) >= max_text_chars if max_text_chars else False,
                }
            )
            return result

    async def state(self) -> dict[str, Any]:
        """Return the active page state and state-gated capabilities."""
        async with self._lock:
            page = await self._active_page()
            return await self._page_summary(page)

    async def send_greeting(self, message: str) -> dict[str, Any]:
        """Send one greeting, but only when a usable BOSS chat composer is present."""
        message = message.strip()
        if not message:
            raise ValueError("Greeting message must not be empty")
        if len(message) > 500:
            raise ValueError("Greeting message must be at most 500 characters")

        async with self._lock:
            page = await self._active_page()
            page_state = await self._detect_page_state(page)
            if not page_state["capabilities"]["send_greeting"]:
                raise ValueError(
                    "boss_send_greeting is unavailable: active page is "
                    f"{page_state['kind']!r}, not a chat page with #chat-input"
                )

            chat_input = page.locator("#chat-input")
            await chat_input.fill(message)
            result = await page.evaluate(_SEND_GREETING_SCRIPT)
            if not result.get("ok"):
                raise ValueError(f"Greeting was not sent: {result.get('reason', 'unknown error')}")
            await page.wait_for_timeout(500)
            return {
                "ok": True,
                "characters": len(message),
                "clicked_text": result.get("clicked_text", ""),
                **await self._page_summary(page),
            }

    async def click(self, ref: str, *, wait_after_ms: int = 500) -> dict[str, Any]:
        async with self._lock:
            page = await self._active_page()
            locator = await self._ref_locator(page, ref)
            assert self.context is not None
            pages_before = set(self.context.pages)
            await locator.click()
            if wait_after_ms:
                await page.wait_for_timeout(max(0, min(wait_after_ms, 10_000)))
            new_pages = [candidate for candidate in self.context.pages if candidate not in pages_before]
            self.page = new_pages[-1] if new_pages else page
            if not self.page.is_closed():
                await self.page.wait_for_load_state("domcontentloaded")
            return {"ok": True, "clicked": ref, **await self._page_summary(self.page)}

    async def click_at(self, x: int, y: int, *, wait_after_ms: int = 500) -> dict[str, Any]:
        if x < 0 or y < 0:
            raise ValueError("Click coordinates must be non-negative")
        async with self._lock:
            page = await self._active_page()
            viewport = await page.evaluate("() => ({width: innerWidth, height: innerHeight})")
            if x > viewport["width"] or y > viewport["height"]:
                raise ValueError(f"Coordinates ({x}, {y}) are outside viewport {viewport}")
            await page.mouse.click(x, y)
            if wait_after_ms:
                await page.wait_for_timeout(max(0, min(wait_after_ms, 10_000)))
            return {"ok": True, "x": x, "y": y, **await self._page_summary(page)}

    async def hover(self, ref: str, *, wait_after_ms: int = 300) -> dict[str, Any]:
        async with self._lock:
            page = await self._active_page()
            locator = await self._ref_locator(page, ref)
            await locator.hover()
            if wait_after_ms:
                await page.wait_for_timeout(max(0, min(wait_after_ms, 5_000)))
            return {"ok": True, "hovered": ref, **await self._page_summary(page)}

    async def fill(
        self,
        ref: str,
        text: str,
        *,
        clear: bool = True,
        press_enter: bool = False,
    ) -> dict[str, Any]:
        async with self._lock:
            page = await self._active_page()
            locator = await self._ref_locator(page, ref)
            if clear:
                await locator.fill(text)
            else:
                await locator.press_sequentially(text)
            if press_enter:
                await locator.press("Enter")
            return {
                "ok": True,
                "filled": ref,
                "characters": len(text),
                "pressed_enter": press_enter,
                **await self._page_summary(page),
            }

    async def press(self, key: str, *, ref: str | None = None) -> dict[str, Any]:
        async with self._lock:
            page = await self._active_page()
            if ref:
                locator = await self._ref_locator(page, ref)
                await locator.press(key)
            else:
                await page.keyboard.press(key)
            return {"ok": True, "key": key, "target": ref or "page", **await self._page_summary(page)}

    async def select_option(self, ref: str, values: list[str]) -> dict[str, Any]:
        async with self._lock:
            page = await self._active_page()
            locator = await self._ref_locator(page, ref)
            selected = await locator.select_option(values)
            return {"ok": True, "selected": selected, "target": ref, **await self._page_summary(page)}

    async def upload(self, ref: str, paths: list[str]) -> dict[str, Any]:
        resolved = self._validate_upload_paths(paths)
        async with self._lock:
            page = await self._active_page()
            locator = await self._ref_locator(page, ref)
            input_type = (await locator.get_attribute("type") or "").lower()
            if await locator.evaluate("el => el.tagName.toLowerCase()") == "input" and input_type == "file":
                await locator.set_input_files([str(path) for path in resolved])
                mode = "input"
            else:
                async with page.expect_file_chooser() as chooser_info:
                    await locator.click()
                chooser = await chooser_info.value
                await chooser.set_files([str(path) for path in resolved])
                mode = "file_chooser"
            return {
                "ok": True,
                "target": ref,
                "mode": mode,
                "files": [path.name for path in resolved],
                **await self._page_summary(page),
            }

    async def scroll(self, delta_y: int, *, ref: str | None = None) -> dict[str, Any]:
        delta_y = max(-10_000, min(delta_y, 10_000))
        async with self._lock:
            page = await self._active_page()
            if ref:
                locator = await self._ref_locator(page, ref)
                await locator.evaluate("(el, value) => el.scrollBy(0, value)", delta_y)
            else:
                await page.mouse.wheel(0, delta_y)
            await page.wait_for_timeout(250)
            return {"ok": True, "delta_y": delta_y, "target": ref or "page", **await self._page_summary(page)}

    async def back(self) -> dict[str, Any]:
        async with self._lock:
            page = await self._active_page()
            await page.go_back(wait_until="domcontentloaded")
            return {"ok": True, **await self._page_summary(page)}

    async def wait(self, milliseconds: int = 1000) -> dict[str, Any]:
        milliseconds = max(0, min(milliseconds, 30_000))
        async with self._lock:
            page = await self._active_page()
            await page.wait_for_timeout(milliseconds)
            return {"ok": True, "waited_ms": milliseconds, **await self._page_summary(page)}

    async def tabs(self) -> dict[str, Any]:
        async with self._lock:
            await self._active_page()
            assert self.context is not None
            pages = [page for page in self.context.pages if not page.is_closed()]
            return {
                "active_index": pages.index(self.page) if self.page in pages else 0,
                "tabs": [
                    {"index": index, "title": await page.title(), "url": page.url} for index, page in enumerate(pages)
                ],
            }

    async def switch_tab(self, index: int) -> dict[str, Any]:
        async with self._lock:
            await self._active_page()
            assert self.context is not None
            pages = [page for page in self.context.pages if not page.is_closed()]
            if index < 0 or index >= len(pages):
                raise ValueError(f"Tab index out of range: {index}; available: 0..{len(pages) - 1}")
            self.page = pages[index]
            await self.page.bring_to_front()
            return {"ok": True, "active_index": index, **await self._page_summary(self.page)}

    async def screenshot(
        self,
        *,
        full_page: bool = False,
        include_image: bool = True,
        annotate_elements: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        async with self._lock:
            page = await self._active_page()
            self._screenshot_counter += 1
            path = self.config.artifacts_dir / f"screenshot-{self._screenshot_counter:04d}.png"
            if annotate_elements:
                elements = await page.evaluate(
                    _ELEMENT_SCRIPT,
                    {"maxElements": 250, "includeOffscreen": full_page},
                )
                self._remember_references(page, elements)
                await page.evaluate(_OVERLAY_SCRIPT)
            try:
                raw = await page.screenshot(path=str(path), full_page=full_page, type="png")
            finally:
                if annotate_elements:
                    await page.evaluate(_REMOVE_OVERLAY_SCRIPT)
            summary = await self._page_summary(page)
            metadata = {
                "path": str(path),
                "bytes": len(raw),
                "full_page": full_page,
                "annotated_elements": annotate_elements,
                **summary,
            }
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": json.dumps(metadata, ensure_ascii=False),
                }
            ]
            if include_image:
                content.append(
                    {
                        "type": "image",
                        "base64": base64.b64encode(raw).decode("ascii"),
                        "mime_type": "image/png",
                    }
                )
            return content, metadata

    async def _page_summary(self, page: Page) -> dict[str, Any]:
        assert self.context is not None
        pages = [candidate for candidate in self.context.pages if not candidate.is_closed()]
        state = await self._detect_page_state(page)
        return {
            "url": page.url,
            "title": await page.title(),
            "tab_index": pages.index(page) if page in pages else 0,
            "tab_count": len(pages),
            "page_state": state,
        }

    async def _detect_page_state(self, page: Page) -> dict[str, Any]:
        parsed = urlparse(page.url)
        host = (parsed.hostname or "").lower()
        path = parsed.path.rstrip("/") or "/"
        chat_input = page.locator("#chat-input")
        chat_input_ready = await chat_input.count() == 1 and await chat_input.is_visible()

        if "security" in path or "verify" in path or "captcha" in path:
            kind = "security_verification"
            reason = "URL indicates a security or verification page"
        elif "passport" in host or "/login" in path:
            kind = "login"
            reason = "URL indicates a login page"
        elif chat_input_ready or "/web/geek/chat" in path:
            kind = "chat"
            reason = "BOSS chat URL or visible #chat-input detected"
        elif path.startswith("/web/geek/jobs"):
            kind = "job_search"
            reason = "BOSS job search page detected"
        elif host == "zhipin.com" or host.endswith(".zhipin.com"):
            kind = "boss_home" if path == "/" else "boss_page"
            reason = "BOSS homepage detected" if path == "/" else "BOSS page detected"
        elif page.url == "about:blank":
            kind = "blank"
            reason = "Blank browser tab"
        else:
            kind = "other"
            reason = "Page does not match a known BOSS state"

        return {
            "kind": kind,
            "reason": reason,
            "capabilities": {
                "general_browser_actions": True,
                "send_greeting": kind == "chat" and chat_input_ready,
            },
            "available_special_tools": ["boss_send_greeting"] if kind == "chat" and chat_input_ready else [],
        }

    async def _ref_locator(self, page: Page, ref: str):
        if not _REF_RE.fullmatch(ref):
            raise ValueError(f"Invalid element reference: {ref!r}")
        locator = page.locator(f'[data-boss-react-ref="{ref}"]')
        if await locator.count() == 1:
            return locator

        descriptor = self._references.get(ref) if self._references_page is page else None
        if descriptor is None:
            raise ValueError(f"Element {ref} is missing or ambiguous; call browser_observe again")

        candidates = []
        if descriptor.get("id"):
            candidates.append(page.locator(f"[id={json.dumps(descriptor['id'])}]"))
        if descriptor.get("name"):
            candidates.append(page.locator(f"{descriptor['tag']}[name={json.dumps(descriptor['name'])}]"))
        if descriptor.get("href"):
            candidates.append(page.locator(f"a[href={json.dumps(descriptor['href'])}]"))
        if descriptor.get("aria_label"):
            candidates.append(page.locator(f"[aria-label={json.dumps(descriptor['aria_label'])}]"))
        if descriptor.get("placeholder"):
            candidates.append(page.locator(f"[placeholder={json.dumps(descriptor['placeholder'])}]"))
        if descriptor.get("title"):
            candidates.append(page.locator(f"[title={json.dumps(descriptor['title'])}]"))

        for candidate in candidates:
            if await candidate.count() == 1:
                return candidate

        text = descriptor.get("text", "")
        if text:
            text_candidates = page.locator(descriptor["tag"])
            exact_matches = []
            for index in range(min(await text_candidates.count(), 500)):
                candidate = text_candidates.nth(index)
                candidate_text = await candidate.evaluate(
                    "el => (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 240)"
                )
                if candidate_text == text:
                    exact_matches.append(candidate)
            if len(exact_matches) == 1:
                return exact_matches[0]

        raise ValueError(f"Element {ref} changed and could not be relocated; call browser_observe again")

    def _remember_references(self, page: Page, elements: list[dict[str, Any]]) -> None:
        self._references_page = page
        self._references = {element["ref"]: element for element in elements}

    def _validate_upload_paths(self, paths: list[str]) -> list[Path]:
        if not paths:
            raise ValueError("At least one upload path is required")
        if not self.config.upload_roots:
            raise PermissionError("Uploads are disabled; configure BrowserConfig.upload_roots")
        resolved: list[Path] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Upload path is not a file: {path}")
            allowed = any(path == root or path.is_relative_to(root) for root in self.config.upload_roots)
            if not allowed:
                raise PermissionError(f"Upload path is outside configured roots: {path}")
            resolved.append(path)
        return resolved
