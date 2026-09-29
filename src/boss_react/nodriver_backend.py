"""Self-contained nodriver lifecycle and BOSS chat operations."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import nodriver as uc
from nodriver import cdp

logger = logging.getLogger(__name__)

_browser: uc.Browser | None = None
_tab: uc.Tab | None = None


def get_tab() -> uc.Tab | None:
    return _tab


def get_browser() -> uc.Browser | None:
    return _browser


def set_tab(tab: uc.Tab) -> None:
    global _tab
    _tab = tab


def _ensure_localhost_bypasses_proxy() -> None:
    no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    values = [value.strip() for value in no_proxy.split(",") if value.strip()]
    lowered = {value.lower() for value in values}
    values.extend(host for host in ("127.0.0.1", "localhost") if host not in lowered)
    bypass = ",".join(values)
    os.environ["no_proxy"] = bypass
    os.environ["NO_PROXY"] = bypass


def _clear_profile_locks(profile_dir: Path) -> None:
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        try:
            (profile_dir / name).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("无法清理 Chrome profile 锁 %s: %s", name, exc)


async def open_browser(url: str, profile_dir: Path, attempts: int = 3) -> None:
    global _browser, _tab
    profile_dir = profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    _ensure_localhost_bypasses_proxy()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        _clear_profile_locks(profile_dir)
        config = uc.Config()
        config.user_data_dir = str(profile_dir)
        config.headless = False
        try:
            _browser = await uc.start(config=config)
            _tab = await _browser.get(url)
            await _tab.activate()
            await _tab.bring_to_front()
            await asyncio.sleep(2)
            logger.info("本地 nodriver 浏览器已启动: profile=%s", profile_dir)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("浏览器启动失败 (%d/%d): %s", attempt, attempts, exc)
            await shutdown()
            if attempt < attempts:
                await asyncio.sleep(2 * attempt)
    assert last_error is not None
    raise last_error


async def shutdown() -> None:
    global _browser, _tab
    browser, _browser, _tab = _browser, None, None
    if browser is not None:
        try:
            browser.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("关闭 nodriver 浏览器失败: %s", exc)


async def safe_evaluate(expression: str, timeout: float = 15) -> dict[str, Any]:
    if _tab is None:
        raise RuntimeError("nodriver tab is unavailable")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            raw = await asyncio.wait_for(_tab.evaluate(expression), timeout=timeout)
            if isinstance(raw, tuple):
                raw = raw[0]
            if isinstance(raw, str):
                value = json.loads(raw)
            else:
                value = raw
            if not isinstance(value, dict):
                raise TypeError(f"page evaluation returned {type(value).__name__}")
            return value
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(0.25)
    assert last_error is not None
    raise last_error


async def send_chat_message(text: str) -> None:
    if _tab is None:
        raise RuntimeError("nodriver tab is unavailable")
    chat = await _tab.select("#chat-input", timeout=10)
    if not chat:
        raise RuntimeError("chat input (#chat-input) 未找到")
    await chat.send_keys(text)
    await asyncio.sleep(0.5)
    result = await safe_evaluate(
        f"""
        JSON.stringify((() => {{
          const expected = {json.dumps(text, ensure_ascii=False)};
          const visible = (el) => {{
            if (!el) return false;
            const cs = getComputedStyle(el), r = el.getBoundingClientRect();
            return cs.display !== 'none' && cs.visibility !== 'hidden'
              && r.width > 0 && r.height > 0;
          }};
          const readInput = (el) => (
            ('value' in el ? el.value : '') || el.innerText || el.textContent || ''
          ).trim();
          const input = document.querySelector('#chat-input');
          if (!input) return {{ok: false, reason: 'input_not_found'}};
          input.focus();
          let current = readInput(input);
          if (!current.includes(expected.slice(0, Math.min(20, expected.length)))) {{
            if ('value' in input) input.value = expected;
            else input.textContent = expected;
            input.dispatchEvent(new InputEvent('input', {{bubbles: true, data: expected}}));
            input.dispatchEvent(new Event('change', {{bubbles: true}}));
            current = readInput(input);
          }}
          if (!current) return {{ok: false, reason: 'input_empty'}};
          const inputRect = input.getBoundingClientRect();
          const candidates = Array.from(document.querySelectorAll(
            'button, a, div, span, [role="button"]'
          )).map((el) => {{
            if (!visible(el) || el.disabled || el.getAttribute('aria-disabled') === 'true') return null;
            const txt = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, '');
            const cls = String(el.className || '');
            const marker = [cls, el.getAttribute('aria-label') || '', el.getAttribute('title') || '']
              .join(' ').toLowerCase();
            const r = el.getBoundingClientRect();
            const near = Math.abs((r.top + r.bottom - inputRect.top - inputRect.bottom) / 2) < 160;
            const media = /sendimg|image|img|picture|photo|upload|file|emoji|face/.test(marker);
            const exact = txt === '发送' || txt.toLowerCase() === 'send';
            const sendClass = /(^|[-_\\s])send($|[-_\\s])|btn-send|send-btn|chat-send/.test(marker);
            return near && !media && (exact || sendClass)
              ? {{el, score: exact ? 0 : 1, text: txt, className: cls}} : null;
          }}).filter(Boolean).sort((a, b) => a.score - b.score);
          const button = candidates[0]?.el;
          if (!button) return {{ok: false, reason: 'send_button_not_found', inputLen: current.length}};
          button.click();
          return {{ok: true, inputLen: current.length,
            clickedText: (button.innerText || button.textContent || '').trim(),
            clickedClass: String(button.className || '')}};
        }})())
        """
    )
    if not result.get("ok"):
        raise RuntimeError(f"发送按钮点击失败: {result}")
    await asyncio.sleep(1)


async def _file_input_node_id() -> int | None:
    if _tab is None:
        return None
    document = await _tab.send(cdp.dom.get_document(depth=1, pierce=True))
    for selector in (
        "input[type='file'][accept*='image']",
        "input[type='file'][accept*='png']",
        "input[type='file']",
    ):
        node_id = await _tab.send(cdp.dom.query_selector(document.node_id, selector))
        if node_id:
            return node_id
    return None


async def _click_image_button() -> dict[str, Any]:
    return await safe_evaluate(
        """
        JSON.stringify((() => {
          const visible = (el) => {
            if (!el) return false;
            const cs = getComputedStyle(el), r = el.getBoundingClientRect();
            return cs.display !== 'none' && cs.visibility !== 'hidden'
              && r.width > 0 && r.height > 0;
          };
          const inputRect = document.querySelector('#chat-input')?.getBoundingClientRect();
          const candidates = Array.from(document.querySelectorAll(
            'button, a, div, span, [role="button"]'
          )).map((el) => {
            if (!visible(el)) return null;
            const cls = String(el.className || '');
            const marker = [cls, el.getAttribute('aria-label') || '',
              el.getAttribute('title') || '', el.innerText || el.textContent || '']
              .join(' ').toLowerCase();
            if (!/sendimg|image|img|picture|photo|upload|file/.test(marker)) return null;
            const r = el.getBoundingClientRect();
            const near = !inputRect
              || Math.abs((r.top + r.bottom - inputRect.top - inputRect.bottom) / 2) < 180;
            return near ? {el} : null;
          }).filter(Boolean);
          const button = candidates[0]?.el;
          if (!button) return {ok: false, reason: 'image_button_not_found'};
          button.click();
          return {ok: true};
        })())
        """
    )


async def send_chat_image(image_path: str) -> None:
    if _tab is None:
        raise RuntimeError("nodriver tab is unavailable")
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"待发送图片不存在: {path}")

    node_id = await _file_input_node_id()
    if not node_id:
        await _tab.send(cdp.page.set_intercept_file_chooser_dialog(True, cancel=True))
        try:
            clicked = await _click_image_button()
        finally:
            await _tab.send(cdp.page.set_intercept_file_chooser_dialog(False))
        if not clicked.get("ok"):
            raise RuntimeError(f"图片按钮点击失败: {clicked}")
        for _ in range(20):
            node_id = await _file_input_node_id()
            if node_id:
                break
            await asyncio.sleep(0.25)
    if not node_id:
        raise RuntimeError("图片上传 input[type=file] 未找到")

    await _tab.send(cdp.dom.set_file_input_files([str(path)], node_id=node_id))
    await asyncio.sleep(2)
    result = await safe_evaluate(
        """
        JSON.stringify((() => {
          const visible = (el) => {
            if (!el) return false;
            const cs = getComputedStyle(el), r = el.getBoundingClientRect();
            return cs.display !== 'none' && cs.visibility !== 'hidden'
              && r.width > 0 && r.height > 0;
          };
          const candidates = Array.from(document.querySelectorAll(
            'button, a, div, span, [role="button"]'
          )).map((el) => {
            if (!visible(el) || el.disabled || el.getAttribute('aria-disabled') === 'true') return null;
            const txt = (el.innerText || el.textContent || '').trim().replace(/\\s+/g, '');
            const cls = String(el.className || '');
            const marker = [cls, el.getAttribute('aria-label') || '', el.getAttribute('title') || '']
              .join(' ').toLowerCase();
            const exact = ['发送', '确定', '确认'].includes(txt) || txt.toLowerCase() === 'send';
            const sendClass = /(^|[-_\\s])send($|[-_\\s])|btn-send|send-btn|chat-send/.test(marker);
            return exact || sendClass ? {el, score: exact ? 0 : 1} : null;
          }).filter(Boolean).sort((a, b) => a.score - b.score);
          const button = candidates[0]?.el;
          if (!button) return {ok: true, clicked: false, reason: 'no_confirm_button'};
          button.click();
          return {ok: true, clicked: true};
        })())
        """
    )
    if not result.get("ok"):
        raise RuntimeError(f"图片发送确认失败: {result}")
    await asyncio.sleep(1)
