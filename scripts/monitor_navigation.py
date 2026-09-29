"""Observe one page through raw CDP without browser-level auto-attach."""

from __future__ import annotations

import argparse
import asyncio
import json
import urllib.request
from datetime import UTC, datetime

import websockets
from websockets.exceptions import ConnectionClosed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdp-http", required=True)
    parser.add_argument("--url-contains", default="zhipin.com")
    parser.add_argument("--runtime-diagnostics", action="store_true")
    return parser.parse_args()


def _stamp() -> str:
    return datetime.now(tz=UTC).astimezone().strftime("%H:%M:%S.%f")[:-3]


def _log(event: str, detail: str) -> None:
    print(f"{_stamp()} {event:<18} {detail}", flush=True)


def _targets(cdp_http: str) -> list[dict]:
    with urllib.request.urlopen(f"{cdp_http.rstrip('/')}/json/list", timeout=3) as response:
        return json.load(response)


async def _find_target(cdp_http: str, url_contains: str) -> dict:
    while True:
        targets = await asyncio.to_thread(_targets, cdp_http)
        for target in targets:
            if target.get("type") == "page" and url_contains in target.get("url", ""):
                return target
        _log("waiting", f"no page URL containing {url_contains!r}")
        await asyncio.sleep(1)


async def _main() -> None:
    args = _parse_args()
    target = await _find_target(args.cdp_http, args.url_contains)
    _log("watching", f"target={target['id']} url={target['url']}")

    async with websockets.connect(
        target["webSocketDebuggerUrl"],
        origin="http://localhost",
        max_size=None,
    ) as socket:
        commands = [
            {"id": 1, "method": "Page.enable"},
            {"id": 2, "method": "Network.enable"},
            {"id": 3, "method": "Page.setLifecycleEventsEnabled", "params": {"enabled": True}},
        ]
        if args.runtime_diagnostics:
            commands.extend(
                [
                    {"id": 4, "method": "Runtime.enable"},
                    {"id": 5, "method": "Log.enable"},
                    {
                "id": 6,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": """
                        (() => {
                          const originalClose = window.close.bind(window);
                          window.close = (...args) => {
                            console.warn('[boss-react-monitor] window.close called', new Error().stack);
                            return originalClose(...args);
                          };
                          addEventListener('beforeunload', () => {
                            console.warn('[boss-react-monitor] beforeunload', location.href);
                          });
                          addEventListener('pagehide', (event) => {
                            console.warn('[boss-react-monitor] pagehide', event.persisted, location.href);
                          });
                          return true;
                        })()
                    """,
                    "returnByValue": True,
                },
                    },
                ]
            )
        for command in commands:
            await socket.send(json.dumps(command))
        _log("ready", "raw page CDP listener attached; no navigation commands sent")

        try:
            async for raw_message in socket:
                message = json.loads(raw_message)
                method = message.get("method")
                params = message.get("params", {})
                if method == "Network.requestWillBeSent" and params.get("type") == "Document":
                    request = params.get("request", {})
                    redirect = params.get("redirectResponse")
                    if redirect:
                        _log("redirect", f"{redirect.get('status')} -> {request.get('url')}")
                    else:
                        _log("document request", f"{request.get('method')} {request.get('url')}")
                elif method == "Network.responseReceived" and params.get("type") == "Document":
                    response = params.get("response", {})
                    _log("document response", f"{response.get('status')} {response.get('url')}")
                elif method == "Page.frameNavigated":
                    frame = params.get("frame", {})
                    if not frame.get("parentId"):
                        _log("frame navigated", frame.get("url", ""))
                elif method == "Page.lifecycleEvent":
                    name = params.get("name")
                    if name in {"DOMContentLoaded", "load"}:
                        _log("lifecycle", name)
                elif method == "Runtime.consoleAPICalled":
                    values = [item.get("value", item.get("description", "")) for item in params.get("args", [])]
                    if any("boss-react-monitor" in str(value) for value in values):
                        _log("console", " | ".join(str(value) for value in values))
                elif method == "Inspector.detached":
                    _log("inspector detached", str(params))
        except ConnectionClosed as exc:
            _log("disconnected", f"code={exc.code} reason={exc.reason!r}")


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
