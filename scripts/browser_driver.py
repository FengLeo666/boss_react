"""Keep one PlaywrightBrowserMiddleware alive and invoke its tools as JSON lines."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from boss_react import BrowserConfig, PlaywrightBrowserMiddleware


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=Path("browser-profile"))
    parser.add_argument("--cdp-url")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--start-url", default="https://www.zhipin.com/")
    parser.add_argument("--allowed-host", action="append", default=[])
    parser.add_argument("--upload-root", type=Path, action="append", default=[])
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def _json_default(value: Any) -> str:
    return str(value)


async def _main() -> None:
    args = _parse_args()
    middleware = PlaywrightBrowserMiddleware(
        BrowserConfig(
            user_data_dir=args.profile,
            artifacts_dir=args.artifacts,
            headless=args.headless,
            browser_channel="chrome",
            cdp_url=args.cdp_url,
            start_url=args.start_url,
            allowed_hosts=tuple(args.allowed_host),
            upload_roots=tuple(args.upload_root),
        )
    )
    tools = {item.name: item for item in middleware.tools}
    try:
        await middleware.session.start()
        print(
            json.dumps(
                {
                    "event": "ready",
                    "tools": sorted(tools),
                    "profile": str(args.profile.resolve()),
                    "cdp_url": args.cdp_url,
                    "start_url": middleware.session.config.start_url,
                    "current_url": middleware.session.page.url if middleware.session.page else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        while True:
            line = await asyncio.to_thread(sys.stdin.readline)
            if not line:
                break
            try:
                request = json.loads(line)
                if request.get("command") == "close":
                    print(json.dumps({"ok": True, "closing": True}), flush=True)
                    break
                name = request["tool"]
                if name not in tools:
                    raise ValueError(f"Unknown tool: {name}")
                result = await tools[name].ainvoke(request.get("args", {}))
                payload = {"ok": True, "tool": name, "result": result}
            except Exception as exc:  # noqa: BLE001 - the driver must stay alive for inspection
                payload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            print(json.dumps(payload, ensure_ascii=False, default=_json_default), flush=True)
    finally:
        await middleware.aclose()


if __name__ == "__main__":
    asyncio.run(_main())
