"""LangChain middleware that exposes a persistent browser as agent tools."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolException, tool
from langchain_core.messages import ToolMessage
from playwright.async_api import Error as PlaywrightError

from .browser import BrowserConfig, BrowserSession


class PlaywrightBrowserMiddleware(AgentMiddleware):
    """Register browser tools and retain one Playwright session across runs."""

    name = "playwright_browser"

    def __init__(
        self,
        config: BrowserConfig | None = None,
        *,
        session: BrowserSession | None = None,
    ) -> None:
        super().__init__()
        self.session = session or BrowserSession(config)
        self.tools = self._build_tools()

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        """Ensure the browser exists before each agent invocation."""
        del state, runtime
        await self.session.start()

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[ToolMessage]],
    ) -> ToolMessage:
        """Turn expected browser failures into observations the agent can recover from."""
        try:
            return await handler(request)
        except (PlaywrightError, TimeoutError, ValueError, PermissionError, ToolException) as exc:
            return ToolMessage(
                content=f"Browser tool failed: {type(exc).__name__}: {exc}",
                tool_call_id=request.tool_call["id"],
                name=request.tool_call["name"],
                status="error",
            )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Expose the greeting tool to the model only while chat input is usable."""
        page_state = (await self.session.state())["page_state"]
        tools = request.tools
        if not page_state["capabilities"]["send_greeting"]:
            tools = [tool for tool in tools if getattr(tool, "name", None) != "boss_send_greeting"]
        return await handler(request.override(tools=tools))

    async def aclose(self) -> None:
        """Explicitly close the retained browser session."""
        await self.session.close()

    def _build_tools(self) -> list[Any]:
        session = self.session

        @tool("browser_observe")
        async def browser_observe(
            max_elements: int = 250,
            max_text_chars: int = 16_000,
            include_offscreen: bool = True,
        ) -> dict[str, Any]:
            """Read page text and enumerate visible interactive elements with e1/e2 references."""
            return await session.observe(
                max_elements=max_elements,
                max_text_chars=max_text_chars,
                include_offscreen=include_offscreen,
            )

        @tool("browser_state")
        async def browser_state() -> dict[str, Any]:
            """Report the current page kind and which state-gated actions are available."""
            return await session.state()

        @tool("boss_send_greeting")
        async def boss_send_greeting(message: str) -> dict[str, Any]:
            """Send a greeting in BOSS chat; available only when a visible #chat-input exists."""
            return await session.send_greeting(message)

        @tool("browser_click")
        async def browser_click(ref: str, wait_after_ms: int = 500) -> dict[str, Any]:
            """Click one element reference returned by browser_observe."""
            return await session.click(ref, wait_after_ms=wait_after_ms)

        @tool("browser_fill")
        async def browser_fill(
            ref: str,
            text: str,
            clear: bool = True,
            press_enter: bool = False,
        ) -> dict[str, Any]:
            """Fill or append text in an input reference; optionally press Enter."""
            return await session.fill(ref, text, clear=clear, press_enter=press_enter)

        @tool("browser_click_at")
        async def browser_click_at(x: int, y: int, wait_after_ms: int = 500) -> dict[str, Any]:
            """Click viewport coordinates from a recent screenshot when no element ref exists."""
            return await session.click_at(x, y, wait_after_ms=wait_after_ms)

        @tool("browser_hover")
        async def browser_hover(ref: str, wait_after_ms: int = 300) -> dict[str, Any]:
            """Hover an element reference to reveal menus, tooltips, or hidden controls."""
            return await session.hover(ref, wait_after_ms=wait_after_ms)

        @tool("browser_press")
        async def browser_press(key: str, ref: str | None = None) -> dict[str, Any]:
            """Press a Playwright key such as Enter, Escape, ArrowDown, or Control+A."""
            return await session.press(key, ref=ref)

        @tool("browser_select")
        async def browser_select(ref: str, values: list[str]) -> dict[str, Any]:
            """Select one or more option values in a select element."""
            return await session.select_option(ref, values)

        @tool("browser_upload")
        async def browser_upload(ref: str, paths: list[str]) -> dict[str, Any]:
            """Attach local files to a file input; paths must be inside configured upload roots."""
            return await session.upload(ref, paths)

        @tool("browser_scroll")
        async def browser_scroll(delta_y: int, ref: str | None = None) -> dict[str, Any]:
            """Scroll the page or a referenced scroll container vertically."""
            return await session.scroll(delta_y, ref=ref)

        @tool("browser_back")
        async def browser_back() -> dict[str, Any]:
            """Navigate the active tab back once."""
            return await session.back()

        @tool("browser_wait")
        async def browser_wait(milliseconds: int = 1000) -> dict[str, Any]:
            """Wait briefly for a page transition, animation, or async content."""
            return await session.wait(milliseconds)

        @tool("browser_tabs")
        async def browser_tabs() -> dict[str, Any]:
            """List all tabs and identify the active tab."""
            return await session.tabs()

        @tool("browser_switch_tab")
        async def browser_switch_tab(index: int) -> dict[str, Any]:
            """Switch to a tab index returned by browser_tabs."""
            return await session.switch_tab(index)

        @tool("browser_screenshot", response_format="content_and_artifact")
        async def browser_screenshot(
            full_page: bool = False,
            include_image: bool = True,
            annotate_elements: bool = True,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            """Capture a PNG for a vision model, optionally labeling controls with e1/e2 refs."""
            return await session.screenshot(
                full_page=full_page,
                include_image=include_image,
                annotate_elements=annotate_elements,
            )

        return [
            browser_observe,
            browser_state,
            boss_send_greeting,
            browser_click,
            browser_fill,
            browser_click_at,
            browser_hover,
            browser_press,
            browser_select,
            browser_upload,
            browser_scroll,
            browser_back,
            browser_wait,
            browser_tabs,
            browser_switch_tab,
            browser_screenshot,
        ]
