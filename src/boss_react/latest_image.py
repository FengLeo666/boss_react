"""Keep only the latest usable image in the agent's persisted messages."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AnyMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

_IMAGE_TYPES = {"image", "image_url"}
_OMITTED_IMAGE = {"type": "text", "text": "[Earlier image omitted from context]"}


def _is_image(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") in _IMAGE_TYPES


def _is_usable_image(block: dict[str, Any]) -> bool:
    if block["type"] == "image_url":
        source = block.get("image_url")
        if isinstance(source, dict):
            source = source.get("url")
        return isinstance(source, str) and bool(source.strip())
    return any(
        isinstance(block.get(key), str) and bool(block[key].strip())
        for key in ("base64", "url", "file_id")
    )


def _prune_images(messages: list[AnyMessage]) -> tuple[list[AnyMessage], int]:
    latest: tuple[int, int] | None = None
    for message_index, message in enumerate(messages):
        if isinstance(message.content, list):
            for block_index, block in enumerate(message.content):
                if _is_image(block) and _is_usable_image(block):
                    latest = (message_index, block_index)

    replacements: list[AnyMessage] = []
    removed = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message.content, list):
            continue
        content = []
        for block_index, block in enumerate(message.content):
            if _is_image(block) and (message_index, block_index) != latest:
                removed += 1
            else:
                content.append(block)
        if len(content) != len(message.content):
            replacements.append(
                message.model_copy(update={"content": content or [_OMITTED_IMAGE.copy()]})
            )
    return replacements, removed


class LatestImageMiddleware(AgentMiddleware):
    """Persist text and the latest valid image while dropping older images."""

    def before_agent(self, state: dict[str, Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        del runtime
        return self._prune(state)

    def before_model(self, state: dict[str, Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        del runtime
        return self._prune(state)

    @staticmethod
    def _prune(state: dict[str, Any]) -> dict[str, Any] | None:
        replacements, removed = _prune_images(state["messages"])
        if not removed:
            return None
        logger.info("已移除旧图片块: removed=%d", removed)
        return {"messages": replacements}
