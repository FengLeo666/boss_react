"""Agent-node context compaction with a cache-friendly trailing prompt."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    hook_config,
)
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)

_ROLE_KEY = "boss_react_role"
_RESUME_ROLE = "resume"
_TASK_ROLE = "task"
_COMPACTION_REQUEST_ROLE = "compaction_request"
_COMPACTED_CONTEXT_ROLE = "compacted_context"


class CompactionAgentState(AgentState):
    manual_compact: NotRequired[bool]

COMPACTION_PROMPT = """你现在只执行上下文压缩，不执行求职任务。
禁止调用任何工具，也不要提出新的操作计划。请把此前对话中继续完成任务所必需的信息压缩为一份简洁但无损的上下文，尤其保留：已查看和操作过的岗位、页面状态、已发送内容、失败原因、用户约束、尚未完成的步骤，以及避免重复操作所需的信息。

第一条简历消息和本次最新任务消息会由程序单独保留，不要复述简历全文，也不要把本次任务改写成新任务。只输出压缩后的历史上下文，不要添加前言、结语或工具调用。"""


def resume_message(content: str) -> HumanMessage:
    return HumanMessage(content=content, additional_kwargs={_ROLE_KEY: _RESUME_ROLE})


def task_message(content: str) -> HumanMessage:
    return HumanMessage(content=content, additional_kwargs={_ROLE_KEY: _TASK_ROLE})


def _role(message: AnyMessage) -> str:
    return str(message.additional_kwargs.get(_ROLE_KEY, ""))


def _has_role(message: AnyMessage, role: str) -> bool:
    return isinstance(message, HumanMessage) and _role(message) == role


def _response_messages(response: Any) -> list[AnyMessage]:
    if isinstance(response, AIMessage):
        return [response]
    if isinstance(response, ExtendedModelResponse):
        return list(response.model_response.result)
    if isinstance(response, ModelResponse):
        return list(response.result)
    raise TypeError(f"Unsupported model response type: {type(response).__name__}")


def _image_count(messages: list[AnyMessage]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, dict) and block.get("type") in {"image", "image_url"}
    )


class AgentNodeCompactionMiddleware(AgentMiddleware):
    """Compact state through the agent's own model node before normal execution."""

    state_schema = CompactionAgentState

    def __init__(self, trigger_tokens: int, trigger_images: int = 255) -> None:
        if trigger_tokens <= 0:
            raise ValueError("trigger_tokens must be positive")
        if trigger_images <= 0:
            raise ValueError("trigger_images must be positive")
        self.trigger_tokens = trigger_tokens
        self.trigger_images = trigger_images
        self._started_at: float | None = None

    def before_agent(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        del runtime
        messages = state["messages"]
        total_tokens = count_tokens_approximately(messages)
        total_images = _image_count(messages)
        already_requested = any(
            _has_role(message, _COMPACTION_REQUEST_ROLE) for message in messages
        )
        manual = bool(state.get("manual_compact", False))
        should_compact = (
            manual or total_tokens >= self.trigger_tokens or total_images >= self.trigger_images
        ) and not already_requested
        logger.info(
            "上下文检查(before_agent): messages=%d approx_tokens=%d token_trigger=%d images=%d image_trigger=%d manual=%s compact=%s",
            len(messages),
            total_tokens,
            self.trigger_tokens,
            total_images,
            self.trigger_images,
            manual,
            should_compact,
        )
        if not should_compact:
            return None

        self._started_at = time.perf_counter()
        logger.warning(
            "开始上下文压缩: messages=%d approx_tokens=%d images=%d",
            len(messages), total_tokens, total_images,
        )
        return {
            "messages": [
                HumanMessage(
                    content=COMPACTION_PROMPT,
                    additional_kwargs={_ROLE_KEY: _COMPACTION_REQUEST_ROLE},
                )
            ]
        }

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> Any:
        if not request.messages or not _has_role(
            request.messages[-1], _COMPACTION_REQUEST_ROLE
        ):
            return await handler(request)

        response = await handler(
            request.override(tools=[], tool_choice=None, response_format=None)
        )
        tool_calls = [
            tool_call
            for message in _response_messages(response)
            if isinstance(message, AIMessage)
            for tool_call in message.tool_calls
        ]
        if tool_calls:
            names = ", ".join(str(call.get("name", "unknown")) for call in tool_calls)
            raise RuntimeError(f"压缩阶段禁止工具调用，模型却返回了: {names}")
        return response

    @hook_config(can_jump_to=["model", "end"])
    def after_model(
        self, state: dict[str, Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        del runtime
        messages = state["messages"]
        request_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if _has_role(messages[index], _COMPACTION_REQUEST_ROLE)
            ),
            None,
        )
        if request_index is None:
            return None

        response = next(
            (
                message
                for message in messages[request_index + 1 :]
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if response is None or not response.text.strip():
            raise RuntimeError("压缩阶段没有返回有效文本")

        resume = next(
            (message for message in messages if _has_role(message, _RESUME_ROLE)),
            next((message for message in messages if isinstance(message, HumanMessage)), None),
        )
        task = next(
            (
                message
                for message in reversed(messages[:request_index])
                if _has_role(message, _TASK_ROLE)
            ),
            next(
                (
                    message
                    for message in reversed(messages[:request_index])
                    if isinstance(message, HumanMessage)
                    and message is not resume
                    and _role(message) != _COMPACTED_CONTEXT_ROLE
                ),
                None,
            ),
        )
        if resume is None or task is None:
            raise RuntimeError("压缩阶段无法识别简历消息或本次任务消息")

        compacted = HumanMessage(
            content=f"以下是压缩后的历史上下文：\n\n{response.text.strip()}",
            additional_kwargs={_ROLE_KEY: _COMPACTED_CONTEXT_ROLE},
        )
        elapsed = time.perf_counter() - self._started_at if self._started_at else 0.0
        logger.info(
            "上下文压缩完成: elapsed=%.2fs old_messages=%d new_messages=3 summary_chars=%d",
            elapsed,
            len(messages),
            len(response.text.strip()),
        )
        self._started_at = None
        manual = bool(state.get("manual_compact", False))
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                resume,
                task,
                compacted,
            ],
            "manual_compact": False,
            "jump_to": "end" if manual else "model",
        }
