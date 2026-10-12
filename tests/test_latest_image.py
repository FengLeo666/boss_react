from __future__ import annotations

from pathlib import Path

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages

from boss_react.context_compaction import AgentNodeCompactionMiddleware, _image_count, resume_message, task_message
from boss_react.latest_image import LatestImageMiddleware
from boss_react.shallow_sqlite import AsyncShallowSqliteSaver


def test_keep_last_usable_image_and_preserve_text() -> None:
    messages = [
        ToolMessage(
            content=[{"type": "text", "text": "first page"}, {"type": "image", "base64": "Zmlyc3Q="}],
            tool_call_id="first",
            id="message-1",
        ),
        ToolMessage(
            content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,c2Vjb25k"}}],
            tool_call_id="second",
            id="message-2",
        ),
        ToolMessage(
            content=[{"type": "text", "text": "screenshot unavailable"}, {"type": "image", "base64": ""}],
            tool_call_id="failed",
            id="message-3",
        ),
    ]
    middleware = LatestImageMiddleware()

    update = middleware.before_model({"messages": messages}, None)
    assert update is not None
    result = add_messages(messages, update["messages"])

    assert _image_count(result) == 1
    assert result[0].content == [{"type": "text", "text": "first page"}]
    assert result[1].content == messages[1].content
    assert result[2].content == [{"type": "text", "text": "screenshot unavailable"}]
    assert [message.id for message in result] == [message.id for message in messages]
    assert middleware.before_model({"messages": result}, None) is None


def test_image_only_message_keeps_placeholder() -> None:
    messages = [
        ToolMessage(content=[{"type": "image", "base64": "Zmlyc3Q="}], tool_call_id="first", id="first"),
        ToolMessage(content=[{"type": "image", "base64": "c2Vjb25k"}], tool_call_id="second", id="second"),
    ]

    update = LatestImageMiddleware().before_agent({"messages": messages}, None)
    result = add_messages(messages, update["messages"])

    assert result[0].content == [{"type": "text", "text": "[Earlier image omitted from context]"}]
    assert result[1].content == messages[1].content
    assert _image_count(result) == 1


@pytest.mark.asyncio
async def test_old_checkpoint_images_are_pruned_before_compaction_check() -> None:
    class InspectingModel(FakeMessagesListChatModel):
        def _generate(self, messages, *args, **kwargs):
            assert _image_count(messages) == 1
            return super()._generate(messages, *args, **kwargs)

    model = InspectingModel(responses=[AIMessage(content="done")])
    agent = create_agent(
        model=model,
        middleware=[LatestImageMiddleware(), AgentNodeCompactionMiddleware(trigger_tokens=1_000_000)],
    )
    images = [
        HumanMessage(content=[{"type": "image", "base64": "YQ==", "mime_type": "image/png"}])
        for _ in range(251)
    ]

    result = await agent.ainvoke({"messages": [resume_message("resume"), task_message("task"), *images]})

    assert result["messages"][-1].content == "done"
    assert _image_count(result["messages"]) == 1
    assert len(result["messages"]) == 254


@pytest.mark.asyncio
async def test_tool_images_are_pruned_between_model_calls_and_in_checkpoint(tmp_path: Path) -> None:
    seen_images: list[int] = []
    screenshot_data = iter(["Zmlyc3Q=", "c2Vjb25k"])

    class InspectingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, *args, **kwargs):
            seen_images.append(_image_count(messages))
            return super()._generate(messages, *args, **kwargs)

    @tool
    def screenshot() -> list[dict[str, str]]:
        """Return the current page screenshot."""
        return [{"type": "image", "base64": next(screenshot_data), "mime_type": "image/png"}]

    model = InspectingModel(responses=[
        AIMessage(content="", tool_calls=[{"name": "screenshot", "args": {}, "id": "shot-1"}]),
        AIMessage(content="", tool_calls=[{"name": "screenshot", "args": {}, "id": "shot-2"}]),
        AIMessage(content="done"),
    ])
    config = {"configurable": {"thread_id": "latest-image"}}
    async with AsyncShallowSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        agent = create_agent(
            model=model,
            tools=[screenshot],
            middleware=[LatestImageMiddleware(), AgentNodeCompactionMiddleware(trigger_tokens=1_000_000)],
            checkpointer=saver,
        )
        result = await agent.ainvoke({"messages": [resume_message("resume"), task_message("task")]}, config)
        saved = await saver.aget_tuple(config)

    assert result["messages"][-1].content == "done"
    assert seen_images == [0, 1, 1]
    assert _image_count(saved.checkpoint["channel_values"]["messages"]) == 1
    assert saved.checkpoint["channel_values"]["messages"][-2].content[0]["base64"] == "c2Vjb25k"
