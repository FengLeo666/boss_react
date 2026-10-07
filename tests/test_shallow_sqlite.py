from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphDrained
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import RunControl

from boss_react.shallow_sqlite import AsyncShallowSqliteSaver


def _checkpoint(checkpoint_id: str) -> dict:
    return {"id": checkpoint_id, "channel_values": {"value": checkpoint_id}}


@pytest.mark.asyncio
async def test_shallow_saver_keeps_latest_checkpoint_and_required_writes(tmp_path: Path) -> None:
    path = str(tmp_path / "checkpoints.sqlite")
    first = {"configurable": {"thread_id": "one", "checkpoint_ns": ""}}
    other = {"configurable": {"thread_id": "two", "checkpoint_ns": ""}}

    async with AsyncShallowSqliteSaver.from_conn_string(path) as saver:
        first = await saver.aput(first, _checkpoint("001"), {"source": "input", "step": 1}, {})
        await saver.aput_writes(first, [("value", "pending")], "task-1")
        await saver.aput_writes(first, [("value", "pending")], "task-1")
        await saver.aput(other, _checkpoint("other"), {"source": "input", "step": 1}, {})

        second = await saver.aput(first, _checkpoint("002"), {"source": "loop", "step": 2}, {})
        assert (await saver.aget_tuple(first)).checkpoint["id"] == "002"
        assert (await saver.aget_tuple(second)).parent_config is None
        assert len([item async for item in saver.alist({"configurable": {"thread_id": "one"}})]) == 1

        await saver.aput_writes(second, [("value", "latest")], "task-2")
        third = await saver.aput(second, _checkpoint("003"), {"source": "loop", "step": 3}, {})
        assert (await saver.aget_tuple(third)).checkpoint["id"] == "003"

        async with saver.conn.execute(
            "SELECT checkpoint_id FROM writes WHERE thread_id = ? ORDER BY checkpoint_id", ("one",)
        ) as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["002"]
        async with saver.conn.execute(
            "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?", ("two",)
        ) as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["other"]


@pytest.mark.asyncio
async def test_shallow_saver_lazily_prunes_existing_history(tmp_path: Path) -> None:
    path = str(tmp_path / "checkpoints.sqlite")
    config = {"configurable": {"thread_id": "existing", "checkpoint_ns": ""}}
    async with AsyncSqliteSaver.from_conn_string(path) as saver:
        first = await saver.aput(config, _checkpoint("001"), {"source": "input", "step": 1}, {})
        await saver.aput(first, _checkpoint("002"), {"source": "loop", "step": 2}, {})

    async with AsyncShallowSqliteSaver.from_conn_string(path) as saver:
        current = await saver.aget_tuple(config)
        assert current.checkpoint["id"] == "002"
        await saver.aput(current.config, _checkpoint("003"), {"source": "loop", "step": 3}, {})
        async with saver.conn.execute("SELECT checkpoint_id FROM checkpoints") as cursor:
            assert [row[0] for row in await cursor.fetchall()] == ["003"]


@pytest.mark.asyncio
async def test_graph_can_resume_after_drain_with_shallow_saver(tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first(_state: dict) -> dict:
        entered.set()
        await release.wait()
        return {"value": 1}

    async def second(_state: dict) -> dict:
        return {"value": 2}

    builder = StateGraph(dict)
    builder.add_node("first", first)
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    config = {"configurable": {"thread_id": "resume"}}
    path = str(tmp_path / "graph.sqlite")

    async with AsyncShallowSqliteSaver.from_conn_string(path) as saver:
        graph = builder.compile(checkpointer=saver)
        control = RunControl()
        run_task = asyncio.create_task(graph.ainvoke({"value": 0}, config, control=control))
        await entered.wait()
        control.request_drain("user_escape")
        release.set()
        with pytest.raises(GraphDrained):
            await run_task
        assert (await graph.aget_state(config)).next == ("second",)

    async with AsyncShallowSqliteSaver.from_conn_string(path) as saver:
        graph = builder.compile(checkpointer=saver)
        assert (await graph.ainvoke(None, config))["value"] == 2
        assert len([item async for item in saver.alist(config)]) == 1
