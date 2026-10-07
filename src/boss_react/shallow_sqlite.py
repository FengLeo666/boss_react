"""SQLite checkpointer that retains only the latest checkpoint per thread."""

from __future__ import annotations

import json

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class AsyncShallowSqliteSaver(AsyncSqliteSaver):
    """Drop-in async SQLite saver without checkpoint history or time travel.

    Uses the standard SQLite saver tables, so existing databases remain readable.
    Old checkpoints for a thread are removed on its next write, not at startup.
    """

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        # Like AsyncShallowPostgresSaver, a checkpoint ID is not a time-travel request.
        current_config = {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
            }
        }
        result = await super().aget_tuple(current_config)
        return result._replace(parent_config=None) if result is not None else None

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Atomically replace a thread's checkpoint and prune obsolete writes."""
        await self.setup()
        configurable = config["configurable"]
        thread_id = str(configurable["thread_id"])
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id = configurable.get("checkpoint_id") or ""
        type_, serialized_checkpoint = self.serde.dumps_typed(checkpoint)
        serialized_metadata = json.dumps(
            get_checkpoint_metadata(config, metadata), ensure_ascii=False
        ).encode("utf-8", "ignore")

        async with self.lock:
            try:
                await self.conn.execute("BEGIN IMMEDIATE")
                await self.conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id <> ?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )
                await self.conn.execute(
                    "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? "
                    "AND checkpoint_id NOT IN (?, ?)",
                    (thread_id, checkpoint_ns, checkpoint_id, parent_id),
                )
                await self.conn.execute(
                    "INSERT OR REPLACE INTO checkpoints "
                    "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (thread_id, checkpoint_ns, checkpoint_id, parent_id or None, type_, serialized_checkpoint, serialized_metadata),
                )
                await self.conn.commit()
            except BaseException:
                await self.conn.rollback()
                raise

        return {
            "configurable": {
                "thread_id": configurable["thread_id"],
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }
