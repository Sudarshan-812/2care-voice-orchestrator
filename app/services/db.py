import logging
from datetime import datetime
from typing import Any, Optional

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class DatabaseManager:
    """Manages the asyncpg connection pool, call-session state, and appointment slot locks."""

    def __init__(self) -> None:
        self.pool: Optional[asyncpg.Pool] = None

    async def init_db(self) -> None:
        self.pool = await asyncpg.create_pool(dsn=settings.DATABASE_URL)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS call_sessions (
                    call_id TEXT PRIMARY KEY,
                    from_number TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
                    context_snapshot TEXT
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS appointment_locks (
                    slot_time TIMESTAMPTZ NOT NULL,
                    business_id INTEGER NOT NULL,
                    practitioner_id INTEGER NOT NULL,
                    locked_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (slot_time, business_id, practitioner_id)
                )
                """
            )
        logger.info("Database initialized: call_sessions and appointment_locks ready")

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def upsert_call_state(
        self,
        call_id: str,
        from_number: Optional[str],
        status: str,
        context_snapshot: Optional[str],
    ) -> None:
        assert self.pool is not None
        await self.pool.execute(
            """
            INSERT INTO call_sessions (call_id, from_number, status, last_updated, context_snapshot)
            VALUES ($1, $2, $3, now(), $4)
            ON CONFLICT (call_id) DO UPDATE SET
                from_number = EXCLUDED.from_number,
                status = EXCLUDED.status,
                last_updated = now(),
                context_snapshot = EXCLUDED.context_snapshot
            """,
            call_id,
            from_number,
            status,
            context_snapshot,
        )

    async def get_call_state(self, from_number: str) -> Optional[dict[str, Any]]:
        assert self.pool is not None
        row = await self.pool.fetchrow(
            """
            SELECT call_id, from_number, status, last_updated, context_snapshot
            FROM call_sessions
            WHERE from_number = $1
            ORDER BY last_updated DESC
            LIMIT 1
            """,
            from_number,
        )
        return dict(row) if row is not None else None

    async def acquire_slot_lock(self, slot_time: str, business_id: int, practitioner_id: int) -> bool:
        """Attempt to claim an appointment slot. Returns False if another request already holds it."""
        assert self.pool is not None
        try:
            await self.pool.execute(
                """
                INSERT INTO appointment_locks (slot_time, business_id, practitioner_id)
                VALUES ($1, $2, $3)
                """,
                _parse_iso(slot_time),
                business_id,
                practitioner_id,
            )
        except asyncpg.UniqueViolationError:
            logger.warning(
                "Slot lock contention: slot_time=%s business_id=%s practitioner_id=%s",
                slot_time,
                business_id,
                practitioner_id,
            )
            return False
        return True


db = DatabaseManager()
