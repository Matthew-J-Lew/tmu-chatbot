from __future__ import annotations

import asyncpg
from typing import Optional

from app.api.config import PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

_pool: Optional[asyncpg.Pool] = None


async def init_db_pool() -> None:
    """
    Create a global asyncpg connection pool.

    Why this exists (non-technical explanation):
    - Opening a brand-new database connection for every user question is slow.
    - A "pool" keeps a few connections ready so queries are fast and consistent.
    """
    global _pool
    if _pool is not None:
        return

    _pool = await asyncpg.create_pool(
        host=PGHOST,
        port=PGPORT,
        database=PGDATABASE,
        user=PGUSER,
        password=PGPASSWORD,
        min_size=1,
        max_size=10,
        timeout=10,
    )


async def close_db_pool() -> None:
    """Gracefully close the DB pool when the API shuts down."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the active pool (assumes init_db_pool() has run)."""
    if _pool is None:
        raise RuntimeError("DB pool not initialized. Call init_db_pool() on startup.")
    return _pool
