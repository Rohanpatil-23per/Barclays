"""
IMMUNEX — Layer 3: Database Layer
===================================
Manages an asyncpg connection pool for the PostgreSQL + pgvector backend
used by the RLHF dual-table memory system.

Tables
------
  expert_demonstrations  — stores human-approved / human-overridden state-action
                           pairs for offline RLHF fine-tuning of the DQN.
  rejected_demonstrations — stores DQN-proposed actions that were explicitly
                            rejected by a human admin, used as negative examples
                            for cosine-distance rejection-memory lookups.

Environment variables
---------------------
  DATABASE_URL  — asyncpg DSN, e.g.
                  "postgresql://user:pass@localhost:5432/immunex"
                  If unset or unreachable, the pool is None and all callers
                  degrade gracefully (no-op) rather than crashing.

[IMMUNEX-PATCH] New file — adds asyncpg/pgvector RLHF memory layer.
"""

from __future__ import annotations

import os
import structlog

# [IMMUNEX-PATCH] asyncpg is the async PostgreSQL driver; pgvector DDL is
# issued as raw SQL via the pool so no extra Python library is needed.
try:
    import asyncpg
    _ASYNCPG_AVAILABLE = True
except ImportError:
    _ASYNCPG_AVAILABLE = False

logger = structlog.get_logger("immunex.database")

# [IMMUNEX-PATCH] Module-level pool reference — set by init_pool(), cleared
# by close_pool().  All callers use get_pool() so they always see None when
# the DB is unavailable rather than stale connection objects.
_pool: "asyncpg.Pool | None" = None  # type: ignore[name-defined]

_DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

# ── DDL statements ─────────────────────────────────────────────────────────

# [IMMUNEX-PATCH] Enable pgvector extension (idempotent).
_SQL_ENABLE_VECTOR = "CREATE EXTENSION IF NOT EXISTS vector;"

# [IMMUNEX-PATCH] expert_demonstrations: positive RLHF examples written when
# a human approves or overrides a DQN suggestion.
# state_vector is 128-dimensional matching the DQN observation space.
# expert_actions is INTEGER[] to support the new multi-action architecture.
_SQL_EXPERT_TABLE = """
CREATE TABLE IF NOT EXISTS expert_demonstrations (
    id             BIGSERIAL    PRIMARY KEY,
    alert_id       TEXT         NOT NULL,
    admin_id       TEXT         NOT NULL DEFAULT 'system',
    state_vector   vector(128)  NOT NULL,
    expert_actions INTEGER[]    NOT NULL,
    timestamp      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
"""

# [IMMUNEX-PATCH] rejected_demonstrations: negative RLHF examples written
# when a human admin explicitly rejects a set of DQN-proposed actions.
# rejection_reason is free-text for audit purposes.
_SQL_REJECTED_TABLE = """
CREATE TABLE IF NOT EXISTS rejected_demonstrations (
    id                BIGSERIAL   PRIMARY KEY,
    alert_id          TEXT        NOT NULL,
    admin_id          TEXT        NOT NULL DEFAULT 'system',
    state_vector      vector(128) NOT NULL,
    rejected_actions  INTEGER[]   NOT NULL,
    rejection_reason  TEXT,
    timestamp         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# [IMMUNEX-PATCH] Cosine-distance index on rejected_demonstrations so the
# rejection-memory lookup stays sub-millisecond even with thousands of rows.
_SQL_REJECTED_INDEX = """
CREATE INDEX IF NOT EXISTS idx_rejected_demos_vector
    ON rejected_demonstrations
    USING ivfflat (state_vector vector_cosine_ops)
    WITH (lists = 100);
"""


# ── Pool lifecycle ──────────────────────────────────────────────────────────

async def init_pool() -> None:
    """
    Creates the asyncpg connection pool and initialises the schema.

    Called once from the FastAPI lifespan function on startup.
    Degrades gracefully — if the DB is unreachable, _pool stays None
    and all callers receive None from get_pool().

    [IMMUNEX-PATCH] Graceful degradation: pool=None means rejection-memory
    lookups are skipped and RLHF inserts are no-ops.
    """
    global _pool

    if not _ASYNCPG_AVAILABLE:
        logger.warning(
            "asyncpg_not_installed",
            hint="pip install asyncpg  — RLHF memory will be disabled",
        )
        return

    if not _DATABASE_URL:
        logger.warning(
            "database_url_not_set",
            hint="Set DATABASE_URL env var to enable RLHF memory",
        )
        return

    try:
        # [IMMUNEX-PATCH] min_size=2 keeps warm connections ready; max_size=10
        # is conservative for a single-node banking deployment.
        _pool = await asyncpg.create_pool(
            dsn=_DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=5,  # 5 s per query — fail fast
        )
        logger.info("asyncpg_pool_created", dsn=_DATABASE_URL.split("@")[-1])

        async with _pool.acquire() as conn:
            # [IMMUNEX-PATCH] Execute DDL in order: extension → tables → index
            await conn.execute(_SQL_ENABLE_VECTOR)
            await conn.execute(_SQL_EXPERT_TABLE)
            await conn.execute(_SQL_REJECTED_TABLE)
            await conn.execute(_SQL_REJECTED_INDEX)

        logger.info("database_schema_initialised")

    except Exception as exc:
        logger.error(
            "database_init_failed",
            error=str(exc),
            hint="RLHF memory disabled — pipeline will run without it",
        )
        _pool = None


async def close_pool() -> None:
    """
    Closes the asyncpg pool gracefully on application shutdown.

    [IMMUNEX-PATCH] Called from the FastAPI lifespan cleanup block (after yield).
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("asyncpg_pool_closed")


def get_pool() -> "asyncpg.Pool | None":  # type: ignore[name-defined]
    """
    Returns the active asyncpg pool, or None if the DB is unavailable.
    All callers must handle None to support graceful degradation.

    [IMMUNEX-PATCH] Single-point accessor — avoids module-level global leaking.
    """
    return _pool


# ── RLHF write helpers ──────────────────────────────────────────────────────

async def insert_expert_demonstration(
    alert_id:       str,
    state_vector:   list[float],
    expert_actions: list[int],
    admin_id:       str = "system",
) -> None:
    """
    Inserts a positive RLHF example into expert_demonstrations.

    Called by POST /admin/override (Step E) and POST /approve after
    human confirmation.

    [IMMUNEX-PATCH] Uses asyncpg's native array support for expert_actions
    (INTEGER[]) and pgvector's vector cast for the 128-dim state.
    """
    pool = get_pool()
    if pool is None:
        logger.warning(
            "rlhf_insert_skipped",
            table="expert_demonstrations",
            alert_id=alert_id,
            reason="No database pool",
        )
        return

    try:
        async with pool.acquire() as conn:
            # [IMMUNEX-PATCH] pgvector expects the vector as a string like
            # '[0.1, 0.2, ...]' or a Python list directly via the codec.
            await conn.execute(
                """
                INSERT INTO expert_demonstrations
                    (alert_id, admin_id, state_vector, expert_actions)
                VALUES ($1, $2, $3::vector, $4)
                """,
                alert_id,
                admin_id,
                str(state_vector),       # pgvector text cast
                expert_actions,          # asyncpg native INTEGER[]
            )
        logger.info(
            "expert_demonstration_inserted",
            alert_id=alert_id,
            actions=expert_actions,
        )
    except Exception as exc:
        logger.error(
            "expert_demonstration_insert_failed",
            alert_id=alert_id,
            error=str(exc),
        )


async def insert_rejected_demonstration(
    alert_id:         str,
    state_vector:     list[float],
    rejected_actions: list[int],
    rejection_reason: str = "",
    admin_id:         str = "system",
) -> None:
    """
    Inserts a negative RLHF example into rejected_demonstrations.

    Called by POST /admin/reject.

    [IMMUNEX-PATCH] rejection_reason is stored for downstream audit / RLHF
    labelling — the DQN training loop can use these as hard-negative examples.
    """
    pool = get_pool()
    if pool is None:
        logger.warning(
            "rlhf_insert_skipped",
            table="rejected_demonstrations",
            alert_id=alert_id,
            reason="No database pool",
        )
        return

    try:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO rejected_demonstrations
                    (alert_id, admin_id, state_vector, rejected_actions,
                     rejection_reason)
                VALUES ($1, $2, $3::vector, $4, $5)
                """,
                alert_id,
                admin_id,
                str(state_vector),
                rejected_actions,
                rejection_reason or None,
            )
        logger.info(
            "rejected_demonstration_inserted",
            alert_id=alert_id,
            actions=rejected_actions,
        )
    except Exception as exc:
        logger.error(
            "rejected_demonstration_insert_failed",
            alert_id=alert_id,
            error=str(exc),
        )


# ── Rejection memory lookup ─────────────────────────────────────────────────

async def is_action_rejected(
    state_vector:    list[float],
    candidate_action: int,
    cosine_threshold: float = 0.15,
) -> bool:
    """
    Returns True if a very similar state (cosine distance < threshold) was
    previously rejected alongside this candidate action.

    Used by response_engine.py before finalising the DQN action selection
    to prevent the model from re-proposing previously rejected actions in
    nearly identical situations.

    [IMMUNEX-PATCH] Cosine distance search using pgvector's <=> operator.
    Threshold 0.15 ≈ states that are ~93 % similar — tight enough to avoid
    false positives on genuinely novel threats.
    """
    pool = get_pool()
    if pool is None:
        return False  # [IMMUNEX-PATCH] Graceful degradation — no DB = no filter

    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT 1
                FROM rejected_demonstrations
                WHERE rejected_actions @> ARRAY[$1]::integer[]
                  AND (state_vector <=> $2::vector) < $3
                LIMIT 1
                """,
                candidate_action,
                str(state_vector),
                cosine_threshold,
            )
        if row is not None:
            logger.info(
                "rejection_memory_hit",
                candidate_action=candidate_action,
                threshold=cosine_threshold,
            )
            return True
        return False

    except Exception as exc:
        logger.warning(
            "rejection_memory_lookup_failed",
            error=str(exc),
        )
        return False  # [IMMUNEX-PATCH] Fail open — don't block on DB errors
