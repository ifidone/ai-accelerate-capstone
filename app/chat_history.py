"""Persistent per-user chat history for LabBot."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from . import config

DATABASE_PATH = config.DATA_DIR / "labbot_history.sqlite3"


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize() -> None:
    """Create the history table if it does not exist yet."""
    with _connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id
            ON chat_messages (user_id, id)
            """
        )


def append_message(
    user_id: str,
    role: str,
    content: str,
) -> None:
    """Save one user or assistant message."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")

    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO chat_messages (
                user_id,
                role,
                content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                role,
                content,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_recent_messages(
    user_id: str,
    limit: int = 20,
) -> list[dict]:
    """Return recent messages in chronological order for graph context."""
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {
            "role": row["role"],
            "content": row["content"],
        }
        for row in reversed(rows)
    ]


def get_display_messages(
    user_id: str,
    limit: int = 100,
) -> list[dict]:
    """Return history for rendering in the browser, oldest first."""
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id, role, content, created_at
            FROM chat_messages
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "created_at": row["created_at"],
        }
        for row in reversed(rows)
    ]


def clear_messages(user_id: str) -> None:
    """Delete one authenticated user's history."""
    with _connection() as connection:
        connection.execute(
            """
            DELETE FROM chat_messages
            WHERE user_id = ?
            """,
            (user_id,),
        )