from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from matgpt.message import Message, Role


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    timestamp TEXT NOT NULL,
    position INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def save_conversation(self, conv_id: str, name: str, messages: list[Message]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (id, name, updated_at) VALUES (?, ?, ?)",
                (conv_id, name, now),
            )
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.executemany(
                "INSERT INTO messages (conversation_id, role, content, token_count, timestamp, position) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        conv_id,
                        m.role.value,
                        m.content,
                        m.token_count,
                        m.timestamp.isoformat(),
                        i,
                    )
                    for i, m in enumerate(messages)
                ],
            )

    def load_conversation(self, conv_id: str) -> tuple[str, list[Message]] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT name FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if row is None:
                return None
            name: str = row["name"]
            msg_rows = conn.execute(
                "SELECT role, content, token_count, timestamp FROM messages "
                "WHERE conversation_id = ? ORDER BY position",
                (conv_id,),
            ).fetchall()
            messages = [
                Message(
                    role=Role(r["role"]),
                    content=r["content"],
                    token_count=r["token_count"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                )
                for r in msg_rows
            ]
        return name, messages

    def list_conversations(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.name, c.updated_at, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN messages m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_conversation(self, conv_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    # ── RAG / Embeddings ──────────────────────────────────────────────────────

    def save_embedding(self, conv_id: str, content: str, embedding: bytes) -> None:
        """Store a serialized embedding for a piece of content in a conversation."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO embeddings (conversation_id, content, embedding, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conv_id, content, embedding, now),
            )

    def search_similar(
        self,
        query_embedding: list[float],
        top_k: int = 4,
        exclude_conv_id: str | None = None,
    ) -> list[dict]:
        """Return up to top_k embedding rows sorted by cosine similarity.

        Excludes the current conversation so the model only sees *other*
        sessions' memories. Returns dicts with keys: content, similarity.
        """
        from matgpt.embeddings import cosine_similarity, deserialize

        with self._connect() as conn:
            if exclude_conv_id:
                rows = conn.execute(
                    "SELECT content, embedding FROM embeddings WHERE conversation_id != ?",
                    (exclude_conv_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT content, embedding FROM embeddings"
                ).fetchall()

        scored: list[tuple[float, str]] = []
        for row in rows:
            vec = deserialize(row["embedding"])
            sim = cosine_similarity(query_embedding, vec)
            scored.append((sim, row["content"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"content": content, "similarity": sim}
            for sim, content in scored[:top_k]
        ]

    def export_json(self, conv_id: str) -> str:
        result = self.load_conversation(conv_id)
        if result is None:
            raise ValueError(f"Conversation '{conv_id}' not found.")
        name, messages = result
        data = {
            "id": conv_id,
            "name": name,
            "messages": [
                {
                    "role": m.role.value,
                    "content": m.content,
                    "timestamp": m.timestamp.isoformat(),
                    "token_count": m.token_count,
                }
                for m in messages
            ],
        }
        return json.dumps(data, indent=2)
