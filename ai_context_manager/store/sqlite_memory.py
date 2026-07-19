"""SQLite implementation of the memory-store contract."""

import json
import sqlite3
from typing import Dict, List, Optional

from ai_context_manager.store.base import MemoryStore
from ai_context_manager.store.errors import StorageReadError, StorageWriteError


class SQLiteMemoryStore(MemoryStore):
    """Durable component storage using one JSON payload per component.

    SQLite serializes writes transactionally. A store instance owns one connection
    and should not be shared across threads; create one instance per thread.
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        try:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            with self.conn:
                self.conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_components (
                        id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise StorageReadError(f"Cannot open SQLite memory store: {db_path}") from exc

    def load_all(self) -> List[Dict]:
        try:
            rows = self.conn.execute(
                "SELECT payload FROM memory_components ORDER BY rowid"
            ).fetchall()
            return [json.loads(row["payload"]) for row in rows]
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise StorageReadError(f"Cannot read SQLite memory store: {self.db_path}") from exc

    def save_component(self, component: Dict) -> None:
        component_id = component.get("id")
        if not component_id:
            raise ValueError("Stored component must have a non-empty ID")
        try:
            payload = json.dumps(component)
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO memory_components (id, payload) VALUES (?, ?)
                    ON CONFLICT(id) DO UPDATE SET payload = excluded.payload
                    """,
                    (component_id, payload),
                )
        except (sqlite3.Error, TypeError, ValueError) as exc:
            raise StorageWriteError(f"Cannot write SQLite memory store: {self.db_path}") from exc

    def delete_component(self, component_id: str) -> None:
        try:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM memory_components WHERE id = ?", (component_id,)
                )
        except sqlite3.Error as exc:
            raise StorageWriteError(f"Cannot write SQLite memory store: {self.db_path}") from exc

    def get_component(self, component_id: str) -> Optional[Dict]:
        try:
            row = self.conn.execute(
                "SELECT payload FROM memory_components WHERE id = ?", (component_id,)
            ).fetchone()
            return json.loads(row["payload"]) if row else None
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise StorageReadError(f"Cannot read SQLite memory store: {self.db_path}") from exc

    def close(self) -> None:
        self.conn.close()
