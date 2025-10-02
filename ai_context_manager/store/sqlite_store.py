import sqlite3
from datetime import datetime
from typing import List, Dict
from .base import FeedbackStore

class SQLiteFeedbackStore(FeedbackStore):
    def __init__(self, db_path="feedback.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    component_id TEXT NOT NULL,
                    component_type TEXT NOT NULL,
                    score REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_component_id ON feedback (component_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_component_type ON feedback (component_type)")

    def add_feedback(self, component_id: str, score: float, component_type: str):
        now = datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute("""
                INSERT INTO feedback (component_id, component_type, score, timestamp)
                VALUES (?, ?, ?, ?)
            """, (component_id, component_type, score, now))

    def get_scores(self, component_id: str) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT score, timestamp FROM feedback
            WHERE component_id = ?
        """, (component_id,))
        return [dict(row) for row in cur.fetchall()]

    def get_scores_by_type(self, component_type: str) -> List[Dict]:
        cur = self.conn.cursor()
        cur.execute("""
            SELECT score, timestamp FROM feedback
            WHERE component_type = ?
        """, (component_type,))
        return [dict(row) for row in cur.fetchall()]

    def get_tracked_component_ids(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT component_id FROM feedback")
        return [row["component_id"] for row in cur.fetchall()]
    
    def get_tracked_component_types(self) -> List[str]:
        cur = self.conn.cursor()
        cur.execute("SELECT DISTINCT component_type FROM feedback")
        return [row["component_type"] for row in cur.fetchall()]


    def close(self):
        self.conn.close()
