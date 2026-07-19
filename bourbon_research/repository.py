import json
from pathlib import Path
import re
import sqlite3
from typing import Dict, List, Optional

from bourbon_research.models import ExtractedClaim, FetchedSource, Project


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    slug TEXT NOT NULL UNIQUE,
    subject TEXT NOT NULL,
    objective TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    UNIQUE(project_id, text)
);
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    publisher TEXT,
    source_class TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    content_hash TEXT NOT NULL,
    snapshot_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'read',
    UNIQUE(project_id, url)
);
CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    quotation TEXT,
    memory_id TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, source_id, text)
);
CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    left_claim_id INTEGER NOT NULL REFERENCES claims(id),
    right_claim_id INTEGER NOT NULL REFERENCES claims(id),
    status TEXT NOT NULL DEFAULT 'open',
    resolution TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, left_claim_id, right_claim_id)
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    command TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at TEXT,
    summary TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "research-project"


class ResearchRepository:
    def __init__(self, workspace: str = ".bourbon-research"):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.workspace / "research.db"))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)

    def create_project(self, subject: str, objective: Optional[str] = None) -> Project:
        base = slugify(subject)
        slug, suffix = base, 2
        while self.conn.execute("SELECT 1 FROM projects WHERE slug = ?", (slug,)).fetchone():
            slug, suffix = f"{base}-{suffix}", suffix + 1
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO projects (slug, subject, objective) VALUES (?, ?, ?)",
                (slug, subject, objective or f"Research {subject} using traceable evidence."),
            )
        project = self.get_project(slug)
        (self.workspace / "projects" / slug / "sources").mkdir(parents=True, exist_ok=True)
        self.set_current_project(slug)
        return project

    def get_project(self, slug: Optional[str] = None) -> Project:
        selected = slug or self.current_project_slug()
        if not selected:
            raise ValueError("No project selected; create one or pass --project")
        row = self.conn.execute("SELECT * FROM projects WHERE slug = ?", (selected,)).fetchone()
        if not row:
            raise ValueError(f"Unknown project: {selected}")
        return Project(row["id"], row["slug"], row["subject"], row["objective"], row["created_at"])

    def list_projects(self) -> List[Dict]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM projects ORDER BY id")]

    def set_current_project(self, slug: str) -> None:
        (self.workspace / "current-project").write_text(slug)

    def current_project_slug(self) -> Optional[str]:
        path = self.workspace / "current-project"
        return path.read_text().strip() if path.exists() else None

    def replace_questions(self, project_id: int, questions: List[str]) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM questions WHERE project_id = ?", (project_id,))
            self.conn.executemany(
                "INSERT INTO questions (project_id, text) VALUES (?, ?)",
                [(project_id, question) for question in questions],
            )

    def questions(self, project_id: int) -> List[Dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM questions WHERE project_id = ? ORDER BY id", (project_id,)
        )]

    def save_source(self, project: Project, source: FetchedSource, content_hash: str) -> int:
        relative = Path("projects") / project.slug / "sources" / f"{content_hash}.txt"
        (self.workspace / relative).write_text(source.text)
        with self.conn:
            self.conn.execute(
                """INSERT INTO sources
                   (project_id, url, title, publisher, source_class, published_at,
                    content_hash, snapshot_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(project_id, url) DO UPDATE SET
                    title=excluded.title, publisher=excluded.publisher,
                    source_class=excluded.source_class, published_at=excluded.published_at,
                    content_hash=excluded.content_hash, snapshot_path=excluded.snapshot_path,
                    fetched_at=CURRENT_TIMESTAMP""",
                (project.id, source.url, source.title, source.publisher,
                 source.source_class, source.published_at, content_hash, str(relative)),
            )
        return self.conn.execute(
            "SELECT id FROM sources WHERE project_id = ? AND url = ?", (project.id, source.url)
        ).fetchone()["id"]

    def source_exists(self, project_id: int, url: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM sources WHERE project_id = ? AND url = ?", (project_id, url)
        ).fetchone() is not None

    def sources_without_claims(self, project_id: int) -> List[Dict]:
        return [dict(row) for row in self.conn.execute(
            """SELECT sources.* FROM sources
               WHERE sources.project_id = ?
                 AND sources.status IN ('read', 'extraction_failed')
                 AND NOT EXISTS (
                     SELECT 1 FROM claims WHERE claims.source_id = sources.id
                 )
               ORDER BY sources.id""",
            (project_id,),
        )]

    def load_source_snapshot(self, row: Dict) -> FetchedSource:
        return FetchedSource(
            row["url"], row["title"],
            (self.workspace / row["snapshot_path"]).read_text(),
            row["publisher"], row["source_class"], row["published_at"],
        )

    def set_source_status(self, source_id: int, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE sources SET status = ? WHERE id = ?", (status, source_id))

    def save_claim(self, project_id: int, source_id: int, claim: ExtractedClaim) -> int:
        with self.conn:
            self.conn.execute(
                """INSERT OR IGNORE INTO claims
                   (project_id, source_id, text, evidence_type, confidence, quotation)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (project_id, source_id, claim.text, claim.evidence_type,
                 claim.confidence, claim.quotation),
            )
        return self.conn.execute(
            "SELECT id FROM claims WHERE project_id = ? AND source_id = ? AND text = ?",
            (project_id, source_id, claim.text),
        ).fetchone()["id"]

    def set_claim_memory(self, claim_id: int, memory_id: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE claims SET memory_id = ? WHERE id = ?", (memory_id, claim_id))

    def sources(self, project_id: int) -> List[Dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM sources WHERE project_id = ? ORDER BY id", (project_id,)
        )]

    def claims(self, project_id: int) -> List[Dict]:
        return [dict(row) for row in self.conn.execute(
            """SELECT claims.*, sources.url, sources.title AS source_title
               FROM claims JOIN sources ON sources.id = claims.source_id
               WHERE claims.project_id = ? ORDER BY claims.id""", (project_id,)
        )]

    def add_contradiction(self, project_id: int, left_id: int, right_id: int) -> None:
        left, right = sorted((left_id, right_id))
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO contradictions (project_id, left_claim_id, right_claim_id) VALUES (?, ?, ?)",
                (project_id, left, right),
            )

    def contradictions(self, project_id: int) -> List[Dict]:
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM contradictions WHERE project_id = ? ORDER BY id", (project_id,)
        )]

    def start_session(self, project_id: int, command: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO sessions (project_id, command) VALUES (?, ?)", (project_id, command)
            )
        return cursor.lastrowid

    def event(self, session_id: int, kind: str, entity_type: str, entity_id, details="") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO events (session_id, kind, entity_type, entity_id, details) VALUES (?, ?, ?, ?, ?)",
                (session_id, kind, entity_type, str(entity_id), details),
            )

    def finish_session(self, session_id: int, summary: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET finished_at=CURRENT_TIMESTAMP, summary=? WHERE id=?",
                (summary, session_id),
            )

    def session_changes(self, project_id: int, session_id: Optional[int] = None) -> List[Dict]:
        if session_id is None:
            row = self.conn.execute(
                "SELECT id FROM sessions WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,)
            ).fetchone()
            if not row:
                return []
            session_id = row["id"]
        return [dict(row) for row in self.conn.execute(
            "SELECT * FROM events WHERE session_id=? ORDER BY id", (session_id,)
        )]

    def close(self):
        self.conn.close()
