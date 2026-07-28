"""Append-only SQLite evidence repository for schema-v2 experiment records.

The global ``evidence_log.sequence`` is the authoritative ordering. Typed tables are
bounded indexes over canonical evidence payloads; ``model_outputs`` projects only the
TaskOutcome identity/reference, status, score, token-count, and latency fields. It does
not duplicate response text or artifact references.

Artifact references are opaque identifiers. This repository stores and returns them as
record fields, but never dereferences, resolves, prints, or logs them. The repository
provides integrity checks, not encryption or redaction. Until a data-retention policy
exists, databases should contain only approved synthetic/non-sensitive Stage 0
artifacts.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Type

from .schema import (
    SCHEMA_VERSION,
    ExperimentResult,
    FeedbackEvent,
    RecordMixin,
    RunManifest,
    SelectionDecision,
    TaskOutcome,
    UtilityEstimate,
)


class RepositoryError(Exception):
    """Base error for repository operations."""


class DuplicateRecordError(RepositoryError, ValueError):
    """Raised when an immutable record identity already exists."""


class ReferenceIntegrityError(RepositoryError, ValueError):
    """Raised when a record refers to missing or inconsistent evidence."""


class RecordNotFoundError(RepositoryError, KeyError):
    """Raised when requested evidence does not exist."""


class IntegrityError(RepositoryError):
    """Raised when persisted evidence or a typed projection is inconsistent."""


@dataclass(frozen=True)
class EvidenceEntry:
    """One raw, canonically serialized entry in global insertion order."""

    sequence: int
    record_type: str
    record_id: str
    schema_version: str
    payload_json: str
    payload_hash: str
    inserted_timestamp: str


@dataclass(frozen=True)
class IntegrityReport:
    """Successful result of a complete repository integrity scan."""

    evidence_rows: int
    per_type_counts: Mapping[str, int]
    ok: bool = True


@dataclass(frozen=True)
class _RecordSpec:
    record_type: str
    record_class: Type[RecordMixin]
    id_attribute: str
    table: str
    index_attributes: Tuple[str, ...]


_SPECS: Tuple[_RecordSpec, ...] = (
    _RecordSpec("run_manifest", RunManifest, "run_id", "experiment_runs", ("run_id",)),
    _RecordSpec(
        "selection_decision",
        SelectionDecision,
        "decision_id",
        "selection_decisions",
        ("decision_id", "run_id", "task_case_id"),
    ),
    _RecordSpec(
        "task_outcome",
        TaskOutcome,
        "outcome_id",
        "outcomes",
        (
            "outcome_id",
            "run_id",
            "task_case_id",
            "selection_decision_id",
            "execution_status",
        ),
    ),
    _RecordSpec(
        "feedback_event",
        FeedbackEvent,
        "event_id",
        "feedback_events",
        ("event_id", "run_id", "task_case_id", "task_family_id"),
    ),
    _RecordSpec(
        "utility_estimate",
        UtilityEstimate,
        "utility_estimate_id",
        "utility_estimates",
        ("utility_estimate_id", "task_family_id", "estimator_version"),
    ),
    _RecordSpec(
        "experiment_result",
        ExperimentResult,
        "experiment_result_id",
        "experiment_results",
        ("experiment_result_id", "run_id"),
    ),
)
_SPEC_BY_TYPE = {spec.record_type: spec for spec in _SPECS}
_SPEC_BY_CLASS = {spec.record_class: spec for spec in _SPECS}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence_log (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    record_type TEXT NOT NULL,
    record_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    inserted_timestamp TEXT NOT NULL,
    UNIQUE(record_type, record_id)
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);

CREATE TABLE IF NOT EXISTS selection_decisions (
    decision_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    task_case_id TEXT NOT NULL,
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);

CREATE TABLE IF NOT EXISTS outcomes (
    outcome_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    task_case_id TEXT NOT NULL,
    selection_decision_id TEXT NOT NULL REFERENCES selection_decisions(decision_id),
    execution_status TEXT NOT NULL,
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);

CREATE TABLE IF NOT EXISTS model_outputs (
    outcome_id TEXT PRIMARY KEY REFERENCES outcomes(outcome_id),
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    task_case_id TEXT NOT NULL,
    selection_decision_id TEXT NOT NULL REFERENCES selection_decisions(decision_id),
    execution_status TEXT NOT NULL,
    normalized_score REAL,
    model_input_tokens INTEGER NOT NULL,
    model_output_tokens INTEGER NOT NULL,
    execution_latency_ms REAL NOT NULL,
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);

CREATE TABLE IF NOT EXISTS feedback_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    task_case_id TEXT NOT NULL,
    task_family_id TEXT NOT NULL,
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);

CREATE TABLE IF NOT EXISTS utility_estimates (
    utility_estimate_id TEXT PRIMARY KEY,
    task_family_id TEXT NOT NULL,
    estimator_version TEXT NOT NULL,
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);

CREATE TABLE IF NOT EXISTS experiment_results (
    experiment_result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
    evidence_sequence INTEGER NOT NULL UNIQUE REFERENCES evidence_log(sequence)
);
"""

_APPEND_ONLY_TABLES = (
    "evidence_log",
    "experiment_runs",
    "selection_decisions",
    "outcomes",
    "model_outputs",
    "feedback_events",
    "utility_estimates",
    "experiment_results",
)


class ExperimentRepository:
    """SQLite-backed, append-only repository for experiment evidence.

    Args:
        database: ``:memory:`` or a filesystem path.
        clock: UTC insertion clock. It may return an aware ``datetime`` or a canonical
            UTC RFC 3339 string ending in ``Z``; injection makes tests deterministic.
        busy_timeout_ms: SQLite lock wait before a concurrent writer fails.
    """

    def __init__(
        self,
        database: Any = ":memory:",
        *,
        clock: Optional[Callable[[], Any]] = None,
        busy_timeout_ms: int = 5000,
    ) -> None:
        if not isinstance(busy_timeout_ms, int) or busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be a nonnegative integer")
        self._database = (
            str(database) if isinstance(database, (str, Path)) else database
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._closed = False
        self._connection = sqlite3.connect(
            self._database,
            isolation_level=None,
            timeout=busy_timeout_ms / 1000,
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("PRAGMA busy_timeout = ?", (busy_timeout_ms,))
        except sqlite3.OperationalError:
            # PRAGMA assignments do not accept bound values on older SQLite releases.
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms:d}")
        if self._database != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA wal_autocheckpoint = 1000")
        self._connection.executescript(_SCHEMA_SQL)
        for table in _APPEND_ONLY_TABLES:
            self._create_append_only_triggers(table)

    def _create_append_only_triggers(self, table: str) -> None:
        self._connection.executescript(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table}_no_update
            BEFORE UPDATE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'append-only table: {table}');
            END;
            CREATE TRIGGER IF NOT EXISTS {table}_no_delete
            BEFORE DELETE ON {table}
            BEGIN
                SELECT RAISE(ABORT, 'append-only table: {table}');
            END;
            """
        )

    def __enter__(self) -> "ExperimentRepository":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the SQLite connection. Closing more than once is harmless."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("experiment repository is closed")

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._ensure_open()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    @staticmethod
    def _canonical_json(record: RecordMixin) -> str:
        return json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    @staticmethod
    def _hash(payload_json: str) -> str:
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def _insertion_timestamp(self) -> str:
        value = self._clock()
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("repository clock must return an aware UTC datetime")
            value = value.astimezone(timezone.utc)
            timespec = "microseconds" if value.microsecond else "seconds"
            return value.isoformat(timespec=timespec).replace("+00:00", "Z")
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("repository clock must return canonical UTC RFC 3339")
            if not value.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(
                None
            ):
                raise ValueError("repository clock must return canonical UTC RFC 3339")
            canonical = (
                parsed.astimezone(timezone.utc)
                .isoformat(timespec="microseconds" if parsed.microsecond else "seconds")
                .replace("+00:00", "Z")
            )
            if canonical != value:
                raise ValueError("repository clock must return canonical UTC RFC 3339")
            return value
        raise ValueError(
            "repository clock must return datetime or canonical UTC string"
        )

    def _append_evidence(self, spec: _RecordSpec, record: RecordMixin) -> int:
        record_id = getattr(record, spec.id_attribute)
        payload_json = self._canonical_json(record)
        try:
            cursor = self._connection.execute(
                "INSERT INTO evidence_log(record_type, record_id, schema_version, "
                "payload_json, payload_hash, inserted_timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    spec.record_type,
                    record_id,
                    SCHEMA_VERSION,
                    payload_json,
                    self._hash(payload_json),
                    self._insertion_timestamp(),
                ),
            )
        except sqlite3.IntegrityError as error:
            if self._connection.execute(
                "SELECT 1 FROM evidence_log WHERE record_type = ? AND record_id = ?",
                (spec.record_type, record_id),
            ).fetchone():
                raise DuplicateRecordError(
                    f"duplicate {spec.record_type} record ID: {record_id}"
                ) from error
            raise
        return int(cursor.lastrowid)

    def append(self, record: RecordMixin) -> int:
        """Dispatch one supported typed record to its atomic append method."""
        method_by_class = {
            RunManifest: self.append_run,
            SelectionDecision: self.append_selection,
            TaskOutcome: self.append_outcome,
            FeedbackEvent: self.append_feedback,
            UtilityEstimate: self.append_utility_estimate,
            ExperimentResult: self.append_experiment_result,
        }
        method = method_by_class.get(type(record))
        if method is None:
            raise ValueError(
                f"unsupported repository record type: {type(record).__name__}"
            )
        return method(record)  # type: ignore[arg-type]

    def append_run(self, record: RunManifest) -> int:
        self._require_type(record, RunManifest)
        with self._transaction():
            sequence = self._append_evidence(_SPEC_BY_TYPE["run_manifest"], record)
            self._insert_typed("experiment_runs", ("run_id",), record, sequence)
        return sequence

    def append_selection(self, record: SelectionDecision) -> int:
        self._require_type(record, SelectionDecision)
        with self._transaction():
            self._require_row("experiment_runs", "run_id", record.run_id, "run")
            sequence = self._append_evidence(
                _SPEC_BY_TYPE["selection_decision"], record
            )
            self._insert_typed(
                "selection_decisions",
                ("decision_id", "run_id", "task_case_id"),
                record,
                sequence,
            )
        return sequence

    def append_outcome(self, record: TaskOutcome) -> int:
        """Atomically append outcome evidence, its typed index, and bounded output projection."""
        self._require_type(record, TaskOutcome)
        with self._transaction():
            self._require_row("experiment_runs", "run_id", record.run_id, "run")
            decision = self._require_row(
                "selection_decisions",
                "decision_id",
                record.selection_decision_id,
                "selection",
            )
            if decision["run_id"] != record.run_id:
                raise ReferenceIntegrityError(
                    f"selection {record.selection_decision_id} does not belong to run {record.run_id}"
                )
            if decision["task_case_id"] != record.task_case_id:
                raise ReferenceIntegrityError(
                    "outcome task_case_id does not match referenced selection"
                )
            sequence = self._append_evidence(_SPEC_BY_TYPE["task_outcome"], record)
            self._insert_typed(
                "outcomes",
                (
                    "outcome_id",
                    "run_id",
                    "task_case_id",
                    "selection_decision_id",
                    "execution_status",
                ),
                record,
                sequence,
            )
            projection_columns = (
                "outcome_id",
                "run_id",
                "task_case_id",
                "selection_decision_id",
                "execution_status",
                "normalized_score",
                "model_input_tokens",
                "model_output_tokens",
                "execution_latency_ms",
            )
            self._insert_typed("model_outputs", projection_columns, record, sequence)
        return sequence

    def append_feedback(self, record: FeedbackEvent) -> int:
        self._require_type(record, FeedbackEvent)
        with self._transaction():
            self._require_row("experiment_runs", "run_id", record.run_id, "run")
            sequence = self._append_evidence(_SPEC_BY_TYPE["feedback_event"], record)
            self._insert_typed(
                "feedback_events",
                ("event_id", "run_id", "task_case_id", "task_family_id"),
                record,
                sequence,
            )
        return sequence

    def append_utility_estimate(self, record: UtilityEstimate) -> int:
        """Append an estimate; estimator_version is the learning-policy version."""
        self._require_type(record, UtilityEstimate)
        with self._transaction():
            for event_id in record.source_event_ids:
                event = self._require_row(
                    "feedback_events", "event_id", event_id, "feedback event"
                )
                if event["task_family_id"] != record.task_family_id:
                    raise ReferenceIntegrityError(
                        f"feedback event {event_id} belongs to a different task family"
                    )
            sequence = self._append_evidence(_SPEC_BY_TYPE["utility_estimate"], record)
            self._insert_typed(
                "utility_estimates",
                ("utility_estimate_id", "task_family_id", "estimator_version"),
                record,
                sequence,
            )
        return sequence

    def append_experiment_result(self, record: ExperimentResult) -> int:
        self._require_type(record, ExperimentResult)
        with self._transaction():
            self._validate_result_references(record)
            sequence = self._append_evidence(_SPEC_BY_TYPE["experiment_result"], record)
            self._insert_typed(
                "experiment_results",
                ("experiment_result_id", "run_id"),
                record,
                sequence,
            )
        return sequence

    @staticmethod
    def _require_type(record: Any, expected: Type[RecordMixin]) -> None:
        if not isinstance(record, expected):
            raise ValueError(f"record must be a {expected.__name__}")

    def _insert_typed(
        self,
        table: str,
        attributes: Tuple[str, ...],
        record: RecordMixin,
        sequence: int,
    ) -> None:
        columns = attributes + ("evidence_sequence",)
        values = tuple(getattr(record, attribute) for attribute in attributes) + (
            sequence,
        )
        placeholders = ", ".join("?" for _ in columns)
        try:
            self._connection.execute(
                f"INSERT INTO {table}({', '.join(columns)}) VALUES ({placeholders})",
                values,
            )
        except sqlite3.IntegrityError as error:
            identity = values[0]
            if self._connection.execute(
                f"SELECT 1 FROM {table} WHERE {attributes[0]} = ?", (identity,)
            ).fetchone():
                raise DuplicateRecordError(
                    f"duplicate {table} record ID: {identity}"
                ) from error
            raise

    def _require_row(
        self, table: str, id_column: str, record_id: str, description: str
    ) -> sqlite3.Row:
        row = self._connection.execute(
            f"SELECT * FROM {table} WHERE {id_column} = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise ReferenceIntegrityError(f"missing {description}: {record_id}")
        return row

    def _validate_result_references(self, record: ExperimentResult) -> None:
        self._require_row("experiment_runs", "run_id", record.run_id, "run")
        groups = (
            (record.outcome_ids, "outcomes", "outcome_id", "outcome"),
            (
                record.selection_decision_ids,
                "selection_decisions",
                "decision_id",
                "selection",
            ),
            (
                record.feedback_event_ids,
                "feedback_events",
                "event_id",
                "feedback event",
            ),
        )
        for identifiers, table, id_column, description in groups:
            for identifier in identifiers:
                row = self._require_row(table, id_column, identifier, description)
                if row["run_id"] != record.run_id:
                    raise ReferenceIntegrityError(
                        f"{description} {identifier} does not belong to {record.run_id}"
                    )
        for identifier in record.utility_estimate_ids:
            self._require_row(
                "utility_estimates",
                "utility_estimate_id",
                identifier,
                "utility estimate",
            )

    def load_run(self, record_id: str) -> RunManifest:
        return self._load("run_manifest", record_id)  # type: ignore[return-value]

    def load_selection(self, record_id: str) -> SelectionDecision:
        return self._load("selection_decision", record_id)  # type: ignore[return-value]

    def load_outcome(self, record_id: str) -> TaskOutcome:
        return self._load("task_outcome", record_id)  # type: ignore[return-value]

    def load_feedback(self, record_id: str) -> FeedbackEvent:
        return self._load("feedback_event", record_id)  # type: ignore[return-value]

    def load_utility_estimate(self, record_id: str) -> UtilityEstimate:
        return self._load("utility_estimate", record_id)  # type: ignore[return-value]

    def load_experiment_result(self, record_id: str) -> ExperimentResult:
        return self._load("experiment_result", record_id)  # type: ignore[return-value]

    def _load(self, record_type: str, record_id: str) -> RecordMixin:
        self._ensure_open()
        row = self._connection.execute(
            "SELECT * FROM evidence_log WHERE record_type = ? AND record_id = ?",
            (record_type, record_id),
        ).fetchone()
        if row is None:
            raise RecordNotFoundError(f"missing {record_type}: {record_id}")
        record = self._decode_evidence(row)
        self._verify_typed_projection(row, record)
        return record

    def list_runs(self) -> List[RunManifest]:
        return self._list("run_manifest")  # type: ignore[return-value]

    def list_selections(self) -> List[SelectionDecision]:
        return self._list("selection_decision")  # type: ignore[return-value]

    def list_outcomes(self) -> List[TaskOutcome]:
        return self._list("task_outcome")  # type: ignore[return-value]

    def list_feedback(self) -> List[FeedbackEvent]:
        return self._list("feedback_event")  # type: ignore[return-value]

    def list_utility_estimates(self) -> List[UtilityEstimate]:
        return self._list("utility_estimate")  # type: ignore[return-value]

    def list_experiment_results(self) -> List[ExperimentResult]:
        return self._list("experiment_result")  # type: ignore[return-value]

    def _list(self, record_type: str) -> List[RecordMixin]:
        self._ensure_open()
        rows = self._connection.execute(
            "SELECT * FROM evidence_log WHERE record_type = ? ORDER BY sequence",
            (record_type,),
        ).fetchall()
        records = []
        for row in rows:
            record = self._decode_evidence(row)
            self._verify_typed_projection(row, record)
            records.append(record)
        return records

    def list_evidence(
        self, *, record_type: Optional[str] = None
    ) -> List[EvidenceEntry]:
        self._ensure_open()
        if record_type is not None and record_type not in _SPEC_BY_TYPE:
            raise ValueError(f"unsupported record_type: {record_type}")
        query = "SELECT * FROM evidence_log"
        parameters: Tuple[Any, ...] = ()
        if record_type is not None:
            query += " WHERE record_type = ?"
            parameters = (record_type,)
        query += " ORDER BY sequence"
        return [self._entry(row) for row in self._connection.execute(query, parameters)]

    def iter_evidence(
        self, *, record_type: Optional[str] = None
    ) -> Iterator[EvidenceEntry]:
        """Iterate a stable snapshot of evidence in authoritative insertion order."""
        return iter(self.list_evidence(record_type=record_type))

    def load_utility_source_events(
        self, utility_estimate_id: str
    ) -> List[FeedbackEvent]:
        """Reconstruct an estimate's ordered raw feedback evidence."""
        estimate = self.load_utility_estimate(utility_estimate_id)
        return [self.load_feedback(event_id) for event_id in estimate.source_event_ids]

    @staticmethod
    def _entry(row: sqlite3.Row) -> EvidenceEntry:
        return EvidenceEntry(
            sequence=row["sequence"],
            record_type=row["record_type"],
            record_id=row["record_id"],
            schema_version=row["schema_version"],
            payload_json=row["payload_json"],
            payload_hash=row["payload_hash"],
            inserted_timestamp=row["inserted_timestamp"],
        )

    def _decode_evidence(self, row: sqlite3.Row) -> RecordMixin:
        actual_hash = self._hash(row["payload_json"])
        if actual_hash != row["payload_hash"]:
            raise IntegrityError(
                f"payload hash mismatch at evidence sequence {row['sequence']}"
            )
        if row["schema_version"] != SCHEMA_VERSION:
            raise IntegrityError(
                f"unsupported schema_version: {row['schema_version']} at evidence sequence "
                f"{row['sequence']}"
            )
        spec = _SPEC_BY_TYPE.get(row["record_type"])
        if spec is None:
            raise IntegrityError(
                f"unknown record_type {row['record_type']} at evidence sequence {row['sequence']}"
            )
        try:
            payload = json.loads(row["payload_json"])
        except (json.JSONDecodeError, TypeError) as error:
            raise IntegrityError(
                f"invalid JSON at evidence sequence {row['sequence']}"
            ) from error
        if not isinstance(payload, dict):
            raise IntegrityError(
                f"payload is not an object at evidence sequence {row['sequence']}"
            )
        if payload.get("schema_version") != row["schema_version"]:
            raise IntegrityError(
                f"schema version mismatch at evidence sequence {row['sequence']}"
            )
        try:
            record = spec.record_class.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise IntegrityError(
                f"invalid {spec.record_type} payload at evidence sequence {row['sequence']}: {error}"
            ) from error
        if getattr(record, spec.id_attribute) != row["record_id"]:
            raise IntegrityError(
                f"record ID mismatch at evidence sequence {row['sequence']}"
            )
        if self._canonical_json(record) != row["payload_json"]:
            raise IntegrityError(
                f"noncanonical payload JSON at evidence sequence {row['sequence']}"
            )
        return record

    def _verify_typed_projection(
        self, evidence: sqlite3.Row, record: RecordMixin
    ) -> None:
        spec = _SPEC_BY_TYPE[evidence["record_type"]]
        identity = getattr(record, spec.id_attribute)
        row = self._connection.execute(
            f"SELECT * FROM {spec.table} WHERE {spec.id_attribute} = ?", (identity,)
        ).fetchone()
        if row is None:
            raise IntegrityError(
                f"missing {spec.table} typed row for evidence sequence {evidence['sequence']}"
            )
        expected = {
            attribute: getattr(record, attribute) for attribute in spec.index_attributes
        }
        expected["evidence_sequence"] = evidence["sequence"]
        if any(row[key] != value for key, value in expected.items()):
            raise IntegrityError(
                f"{spec.table} projection mismatch for evidence sequence {evidence['sequence']}"
            )
        if isinstance(record, TaskOutcome):
            self._verify_model_output(evidence, record)

    def _verify_model_output(self, evidence: sqlite3.Row, record: TaskOutcome) -> None:
        row = self._connection.execute(
            "SELECT * FROM model_outputs WHERE outcome_id = ?", (record.outcome_id,)
        ).fetchone()
        if row is None:
            raise IntegrityError(
                f"missing model_outputs projection for evidence sequence {evidence['sequence']}"
            )
        attributes = (
            "outcome_id",
            "run_id",
            "task_case_id",
            "selection_decision_id",
            "execution_status",
            "normalized_score",
            "model_input_tokens",
            "model_output_tokens",
            "execution_latency_ms",
        )
        expected = {attribute: getattr(record, attribute) for attribute in attributes}
        expected["evidence_sequence"] = evidence["sequence"]
        if any(row[key] != value for key, value in expected.items()):
            raise IntegrityError(
                f"model_outputs projection mismatch for evidence sequence {evidence['sequence']}"
            )

    def verify_integrity(self) -> IntegrityReport:
        """Validate all hashes, v2 payloads, indexes, projections, and references.

        Returns a report only on success; the first detected defect raises
        :class:`IntegrityError` with the affected evidence/table identity.
        """
        self._ensure_open()
        evidence_rows = self._connection.execute(
            "SELECT * FROM evidence_log ORDER BY sequence"
        ).fetchall()
        counts: Counter[str] = Counter()
        for row in evidence_rows:
            record = self._decode_evidence(row)
            self._verify_typed_projection(row, record)
            self._verify_record_references(record)
            counts[row["record_type"]] += 1

        for spec in _SPECS:
            typed_rows = self._connection.execute(
                f"SELECT {spec.id_attribute}, evidence_sequence FROM {spec.table}"
            ).fetchall()
            for typed in typed_rows:
                evidence = self._connection.execute(
                    "SELECT record_type, record_id FROM evidence_log WHERE sequence = ?",
                    (typed["evidence_sequence"],),
                ).fetchone()
                if (
                    evidence is None
                    or evidence["record_type"] != spec.record_type
                    or evidence["record_id"] != typed[spec.id_attribute]
                ):
                    raise IntegrityError(
                        f"orphan or mismatched {spec.table} typed row: "
                        f"{typed[spec.id_attribute]}"
                    )
        model_rows = self._connection.execute(
            "SELECT outcome_id, evidence_sequence FROM model_outputs"
        ).fetchall()
        for model_row in model_rows:
            outcome = self._connection.execute(
                "SELECT evidence_sequence FROM outcomes WHERE outcome_id = ?",
                (model_row["outcome_id"],),
            ).fetchone()
            if (
                outcome is None
                or outcome["evidence_sequence"] != model_row["evidence_sequence"]
            ):
                raise IntegrityError(
                    f"orphan or mismatched model_outputs row: {model_row['outcome_id']}"
                )
        return IntegrityReport(
            evidence_rows=len(evidence_rows),
            per_type_counts=MappingProxyType(dict(sorted(counts.items()))),
        )

    def _verify_record_references(self, record: RecordMixin) -> None:
        try:
            if isinstance(record, RunManifest):
                return
            if isinstance(record, SelectionDecision):
                self._require_row("experiment_runs", "run_id", record.run_id, "run")
                return
            if isinstance(record, TaskOutcome):
                self._require_row("experiment_runs", "run_id", record.run_id, "run")
                decision = self._require_row(
                    "selection_decisions",
                    "decision_id",
                    record.selection_decision_id,
                    "selection",
                )
                if decision["run_id"] != record.run_id:
                    raise ReferenceIntegrityError(
                        "outcome and selection run IDs disagree"
                    )
                if decision["task_case_id"] != record.task_case_id:
                    raise ReferenceIntegrityError(
                        "outcome and selection task-case IDs disagree"
                    )
                return
            if isinstance(record, FeedbackEvent):
                self._require_row("experiment_runs", "run_id", record.run_id, "run")
                return
            if isinstance(record, UtilityEstimate):
                for event_id in record.source_event_ids:
                    event = self._require_row(
                        "feedback_events", "event_id", event_id, "feedback event"
                    )
                    if event["task_family_id"] != record.task_family_id:
                        raise ReferenceIntegrityError(
                            f"feedback event {event_id} task family disagrees"
                        )
                return
            if isinstance(record, ExperimentResult):
                self._validate_result_references(record)
        except ReferenceIntegrityError as error:
            raise IntegrityError(f"reference integrity failure: {error}") from error
