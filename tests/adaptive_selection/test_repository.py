import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from experiments.adaptive_selection.repository import (
    _APPEND_ONLY_TABLES,
    _append_only_trigger_sql,
    DuplicateRecordError,
    ExperimentRepository,
    IntegrityError,
    ReferenceIntegrityError,
)
from experiments.adaptive_selection.schema import (
    CriterionScore,
    ExperimentResult,
    FeedbackEvent,
    RunManifest,
    SelectionDecision,
    TaskOutcome,
    UtilityEstimate,
)

INSERTED_AT = datetime(2026, 7, 28, 13, 0, 0, tzinfo=timezone.utc)


def records():
    run = RunManifest(
        run_id="run-1",
        experiment_version="adaptive-selection-v1",
        protocol_version="protocol-v1",
        dataset_version="support-v1",
        dataset_hash="sha256:dataset",
        selector_mode="adaptive",
        selector_version="selector-v2",
        provider="example-provider",
        model_id="exact-model-id",
        prompt_template_hash="sha256:prompt",
        config_hash="sha256:config",
        code_revision="abc123",
        temperature=0.0,
        seed=7,
        seed_supported=True,
        tool_availability=("calculator",),
        started_timestamp="2026-07-28T12:00:00Z",
        provenance="fixture:manifest:科学",
    )
    decision = SelectionDecision(
        decision_id="decision-1",
        run_id=run.run_id,
        task_case_id="case-1",
        selected_context_item_ids=("policy",),
        selected_token_counts=(8,),
        total_selected_tokens=8,
        token_budget=40,
        selector_score=0.8,
        selector_input_hash="sha256:selector-input",
        candidate_set_hash="sha256:candidates",
        decision_latency_ms=12.5,
        ranking_artifact_reference="opaque-ranking-id",
        ranking_artifact_hash="sha256:ranking",
        trace_artifact_reference=None,
        trace_artifact_hash=None,
        decided_timestamp="2026-07-28T12:01:00Z",
        provenance="fixture:decision:é",
    )
    outcome = TaskOutcome(
        outcome_id="outcome-1",
        run_id=run.run_id,
        task_case_id=decision.task_case_id,
        selection_decision_id=decision.decision_id,
        response_text="Oui — réponse naïve 🚀",
        execution_status="success",
        raw_score=9.0,
        max_score=10.0,
        normalized_score=0.9,
        rubric_id="refund-rubric",
        scorer_id="exact-match-judge",
        scorer_version="v1",
        scorer_hash="sha256:scorer",
        aggregation_method="rubric-weighted-sum",
        aggregation_version="v1",
        criterion_scores=(CriterionScore("accuracy", 9.0, 10.0, 0.9),),
        evaluation_artifact_reference="opaque-evaluation-id",
        evaluation_artifact_hash="sha256:evaluation",
        model_input_tokens=20,
        model_output_tokens=4,
        execution_latency_ms=125.0,
        provider_response_artifact_reference="opaque-provider-response-id",
        provider_response_hash="sha256:response",
        error_category=None,
        completed_timestamp="2026-07-28T12:02:00Z",
        provenance="fixture:outcome:日本語",
    )
    feedback = FeedbackEvent(
        event_id="event-1",
        run_id=run.run_id,
        task_case_id=decision.task_case_id,
        task_family_id="support",
        signal_type="context_utility",
        numeric_value=0.75,
        structured_value=None,
        affected_context_item_ids=("policy",),
        correction_category=None,
        correction_text=None,
        source="oracle",
        occurred_timestamp="2026-07-28T12:03:00Z",
        provenance="fixture:oracle:Δ",
    )
    estimate = UtilityEstimate(
        utility_estimate_id="estimate-1",
        task_family_id=feedback.task_family_id,
        context_attributes=("source:handbook", "kind:policy"),
        estimated_utility=0.7,
        confidence=0.8,
        source_event_ids=(feedback.event_id,),
        estimator_version="learning-policy-v1",
        estimated_timestamp="2026-07-28T12:04:00Z",
        provenance="fixture:estimate:ü",
    )
    result = ExperimentResult(
        experiment_result_id="result-1",
        run_id=run.run_id,
        outcome_ids=(outcome.outcome_id,),
        selection_decision_ids=(decision.decision_id,),
        feedback_event_ids=(feedback.event_id,),
        utility_estimate_ids=(estimate.utility_estimate_id,),
        completed_timestamp="2026-07-28T12:05:00Z",
        provenance="fixture:result:✓",
    )
    return run, decision, outcome, feedback, estimate, result


def repository(database=":memory:"):
    return ExperimentRepository(database, clock=lambda: INSERTED_AT)


def append_complete(repo):
    run, decision, outcome, feedback, estimate, result = records()
    repo.append_run(run)
    repo.append_selection(decision)
    repo.append_outcome(outcome)
    repo.append_feedback(feedback)
    repo.append_utility_estimate(estimate)
    repo.append_experiment_result(result)
    return run, decision, outcome, feedback, estimate, result


def replace_trigger(connection, table, operation, replacement_sql=None):
    trigger_name = f"{table}_no_{operation.lower()}"
    connection.execute(f"DROP TRIGGER {trigger_name}")
    if replacement_sql is not None:
        connection.execute(replacement_sql)


def restore_trigger(connection, table, operation):
    connection.execute(_append_only_trigger_sql(table, operation))


def test_complete_run_typed_round_trip_integrity_and_canonical_json():
    expected = records()
    with repository() as repo:
        append_complete(repo)

        assert repo.load_run("run-1") == expected[0]
        assert repo.load_selection("decision-1") == expected[1]
        assert repo.load_outcome("outcome-1") == expected[2]
        assert repo.load_feedback("event-1") == expected[3]
        assert repo.load_utility_estimate("estimate-1") == expected[4]
        assert repo.load_experiment_result("result-1") == expected[5]
        assert repo.list_runs() == [expected[0]]
        assert repo.list_selections() == [expected[1]]
        assert repo.list_outcomes() == [expected[2]]
        assert repo.list_feedback() == [expected[3]]
        assert repo.list_utility_estimates() == [expected[4]]
        assert repo.list_experiment_results() == [expected[5]]

        evidence = repo.list_evidence()
        assert [entry.sequence for entry in evidence] == [1, 2, 3, 4, 5, 6]
        assert [entry.record_type for entry in evidence] == [
            "run_manifest",
            "selection_decision",
            "task_outcome",
            "feedback_event",
            "utility_estimate",
            "experiment_result",
        ]
        assert all(
            entry.inserted_timestamp == "2026-07-28T13:00:00Z" for entry in evidence
        )
        for entry in evidence:
            assert (
                entry.payload_hash
                == hashlib.sha256(entry.payload_json.encode("utf-8")).hexdigest()
            )
            assert entry.payload_json == json.dumps(
                json.loads(entry.payload_json),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )

        report = repo.verify_integrity()
        assert report.ok
        assert report.evidence_rows == 6
        assert report.per_type_counts["task_outcome"] == 1


def test_global_and_per_type_order_follow_evidence_sequence():
    run, decision, outcome, feedback, _, _ = records()
    second_feedback = replace(
        feedback,
        event_id="event-2",
        numeric_value=-0.25,
        occurred_timestamp="2026-07-28T12:03:30Z",
    )
    second_decision = replace(
        decision,
        decision_id="decision-2",
        task_case_id="case-2",
        decided_timestamp="2026-07-28T12:01:30Z",
    )
    with repository() as repo:
        repo.append_run(run)
        repo.append_feedback(feedback)
        repo.append_selection(decision)
        repo.append_feedback(second_feedback)
        repo.append_selection(second_decision)
        repo.append_outcome(outcome)

        assert [entry.record_id for entry in repo.list_evidence()] == [
            "run-1",
            "event-1",
            "decision-1",
            "event-2",
            "decision-2",
            "outcome-1",
        ]
        assert [item.event_id for item in repo.list_feedback()] == [
            "event-1",
            "event-2",
        ]
        assert [item.decision_id for item in repo.list_selections()] == [
            "decision-1",
            "decision-2",
        ]


def test_unicode_and_provenance_are_preserved_without_ascii_escaping():
    with repository() as repo:
        append_complete(repo)
        entry = next(e for e in repo.list_evidence() if e.record_id == "outcome-1")

        assert "日本語" in entry.payload_json
        assert "🚀" in entry.payload_json
        assert "\\u65e5" not in entry.payload_json
        assert repo.load_outcome("outcome-1").provenance == "fixture:outcome:日本語"


def test_duplicate_record_id_is_rejected_explicitly_without_extra_evidence():
    run = records()[0]
    with repository() as repo:
        repo.append_run(run)
        with pytest.raises(DuplicateRecordError, match="run_manifest.*run-1"):
            repo.append_run(run)
        assert len(repo.list_evidence()) == 1


def test_all_repository_tables_block_sql_updates_and_deletes(tmp_path):
    path = tmp_path / "evidence.sqlite3"
    with repository(path) as repo:
        append_complete(repo)

    with sqlite3.connect(path) as connection:
        for table in _APPEND_ONLY_TABLES:
            with pytest.raises(
                sqlite3.IntegrityError, match=f"append-only table: {table}"
            ):
                connection.execute(f"UPDATE {table} SET rowid = rowid")
            with pytest.raises(
                sqlite3.IntegrityError, match=f"append-only table: {table}"
            ):
                connection.execute(f"DELETE FROM {table}")


@pytest.mark.parametrize(
    "tamper",
    [
        lambda connection: replace_trigger(connection, "evidence_log", "UPDATE"),
        lambda connection: replace_trigger(
            connection,
            "evidence_log",
            "UPDATE",
            """CREATE TRIGGER evidence_log_no_update
               BEFORE UPDATE ON evidence_log BEGIN SELECT 1; END""",
        ),
        lambda connection: replace_trigger(
            connection,
            "evidence_log",
            "UPDATE",
            """CREATE TRIGGER evidence_log_no_update
               AFTER UPDATE ON evidence_log
               BEGIN SELECT RAISE(ABORT, 'append-only table: evidence_log'); END""",
        ),
        lambda connection: replace_trigger(
            connection,
            "evidence_log",
            "UPDATE",
            """CREATE TRIGGER evidence_log_no_update
               BEFORE UPDATE ON experiment_runs
               BEGIN SELECT RAISE(ABORT, 'append-only table: evidence_log'); END""",
        ),
        lambda connection: replace_trigger(
            connection,
            "evidence_log",
            "UPDATE",
            """CREATE TRIGGER evidence_log_no_update
               BEFORE UPDATE ON evidence_log
               BEGIN SELECT RAISE(ABORT, 'append-only  table: evidence_log'); END""",
        ),
    ],
    ids=["missing", "inert", "wrong-timing", "wrong-table", "wrong-raise-body"],
)
def test_existing_repository_rejects_missing_or_forged_trigger_on_open(
    tmp_path, tamper
):
    path = tmp_path / "tampered-trigger.sqlite3"
    with repository(path):
        pass
    with sqlite3.connect(path) as connection:
        tamper(connection)

    with pytest.raises(IntegrityError, match="append-only trigger"):
        repository(path)


def test_failed_open_closes_its_connection(tmp_path, monkeypatch):
    path = tmp_path / "failed-open.sqlite3"
    with repository(path):
        pass
    with sqlite3.connect(path) as connection:
        replace_trigger(connection, "evidence_log", "DELETE")

    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(
        "experiments.adaptive_selection.repository.sqlite3.connect", tracking_connect
    )
    with pytest.raises(IntegrityError, match="append-only trigger"):
        repository(path)

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        opened[0].execute("SELECT 1")


@pytest.mark.parametrize("tamper_kind", ["missing", "inert"])
def test_integrity_scan_rejects_trigger_tampering_after_open(tmp_path, tamper_kind):
    path = tmp_path / "post-open-trigger.sqlite3"
    repo = repository(path)
    try:
        with sqlite3.connect(path) as connection:
            replacement = None
            if tamper_kind == "inert":
                replacement = """CREATE TRIGGER outcomes_no_delete
                    BEFORE DELETE ON outcomes BEGIN SELECT 1; END"""
            replace_trigger(connection, "outcomes", "DELETE", replacement)

        with pytest.raises(IntegrityError, match="append-only trigger"):
            repo.verify_integrity()
    finally:
        repo.close()


def test_failed_foreign_reference_rolls_back_without_evidence_row():
    decision = records()[1]
    with repository() as repo:
        with pytest.raises(ReferenceIntegrityError, match="run.*run-1"):
            repo.append_selection(decision)
        assert repo.list_evidence() == []

        repo.append_run(records()[0])
        bad_outcome = replace(decision, decision_id="other")
        repo.append_selection(bad_outcome)
        with pytest.raises(ReferenceIntegrityError, match="selection.*decision-1"):
            repo.append_outcome(records()[2])
        assert [e.record_type for e in repo.list_evidence()] == [
            "run_manifest",
            "selection_decision",
        ]


def test_outcome_references_must_agree_with_selection_ids_and_run():
    run, decision, outcome, *_ = records()
    with repository() as repo:
        repo.append_run(run)
        repo.append_selection(decision)
        with pytest.raises(
            ReferenceIntegrityError, match="task_case_id does not match"
        ):
            repo.append_outcome(replace(outcome, task_case_id="case-other"))
        assert len(repo.list_evidence()) == 2


def test_utility_sources_are_validated_and_reconstructed_from_raw_events():
    run, _, _, feedback, estimate, _ = records()
    missing = replace(estimate, source_event_ids=("missing-event",))
    with repository() as repo:
        repo.append_run(run)
        with pytest.raises(
            ReferenceIntegrityError, match="feedback event.*missing-event"
        ):
            repo.append_utility_estimate(missing)
        assert [entry.record_type for entry in repo.list_evidence()] == ["run_manifest"]

        repo.append_feedback(feedback)
        repo.append_utility_estimate(estimate)
        assert repo.load_utility_source_events("estimate-1") == [feedback]


def test_correction_is_new_event_and_original_payload_remains_byte_identical():
    run, _, _, feedback, *_ = records()
    correction = replace(
        feedback,
        event_id="event-correction",
        signal_type="correction",
        numeric_value=None,
        structured_value={"severity": "major"},
        correction_category="factual",
        correction_text="Use the approved 30-day policy.",
        occurred_timestamp="2026-07-28T12:04:00Z",
    )
    with repository() as repo:
        repo.append_run(run)
        repo.append_feedback(feedback)
        original = repo.list_evidence(record_type="feedback_event")[0].payload_json
        repo.append_feedback(correction)

        events = repo.list_feedback()
        assert events == [feedback, correction]
        assert (
            repo.list_evidence(record_type="feedback_event")[0].payload_json == original
        )


def test_hash_tampering_is_detected_on_load_and_integrity_scan(tmp_path):
    path = tmp_path / "tampered.sqlite3"
    with repository(path) as repo:
        repo.append_run(records()[0])

    with sqlite3.connect(path) as connection:
        replace_trigger(connection, "evidence_log", "UPDATE")
        connection.execute(
            "UPDATE evidence_log SET payload_json = replace(payload_json, 'abc123', 'def456')"
        )
        restore_trigger(connection, "evidence_log", "UPDATE")

    with repository(path) as repo:
        with pytest.raises(IntegrityError, match="payload hash mismatch"):
            repo.load_run("run-1")
        with pytest.raises(IntegrityError, match="payload hash mismatch"):
            repo.verify_integrity()


def test_model_outputs_is_bounded_projection_and_integrity_checks_it(tmp_path):
    path = tmp_path / "projection.sqlite3"
    outcome = records()[2]
    with repository(path) as repo:
        repo.append_run(records()[0])
        repo.append_selection(records()[1])
        repo.append_outcome(outcome)

    with sqlite3.connect(path) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(model_outputs)")
        ]
        assert "response_text" not in columns
        row = connection.execute(
            "SELECT run_id, task_case_id, selection_decision_id, execution_status, "
            "model_input_tokens, model_output_tokens FROM model_outputs"
        ).fetchone()
        assert row == ("run-1", "case-1", "decision-1", "success", 20, 4)
        replace_trigger(connection, "model_outputs", "UPDATE")
        connection.execute(
            "UPDATE model_outputs SET model_output_tokens = 99 WHERE outcome_id = 'outcome-1'"
        )
        restore_trigger(connection, "model_outputs", "UPDATE")

    with repository(path) as repo:
        with pytest.raises(IntegrityError, match="model_outputs projection mismatch"):
            repo.verify_integrity()


def test_experiment_result_references_exist_and_match_run():
    run, decision, outcome, feedback, estimate, result = records()
    with repository() as repo:
        repo.append_run(run)
        repo.append_selection(decision)
        repo.append_outcome(outcome)
        repo.append_feedback(feedback)
        repo.append_utility_estimate(estimate)

        missing = replace(result, feedback_event_ids=("missing",))
        with pytest.raises(ReferenceIntegrityError, match="feedback event.*missing"):
            repo.append_experiment_result(missing)
        assert all(e.record_type != "experiment_result" for e in repo.list_evidence())

        other_run = replace(run, run_id="run-2")
        repo.append_run(other_run)
        mismatch = replace(result, run_id="run-2")
        with pytest.raises(
            ReferenceIntegrityError, match="outcome.*does not belong to run-2"
        ):
            repo.append_experiment_result(mismatch)


def test_file_database_reopens_and_recovers_full_evidence_stream(tmp_path):
    path = tmp_path / "experiment.sqlite3"
    with repository(path) as repo:
        append_complete(repo)
        before = repo.list_evidence()

    with repository(path) as reopened:
        assert reopened.list_evidence() == before
        assert tuple(reopened.iter_evidence()) == tuple(before)
        assert reopened.load_experiment_result("result-1") == records()[-1]
        assert reopened.verify_integrity().evidence_rows == 6


def test_schema_v1_raw_payload_is_rejected_by_integrity_scan(tmp_path):
    path = tmp_path / "v1.sqlite3"
    with repository(path):
        pass
    payload = {**records()[0].to_dict(), "schema_version": "1"}
    payload_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with sqlite3.connect(path) as connection:
        connection.execute(
            "INSERT INTO evidence_log(record_type, record_id, schema_version, payload_json, "
            "payload_hash, inserted_timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "run_manifest",
                "raw-v1",
                "1",
                payload_json,
                hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                "2026-07-28T13:00:00Z",
            ),
        )

    with repository(path) as repo:
        with pytest.raises(IntegrityError, match="unsupported schema_version: 1"):
            repo.verify_integrity()


def test_microsecond_insertion_timestamp_is_canonical_and_valid():
    inserted_at = datetime(2026, 7, 28, 13, 0, 0, 123456, tzinfo=timezone.utc)
    with ExperimentRepository(":memory:", clock=lambda: inserted_at) as repo:
        repo.append_run(records()[0])

        assert repo.list_evidence()[0].inserted_timestamp == (
            "2026-07-28T13:00:00.123456Z"
        )
        assert repo.verify_integrity().ok


@pytest.mark.parametrize(
    "tampered_timestamp",
    [
        "not-a-timestamp",
        "2026-07-28T13:00:00",
        "2026-07-28T13:00:00+00:00",
        "2026-07-28T09:00:00-04:00",
        "2026-07-28T13:00:00.1Z",
        "2026-07-28T13:00:00.000000Z",
    ],
    ids=[
        "malformed",
        "timezone-less",
        "utc-offset",
        "non-utc-offset",
        "short-fraction",
        "zero-fraction",
    ],
)
def test_tampered_insertion_timestamp_is_rejected_on_decode_and_integrity(
    tmp_path, tampered_timestamp
):
    path = tmp_path / "timestamp.sqlite3"
    with repository(path) as repo:
        repo.append_run(records()[0])

    with sqlite3.connect(path) as connection:
        replace_trigger(connection, "evidence_log", "UPDATE")
        connection.execute(
            "UPDATE evidence_log SET inserted_timestamp = ?", (tampered_timestamp,)
        )
        restore_trigger(connection, "evidence_log", "UPDATE")

    with repository(path) as repo:
        with pytest.raises(IntegrityError, match="invalid insertion timestamp"):
            repo.load_run("run-1")
        with pytest.raises(IntegrityError, match="invalid insertion timestamp"):
            repo.verify_integrity()


def test_context_manager_closes_connection_cleanly():
    repo = repository()
    with repo:
        repo.append_run(records()[0])
    with pytest.raises(RuntimeError, match="closed"):
        repo.list_runs()
