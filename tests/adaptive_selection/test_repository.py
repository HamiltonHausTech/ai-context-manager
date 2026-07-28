import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from experiments.adaptive_selection.repository import (
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


def test_sql_update_and_delete_are_blocked_by_triggers(tmp_path):
    path = tmp_path / "evidence.sqlite3"
    with repository(path) as repo:
        append_complete(repo)

    with sqlite3.connect(path) as connection:
        for statement in (
            "UPDATE evidence_log SET payload_hash = 'bad' WHERE sequence = 1",
            "DELETE FROM evidence_log WHERE sequence = 1",
            "UPDATE outcomes SET execution_status = 'failure' WHERE outcome_id = 'outcome-1'",
            "DELETE FROM model_outputs WHERE outcome_id = 'outcome-1'",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(statement)


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
        connection.execute("DROP TRIGGER evidence_log_no_update")
        connection.execute(
            "UPDATE evidence_log SET payload_json = replace(payload_json, 'abc123', 'def456')"
        )

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
        connection.execute("DROP TRIGGER model_outputs_no_update")
        connection.execute(
            "UPDATE model_outputs SET model_output_tokens = 99 WHERE outcome_id = 'outcome-1'"
        )

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


def test_context_manager_closes_connection_cleanly():
    repo = repository()
    with repo:
        repo.append_run(records()[0])
    with pytest.raises(RuntimeError, match="closed"):
        repo.list_runs()
