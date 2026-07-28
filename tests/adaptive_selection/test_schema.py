import json
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.adaptive_selection.schema import (
    SCHEMA_VERSION,
    ContextItem,
    CriterionScore,
    ExperimentResult,
    FeedbackEvent,
    RubricCriterion,
    RunManifest,
    ScoringRubric,
    SealedEvaluation,
    SelectionDecision,
    TaskCase,
    TaskInputs,
    TaskOutcome,
    TaskProfile,
    UtilityEstimate,
)


def sample_case(split="adaptation"):
    profile = TaskProfile(
        task_family_id="support",
        name="Support reply",
        description="Answer a customer question",
        default_token_budget=100,
        created_timestamp="2026-07-28T12:00:00Z",
        provenance="fixture:task-profile-v1",
    )
    candidates = (
        ContextItem(
            context_item_id="policy",
            content="Refunds are available within 30 days.",
            token_count=8,
            source="handbook",
            confidence=0.95,
            created_timestamp="2026-07-28T12:00:00Z",
            provenance="fixture:policy-v1",
        ),
        ContextItem(
            context_item_id="noise",
            content="The office is painted blue.",
            token_count=6,
            source="notes",
            confidence=0.4,
            created_timestamp="2026-07-28T12:00:00Z",
            provenance="fixture:notes-v1",
        ),
    )
    rubric = ScoringRubric(
        rubric_id="refund-rubric",
        instructions="Score only factual refund guidance.",
        criteria=(RubricCriterion("accuracy", "Correct policy", 1.0),),
        provenance="fixture:rubric-v1",
    )
    return TaskCase(
        task_case_id="case-1",
        split=split,
        inputs=TaskInputs(
            profile=profile,
            task_prompt="Can I return this after two weeks?",
            candidate_context=candidates,
            token_budget=40,
            visible_metadata={"locale": "en-US"},
            provenance="fixture:inputs-v1",
        ),
        sealed_evaluation=SealedEvaluation(
            gold_answer="Yes, within 30 days.",
            scoring_rubric=rubric,
            required_context_item_ids=("policy",),
            useful_context_item_ids=(),
            misleading_context_item_ids=("noise",),
            irrelevant_context_item_ids=(),
            provenance="fixture:sealed-v1",
        ),
        dataset_version="support-v1",
        created_timestamp="2026-07-28T12:00:00Z",
        provenance="fixture:case-v1",
    )


def all_records():
    case = sample_case()
    manifest = RunManifest(
        run_id="run-1",
        experiment_version="adaptive-selection-v1",
        protocol_version="protocol-v1",
        dataset_version="support-v1",
        dataset_hash="sha256:dataset",
        selector_mode="baseline",
        selector_version="selector-v1",
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
        provenance="fixture:manifest-v1",
    )
    decision = SelectionDecision(
        decision_id="decision-1",
        run_id="run-1",
        task_case_id="case-1",
        selected_context_item_ids=("policy",),
        selected_token_counts=(8,),
        total_selected_tokens=8,
        token_budget=40,
        selector_score=0.8,
        selector_input_hash="sha256:selector-input",
        candidate_set_hash="sha256:candidates",
        decision_latency_ms=12.5,
        ranking_artifact_reference="artifact://ranking/1",
        ranking_artifact_hash="sha256:ranking",
        trace_artifact_reference=None,
        trace_artifact_hash=None,
        decided_timestamp="2026-07-28T12:01:00Z",
        provenance="fixture:decision-v1",
    )
    outcome = TaskOutcome(
        outcome_id="outcome-1",
        run_id="run-1",
        task_case_id="case-1",
        selection_decision_id="decision-1",
        response_text="Yes.",
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
        evaluation_artifact_reference="artifact://evaluation/1",
        evaluation_artifact_hash="sha256:evaluation",
        model_input_tokens=20,
        model_output_tokens=4,
        execution_latency_ms=125.0,
        provider_response_artifact_reference="artifact://provider-response/1",
        provider_response_hash="sha256:response",
        error_category=None,
        completed_timestamp="2026-07-28T12:02:00Z",
        provenance="fixture:outcome-v1",
    )
    event = FeedbackEvent(
        event_id="event-1",
        run_id="run-1",
        task_case_id="case-1",
        task_family_id="support",
        signal_type="context_utility",
        numeric_value=0.75,
        structured_value=None,
        affected_context_item_ids=("policy",),
        correction_category=None,
        correction_text=None,
        source="oracle",
        occurred_timestamp="2026-07-28T12:03:00Z",
        provenance="fixture:oracle-v1",
    )
    estimate = UtilityEstimate(
        utility_estimate_id="estimate-1",
        task_family_id="support",
        context_attributes=("source:handbook", "kind:policy"),
        estimated_utility=0.7,
        confidence=0.8,
        source_event_ids=("event-1",),
        estimator_version="mean-v1",
        estimated_timestamp="2026-07-28T12:04:00Z",
        provenance="fixture:estimate-v1",
    )
    result = ExperimentResult(
        experiment_result_id="result-1",
        run_id="run-1",
        outcome_ids=("outcome-1",),
        selection_decision_ids=("decision-1",),
        feedback_event_ids=("event-1",),
        utility_estimate_ids=("estimate-1",),
        completed_timestamp="2026-07-28T12:05:00Z",
        provenance="fixture:result-v1",
    )
    return (
        case.inputs.profile,
        *case.inputs.candidate_context,
        case.sealed_evaluation.scoring_rubric.criteria[0],
        case.sealed_evaluation.scoring_rubric,
        outcome.criterion_scores[0],
        case.inputs,
        case.sealed_evaluation,
        case,
        manifest,
        decision,
        outcome,
        event,
        estimate,
        result,
    )


def record_of_type(record_type):
    return next(record for record in all_records() if isinstance(record, record_type))


@pytest.mark.parametrize("record", all_records())
def test_all_public_records_round_trip_json(record):
    payload = record.to_dict()
    reconstructed = type(record).from_dict(
        json.loads(json.dumps(payload, sort_keys=True))
    )

    assert reconstructed == record
    assert reconstructed.to_dict() == payload
    assert payload["schema_version"] == SCHEMA_VERSION


@pytest.mark.parametrize("record", all_records())
def test_missing_and_unsupported_schema_versions_fail(record):
    payload = record.to_dict()
    payload.pop("schema_version")
    with pytest.raises(ValueError, match="schema_version is required"):
        type(record).from_dict(payload)

    payload["schema_version"] = "999"
    with pytest.raises(ValueError, match="unsupported schema_version: 999"):
        type(record).from_dict(payload)


def test_task_case_has_explicit_selector_visible_seam():
    case = sample_case("held_out")

    visible = case.selector_inputs()

    assert visible is case.inputs
    assert "gold_answer" not in visible.to_dict()
    assert "scoring_rubric" not in visible.to_dict()
    assert case.sealed_evaluation.gold_answer


def test_task_case_validates_split_candidate_ids_and_sealed_sets():
    case = sample_case()
    with pytest.raises(ValueError, match="split must be adaptation or held_out"):
        TaskCase.from_dict({**case.to_dict(), "split": "training"})

    data = case.to_dict()
    data["inputs"]["candidate_context"].append(data["inputs"]["candidate_context"][0])
    with pytest.raises(ValueError, match="candidate context IDs must be unique"):
        TaskCase.from_dict(data)

    data = case.to_dict()
    data["sealed_evaluation"]["useful_context_item_ids"] = ["policy"]
    with pytest.raises(
        ValueError, match="sealed context ID sets must be pairwise disjoint"
    ):
        TaskCase.from_dict(data)

    data = case.to_dict()
    data["sealed_evaluation"]["required_context_item_ids"] = ["missing"]
    with pytest.raises(ValueError, match="sealed context IDs must refer to candidates"):
        TaskCase.from_dict(data)


def test_held_out_visible_metadata_rejects_feedback_and_gold_leakage():
    case = sample_case("held_out")
    for key in ("gold_answer", "adaptation_feedback"):
        data = case.to_dict()
        data["inputs"]["visible_metadata"] = {"nested": {key: "leak"}}
        with pytest.raises(ValueError, match="held_out selector-visible metadata"):
            TaskCase.from_dict(data)

    data = case.to_dict()
    data["inputs"]["candidate_context"][0]["metadata"] = {"adaptation_feedback": "leak"}
    with pytest.raises(ValueError, match="held_out selector-visible metadata"):
        TaskCase.from_dict(data)


def test_numeric_ranges_and_positive_budgets_are_validated():
    item = sample_case().inputs.candidate_context[0].to_dict()
    item["confidence"] = 1.1
    with pytest.raises(ValueError, match="confidence must be between 0 and 1"):
        ContextItem.from_dict(item)

    inputs = sample_case().inputs.to_dict()
    inputs["token_budget"] = 0
    with pytest.raises(ValueError, match="token_budget must be positive"):
        TaskInputs.from_dict(inputs)

    outcome = record_of_type(TaskOutcome).to_dict()
    outcome["normalized_score"] = -0.1
    with pytest.raises(ValueError, match="normalized_score must be between 0 and 1"):
        TaskOutcome.from_dict(outcome)


def test_feedback_values_are_explicit_and_validated():
    event = record_of_type(FeedbackEvent)
    for changes, message in (
        ({"source": "telepathy"}, "source must be one of"),
        ({"signal_type": "vibes"}, "signal_type must be one of"),
        (
            {"numeric_value": None, "structured_value": None},
            "feedback requires exactly one of numeric_value or structured_value",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            FeedbackEvent.from_dict({**event.to_dict(), **changes})


def test_utility_event_ids_are_nonempty_and_deduplicated():
    estimate = record_of_type(UtilityEstimate)
    with pytest.raises(ValueError, match="source_event_ids must not contain empty IDs"):
        UtilityEstimate.from_dict({**estimate.to_dict(), "source_event_ids": [""]})
    with pytest.raises(ValueError, match="source_event_ids must be unique"):
        UtilityEstimate.from_dict(
            {**estimate.to_dict(), "source_event_ids": ["event-1", "event-1"]}
        )


def test_selection_order_token_counts_and_budget_are_validated():
    decision = record_of_type(SelectionDecision)
    with pytest.raises(ValueError, match="selected context IDs must be unique"):
        SelectionDecision.from_dict(
            {
                **decision.to_dict(),
                "selected_context_item_ids": ["policy", "policy"],
                "selected_token_counts": [4, 4],
            }
        )
    with pytest.raises(ValueError, match="selected IDs and token counts must align"):
        SelectionDecision.from_dict({**decision.to_dict(), "selected_token_counts": []})
    with pytest.raises(ValueError, match="total_selected_tokens must equal"):
        SelectionDecision.from_dict({**decision.to_dict(), "total_selected_tokens": 7})
    with pytest.raises(
        ValueError, match="total_selected_tokens must not exceed token_budget"
    ):
        SelectionDecision.from_dict({**decision.to_dict(), "token_budget": 7})


def test_required_strings_fail_with_specific_message():
    profile = sample_case().inputs.profile
    with pytest.raises(ValueError, match="task_family_id must be nonempty"):
        TaskProfile.from_dict({**profile.to_dict(), "task_family_id": "  "})


def test_json_metadata_is_deeply_frozen_canonical_and_round_trips():
    source = {"z": [{"b": 2, "a": [True, None, 1.5]}], "a": "text"}
    item = ContextItem.from_dict(
        {**sample_case().inputs.candidate_context[0].to_dict(), "metadata": source}
    )
    source["z"][0]["a"].append("later")

    assert list(item.to_dict()["metadata"]) == ["a", "z"]
    assert item.to_dict()["metadata"]["z"][0]["a"] == [True, None, 1.5]
    with pytest.raises(TypeError):
        item.metadata["new"] = 1
    with pytest.raises(TypeError):
        item.metadata["z"][0]["a"][0] = False
    assert ContextItem.from_dict(item.to_dict()) == item

    visible = {"nested": [{"ok": True}]}
    inputs = TaskInputs.from_dict(
        {**sample_case().inputs.to_dict(), "visible_metadata": visible}
    )
    visible["nested"][0]["ok"] = False
    assert inputs.to_dict()["visible_metadata"] == {"nested": [{"ok": True}]}


@pytest.mark.parametrize(
    "bad,message",
    [
        ({"bad": {1, 2}}, "metadata.bad contains unsupported JSON value: set"),
        ({"bad": object()}, "metadata.bad contains unsupported JSON value: object"),
        ({1: "bad"}, "metadata object keys must be strings"),
        ({"bad": float("nan")}, "metadata.bad must be finite"),
        ({"bad": float("inf")}, "metadata.bad must be finite"),
    ],
)
def test_json_metadata_rejects_unsupported_values(bad, message):
    with pytest.raises(ValueError, match=message):
        ContextItem.from_dict(
            {**sample_case().inputs.candidate_context[0].to_dict(), "metadata": bad}
        )


@pytest.mark.parametrize("location", ["prompt", "content", "metadata"])
def test_held_out_rejects_normalized_exact_gold_text(location):
    data = sample_case("held_out").to_dict()
    leak = "prefix YES,   WITHIN 30 DAYS. suffix"
    if location == "prompt":
        data["inputs"]["task_prompt"] = leak
    elif location == "content":
        data["inputs"]["candidate_context"][0]["content"] = leak
    else:
        data["inputs"]["visible_metadata"] = {"nested": [leak]}
    with pytest.raises(ValueError, match="normalized gold answer"):
        TaskCase.from_dict(data)


def test_held_out_allows_nonmatching_visible_text():
    data = sample_case("held_out").to_dict()
    data["inputs"]["visible_metadata"] = {"note": "Returns may be possible."}
    assert TaskCase.from_dict(data).split == "held_out"


@pytest.mark.parametrize("gold", ["id", "schema", "version", "context"])
def test_held_out_gold_does_not_match_structural_wire_keys(gold):
    data = sample_case("held_out").to_dict()
    data["sealed_evaluation"]["gold_answer"] = gold

    assert TaskCase.from_dict(data).sealed_evaluation.gold_answer == gold


@pytest.mark.parametrize("gold", ["id", "schema", "version", "context"])
@pytest.mark.parametrize(
    "location", ["profile_value", "candidate_value", "metadata_key", "metadata_value"]
)
def test_held_out_short_gold_still_matches_selector_visible_values(gold, location):
    data = sample_case("held_out").to_dict()
    data["sealed_evaluation"]["gold_answer"] = gold
    leak = f"prefix {gold} suffix"
    if location == "profile_value":
        data["inputs"]["profile"]["name"] = leak
    elif location == "candidate_value":
        data["inputs"]["candidate_context"][0]["source"] = leak
    elif location == "metadata_key":
        data["inputs"]["visible_metadata"] = {leak: "safe"}
    else:
        data["inputs"]["candidate_context"][0]["metadata"] = {"note": leak}

    with pytest.raises(ValueError, match="normalized gold answer"):
        TaskCase.from_dict(data)


def test_feedback_signed_ranges_encodings_and_structured_round_trip():
    event = record_of_type(FeedbackEvent)
    assert (
        FeedbackEvent.from_dict(
            {**event.to_dict(), "numeric_value": -1.0}
        ).numeric_value
        == -1.0
    )
    for signal, value in (
        ("context_utility", -1.1),
        ("preference", 1.1),
        ("task_score", -0.1),
        ("selection_quality", 1.1),
    ):
        with pytest.raises(ValueError, match="numeric_value"):
            FeedbackEvent.from_dict(
                {**event.to_dict(), "signal_type": signal, "numeric_value": value}
            )
    with pytest.raises(ValueError, match="exactly one"):
        FeedbackEvent.from_dict(
            {**event.to_dict(), "structured_value": {"label": "bad"}}
        )

    structured = FeedbackEvent.from_dict(
        {
            **event.to_dict(),
            "numeric_value": None,
            "structured_value": {"labels": ["useful", {"rank": 1}]},
        }
    )
    assert FeedbackEvent.from_dict(structured.to_dict()) == structured
    with pytest.raises(TypeError):
        structured.structured_value["labels"][1]["rank"] = 2


def test_correction_feedback_requirements_and_exclusivity():
    event = record_of_type(FeedbackEvent).to_dict()
    correction = {
        **event,
        "signal_type": "correction",
        "numeric_value": None,
        "structured_value": {"severity": "major"},
        "correction_category": "factual",
        "correction_text": "Use the 30-day policy.",
    }
    assert FeedbackEvent.from_dict(correction).signal_type == "correction"
    for key in ("correction_category", "correction_text"):
        with pytest.raises(ValueError, match=f"correction requires {key}"):
            FeedbackEvent.from_dict({**correction, key: None})
    with pytest.raises(ValueError, match="correction-only fields"):
        FeedbackEvent.from_dict({**event, "correction_text": "wrong"})


def test_selection_audit_fields_tokens_latency_and_pairs():
    decision = record_of_type(SelectionDecision).to_dict()
    empty = {
        **decision,
        "selected_context_item_ids": [],
        "selected_token_counts": [],
        "total_selected_tokens": 0,
    }
    assert SelectionDecision.from_dict(empty).total_selected_tokens == 0
    for value in (-1, 1.5, True):
        with pytest.raises(
            ValueError, match="total_selected_tokens must be a nonnegative integer"
        ):
            SelectionDecision.from_dict({**empty, "total_selected_tokens": value})
    with pytest.raises(
        ValueError, match="zero only when no context items are selected"
    ):
        SelectionDecision.from_dict({**decision, "total_selected_tokens": 0})
    with pytest.raises(
        ValueError, match="decision_latency_ms must be nonnegative and finite"
    ):
        SelectionDecision.from_dict({**decision, "decision_latency_ms": float("inf")})
    for prefix, change in (
        ("ranking", {"ranking_artifact_hash": None}),
        ("trace", {"trace_artifact_reference": "artifact://only"}),
    ):
        with pytest.raises(
            ValueError,
            match=(
                f"{prefix}_artifact_reference and {prefix}_artifact_hash "
                "must be provided together"
            ),
        ):
            SelectionDecision.from_dict({**decision, **change})


def test_task_outcome_success_failure_scores_usage_criteria_and_artifacts():
    outcome = record_of_type(TaskOutcome).to_dict()
    with pytest.raises(
        ValueError, match="normalized_score must equal raw_score / max_score"
    ):
        TaskOutcome.from_dict({**outcome, "normalized_score": 0.8})
    with pytest.raises(ValueError, match="criterion IDs must be unique"):
        TaskOutcome.from_dict(
            {**outcome, "criterion_scores": outcome["criterion_scores"] * 2}
        )
    bad_criterion = {**outcome["criterion_scores"][0], "raw_score": 11.0}
    with pytest.raises(ValueError, match="raw_score must be between 0 and max_score"):
        CriterionScore.from_dict(bad_criterion)
    for field in ("model_input_tokens", "model_output_tokens"):
        with pytest.raises(ValueError, match=f"{field} must be a nonnegative integer"):
            TaskOutcome.from_dict({**outcome, field: -1})
    with pytest.raises(
        ValueError, match="execution_latency_ms must be nonnegative and finite"
    ):
        TaskOutcome.from_dict({**outcome, "execution_latency_ms": float("nan")})
    with pytest.raises(
        ValueError,
        match="evaluation_artifact_reference and evaluation_artifact_hash must be provided together",
    ):
        TaskOutcome.from_dict({**outcome, "evaluation_artifact_hash": None})
    with pytest.raises(ValueError, match="error_category must be null for success"):
        TaskOutcome.from_dict({**outcome, "error_category": "provider_error"})

    failure = {
        **outcome,
        "execution_status": "failure",
        "response_text": "",
        "raw_score": None,
        "max_score": None,
        "normalized_score": None,
        "criterion_scores": [],
        "error_category": "provider_error",
    }
    assert TaskOutcome.from_dict(failure).execution_status == "failure"
    with pytest.raises(ValueError, match="error_category is required for failure"):
        TaskOutcome.from_dict({**failure, "error_category": None})


def test_run_manifest_reproducibility_seed_tools_and_timestamp():
    manifest = record_of_type(RunManifest).to_dict()
    for field in (
        "dataset_hash",
        "prompt_template_hash",
        "config_hash",
        "code_revision",
        "model_id",
        "selector_version",
    ):
        with pytest.raises(ValueError, match=f"{field} must be nonempty"):
            RunManifest.from_dict({**manifest, field: ""})
    with pytest.raises(
        ValueError, match="seed must be null when seed_supported is false"
    ):
        RunManifest.from_dict({**manifest, "seed_supported": False})
    with pytest.raises(
        ValueError, match="seed is required when seed_supported is true"
    ):
        RunManifest.from_dict({**manifest, "seed": None})
    with pytest.raises(ValueError, match="tool_availability must be unique"):
        RunManifest.from_dict(
            {**manifest, "tool_availability": ["calculator", "calculator"]}
        )


def test_utility_source_event_ids_reject_empty_tuple_explicitly():
    estimate = record_of_type(UtilityEstimate)
    with pytest.raises(ValueError, match="source_event_ids must not be empty"):
        UtilityEstimate.from_dict({**estimate.to_dict(), "source_event_ids": []})


def test_v2_is_first_candidate_wire_format_and_unreleased_v1_is_rejected_early():
    fixture = Path(__file__).with_name("fixtures") / "selection_decision_v1.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    # Prove version rejection wins over later tuple conversion/constructor processing.
    payload["selected_token_counts"] = None

    assert SCHEMA_VERSION == "2"
    with pytest.raises(ValueError, match="unsupported schema_version: 1"):
        SelectionDecision.from_dict(payload)


@pytest.mark.parametrize(
    "key",
    [
        "gold_answer",
        "Gold Answer",
        "gold-answer",
        "gold___---   answer",
        "REQUIRED Context---Item IDs",
    ],
)
def test_held_out_metadata_key_denylist_normalizes_case_and_separators(key):
    data = sample_case("held_out").to_dict()
    data["inputs"]["visible_metadata"] = {"nested": {key: "not the answer"}}
    with pytest.raises(ValueError, match="held_out selector-visible metadata"):
        TaskCase.from_dict(data)


@pytest.mark.parametrize(
    "location",
    [
        "profile_description",
        "candidate_source",
        "candidate_id",
        "candidate_provenance",
    ],
)
def test_held_out_scans_every_string_in_complete_task_inputs_payload(location):
    data = sample_case("held_out").to_dict()
    leak = "prefix YES,   WITHIN 30 DAYS. suffix"
    if location == "profile_description":
        data["inputs"]["profile"]["description"] = leak
    elif location == "candidate_source":
        data["inputs"]["candidate_context"][0]["source"] = leak
    elif location == "candidate_id":
        data["inputs"]["candidate_context"][0]["context_item_id"] = leak
        data["sealed_evaluation"]["required_context_item_ids"] = [leak]
    else:
        data["inputs"]["candidate_context"][0]["provenance"] = leak
    with pytest.raises(ValueError, match="normalized gold answer"):
        TaskCase.from_dict(data)


def test_held_out_complete_payload_scan_allows_safe_nonmatching_strings():
    data = sample_case("held_out").to_dict()
    data["inputs"]["profile"]["description"] = "General return guidance."
    data["inputs"]["candidate_context"][0]["source"] = "support portal"
    data["inputs"]["candidate_context"][0]["provenance"] = "fixture:safe"
    assert TaskCase.from_dict(data).split == "held_out"


def test_nested_record_types_are_validated_before_dereferencing():
    case = sample_case()
    outcome = record_of_type(TaskOutcome)
    constructors = (
        (
            lambda: TaskInputs(
                profile={"task_family_id": "wrong"},
                task_prompt="prompt",
                candidate_context=case.inputs.candidate_context,
                token_budget=10,
                visible_metadata={},
                provenance="test",
            ),
            "profile must be a TaskProfile record",
        ),
        (
            lambda: TaskInputs(
                profile=case.inputs.profile,
                task_prompt="prompt",
                candidate_context=({"context_item_id": "wrong"},),
                token_budget=10,
                visible_metadata={},
                provenance="test",
            ),
            "candidate_context must contain ContextItem records",
        ),
        (
            lambda: ScoringRubric(
                rubric_id="rubric",
                instructions="score",
                criteria=({"criterion_id": "wrong"},),
                provenance="test",
            ),
            "criteria must contain RubricCriterion records",
        ),
        (
            lambda: SealedEvaluation(
                gold_answer="answer",
                scoring_rubric={"rubric_id": "wrong"},
                required_context_item_ids=(),
                useful_context_item_ids=(),
                misleading_context_item_ids=(),
                irrelevant_context_item_ids=(),
                provenance="test",
            ),
            "scoring_rubric must be a ScoringRubric record",
        ),
        (
            lambda: TaskCase(
                task_case_id="case",
                split="adaptation",
                inputs={"task_prompt": "wrong"},
                sealed_evaluation=case.sealed_evaluation,
                dataset_version="v1",
                created_timestamp="2026-07-28T12:00:00Z",
                provenance="test",
            ),
            "inputs must be a TaskInputs record",
        ),
        (
            lambda: TaskCase(
                task_case_id="case",
                split="adaptation",
                inputs=case.inputs,
                sealed_evaluation={"gold_answer": "wrong"},
                dataset_version="v1",
                created_timestamp="2026-07-28T12:00:00Z",
                provenance="test",
            ),
            "sealed_evaluation must be a SealedEvaluation record",
        ),
        (
            lambda: replace(
                outcome,
                criterion_scores=({"criterion_id": "wrong"},),
            ),
            "criterion_scores must contain CriterionScore records",
        ),
    )
    for construct, message in constructors:
        with pytest.raises(ValueError, match=message):
            construct()


def test_nested_record_sequences_are_defensively_converted_to_tuples():
    case = sample_case()
    mutable_candidates = list(case.inputs.candidate_context)
    inputs = TaskInputs(
        profile=case.inputs.profile,
        task_prompt="prompt",
        candidate_context=mutable_candidates,
        token_budget=10,
        visible_metadata={},
        provenance="test",
    )
    mutable_candidates.clear()
    assert isinstance(inputs.candidate_context, tuple)
    assert len(inputs.candidate_context) == 2


def test_task_outcome_requires_completed_success_evaluation_evidence():
    outcome = record_of_type(TaskOutcome)
    payload = outcome.to_dict()
    for changes, message in (
        (
            {"raw_score": None, "max_score": None, "normalized_score": None},
            "aggregate scores are required for success",
        ),
        ({"criterion_scores": []}, "criterion_scores are required for success"),
        (
            {
                "evaluation_artifact_reference": None,
                "evaluation_artifact_hash": None,
            },
            "evaluation artifact is required for success",
        ),
        (
            {
                "provider_response_artifact_reference": None,
                "provider_response_hash": None,
            },
            "provider response artifact is required for success",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            TaskOutcome.from_dict({**payload, **changes})


def test_task_outcome_rejects_scored_failure_and_accepts_valid_records():
    success = record_of_type(TaskOutcome)
    assert TaskOutcome.from_dict(success.to_dict()) == success
    failure = {
        **success.to_dict(),
        "execution_status": "failure",
        "response_text": "",
        "raw_score": None,
        "max_score": None,
        "normalized_score": None,
        "criterion_scores": [],
        "evaluation_artifact_reference": None,
        "evaluation_artifact_hash": None,
        "provider_response_artifact_reference": None,
        "provider_response_hash": None,
        "error_category": "provider_error",
    }
    assert TaskOutcome.from_dict(failure).execution_status == "failure"
    with pytest.raises(ValueError, match="aggregate scores must be null for failure"):
        TaskOutcome.from_dict({**failure, "raw_score": 1.0})
    with pytest.raises(ValueError, match="criterion_scores must be empty for failure"):
        TaskOutcome.from_dict(
            {**failure, "criterion_scores": success.to_dict()["criterion_scores"]}
        )


def test_task_outcome_aggregation_and_provider_response_evidence():
    payload = record_of_type(TaskOutcome).to_dict()
    for field in ("aggregation_method", "aggregation_version"):
        with pytest.raises(ValueError, match=f"{field} must be nonempty"):
            TaskOutcome.from_dict({**payload, field: ""})
    for changes in (
        {"provider_response_artifact_reference": None},
        {"provider_response_hash": None},
    ):
        with pytest.raises(
            ValueError,
            match=(
                "provider_response_artifact_reference and provider_response_hash "
                "must be provided together"
            ),
        ):
            TaskOutcome.from_dict({**payload, **changes})


def _timestamp_records():
    records = all_records()
    expected = (
        (TaskProfile, "created_timestamp"),
        (ContextItem, "created_timestamp"),
        (TaskCase, "created_timestamp"),
        (RunManifest, "started_timestamp"),
        (SelectionDecision, "decided_timestamp"),
        (TaskOutcome, "completed_timestamp"),
        (FeedbackEvent, "occurred_timestamp"),
        (UtilityEstimate, "estimated_timestamp"),
        (ExperimentResult, "completed_timestamp"),
    )
    return tuple(
        (next(record for record in records if isinstance(record, record_type)), field)
        for record_type, field in expected
    )


@pytest.mark.parametrize("record,timestamp_field", _timestamp_records())
@pytest.mark.parametrize(
    "bad",
    [
        "2026-07-28T12:00:00",
        "2026-07-28 12:00:00Z",
        "2026-07-28T12:00:00+00:00",
        "not-a-time",
    ],
)
def test_every_timestamp_bearing_record_rejects_noncanonical_values(
    record, timestamp_field, bad
):
    with pytest.raises(
        ValueError, match=f"{timestamp_field} must be canonical UTC RFC 3339"
    ):
        type(record).from_dict({**record.to_dict(), timestamp_field: bad})


@pytest.mark.parametrize("record", all_records())
def test_public_from_dict_rejects_non_mapping_payloads(record):
    with pytest.raises(ValueError, match="payload must be an object"):
        type(record).from_dict(None)


@pytest.mark.parametrize(
    "record_type,field",
    [
        (ScoringRubric, "criteria"),
        (TaskInputs, "candidate_context"),
        (SealedEvaluation, "required_context_item_ids"),
        (SealedEvaluation, "useful_context_item_ids"),
        (SealedEvaluation, "misleading_context_item_ids"),
        (SealedEvaluation, "irrelevant_context_item_ids"),
        (RunManifest, "tool_availability"),
        (SelectionDecision, "selected_context_item_ids"),
        (SelectionDecision, "selected_token_counts"),
        (TaskOutcome, "criterion_scores"),
        (FeedbackEvent, "affected_context_item_ids"),
        (UtilityEstimate, "context_attributes"),
        (UtilityEstimate, "source_event_ids"),
        (ExperimentResult, "outcome_ids"),
        (ExperimentResult, "selection_decision_ids"),
        (ExperimentResult, "feedback_event_ids"),
        (ExperimentResult, "utility_estimate_ids"),
    ],
)
def test_serialized_sequence_fields_are_required(record_type, field):
    payload = record_of_type(record_type).to_dict()
    payload.pop(field)

    with pytest.raises(ValueError, match=rf"^{field} is required$"):
        record_type.from_dict(payload)


@pytest.mark.parametrize(
    "record_type,field",
    [
        (TaskInputs, "profile"),
        (SealedEvaluation, "scoring_rubric"),
        (TaskCase, "inputs"),
        (TaskCase, "sealed_evaluation"),
    ],
)
def test_nested_wire_records_are_required(record_type, field):
    payload = record_of_type(record_type).to_dict()
    payload.pop(field)

    with pytest.raises(ValueError, match=rf"^{field} is required$"):
        record_type.from_dict(payload)


@pytest.mark.parametrize(
    "record_type,field,bad_value",
    [
        (TaskInputs, "profile", []),
        (TaskInputs, "candidate_context", ["not an object"]),
        (ScoringRubric, "criteria", ["not an object"]),
        (SealedEvaluation, "scoring_rubric", []),
        (TaskCase, "inputs", []),
        (TaskCase, "sealed_evaluation", "not an object"),
        (TaskOutcome, "criterion_scores", [7]),
    ],
)
def test_malformed_nested_wire_records_raise_value_error(record_type, field, bad_value):
    payload = record_of_type(record_type).to_dict()
    payload[field] = bad_value

    with pytest.raises(ValueError, match="must be an object"):
        record_type.from_dict(payload)
