import json

import pytest

from experiments.adaptive_selection.schema import (
    SCHEMA_VERSION,
    ContextItem,
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
        decided_timestamp="2026-07-28T12:01:00Z",
        provenance="fixture:decision-v1",
    )
    outcome = TaskOutcome(
        outcome_id="outcome-1",
        run_id="run-1",
        task_case_id="case-1",
        selection_decision_id="decision-1",
        response_text="Yes.",
        normalized_score=0.9,
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
        structured_category=None,
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
    data["inputs"]["candidate_context"][0]["metadata"] = {
        "adaptation_feedback": "leak"
    }
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

    outcome = all_records()[-4].to_dict()
    outcome["normalized_score"] = -0.1
    with pytest.raises(ValueError, match="normalized_score must be between 0 and 1"):
        TaskOutcome.from_dict(outcome)


def test_feedback_values_are_explicit_and_validated():
    event = all_records()[-3]
    for changes, message in (
        ({"source": "telepathy"}, "source must be one of"),
        ({"signal_type": "vibes"}, "signal_type must be one of"),
        (
            {"numeric_value": None, "structured_category": None},
            "feedback requires numeric_value or structured_category",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            FeedbackEvent.from_dict({**event.to_dict(), **changes})


def test_utility_event_ids_are_nonempty_and_deduplicated():
    estimate = all_records()[-2]
    with pytest.raises(ValueError, match="source_event_ids must not contain empty IDs"):
        UtilityEstimate.from_dict({**estimate.to_dict(), "source_event_ids": [""]})
    with pytest.raises(ValueError, match="source_event_ids must be unique"):
        UtilityEstimate.from_dict(
            {**estimate.to_dict(), "source_event_ids": ["event-1", "event-1"]}
        )


def test_selection_order_token_counts_and_budget_are_validated():
    decision = all_records()[-5]
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
