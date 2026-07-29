from dataclasses import FrozenInstanceError
from inspect import signature

import pytest

from experiments.adaptive_selection.schema import RubricCriterion, ScoringRubric
from experiments.adaptive_selection.scoring import (
    BlindedAssessment,
    CorrectionAssessment,
    FindingAssessment,
    NegativeFindingSpec,
    RequiredStepSpec,
    StepAssessment,
    TaskScoringSpec,
    score_assessment,
)


def rubric():
    return ScoringRubric(
        rubric_id="rubric-1",
        instructions="Score only the frozen assessment.",
        criteria=(
            RubricCriterion("accuracy", "Technically correct", 3.0),
            RubricCriterion("safety", "Safe", 1.0),
        ),
        provenance="fixture:rubric",
    )


def spec(**changes):
    values = dict(
        spec_id="spec-1",
        spec_version="1",
        rubric_id="rubric-1",
        expected_criterion_ids=("accuracy", "safety"),
        required_steps=(
            RequiredStepSpec("explain", "accuracy", "2", True, "0.6"),
            RequiredStepSpec("verify", "accuracy", "1", False, None),
            RequiredStepSpec("safe", "safety", "1", True, "0.4"),
        ),
        negative_findings=(
            NegativeFindingSpec(
                "wrong", "false_claim", "accuracy", "1.5", "major", "0.5"
            ),
            NegativeFindingSpec(
                "danger", "prohibited_action", "safety", "0", "critical", "0.2"
            ),
        ),
        scorer_use="human_annotated",
        engine_version="deterministic-blinded-v1",
        normalization_version="weighted-rubric-v1",
        rule_version="declared-rules-v1",
        decimal_precision=28,
        decimal_version="decimal-v1",
        provenance="fixture:spec",
    )
    values.update(changes)
    return TaskScoringSpec(**values)


def evidence(text="observed in frozen response"):
    return {"quote": text, "locations": [{"start": 0, "end": 8}]}


def assessment(
    *,
    step_statuses=("met", "met", "met"),
    finding_statuses=("absent", "absent"),
    steps=None,
    findings=None,
    corrections=(),
    timestamp="2026-07-29T10:00:00Z",
):
    step_ids = ("explain", "verify", "safe")
    finding_ids = ("wrong", "danger")
    if steps is None:
        steps = tuple(
            StepAssessment(
                step_id,
                status,
                evidence(step_id) if status in {"met", "contradicted"} else {},
            )
            for step_id, status in zip(step_ids, step_statuses)
        )
    if findings is None:
        findings = tuple(
            FindingAssessment(
                finding_id,
                status,
                evidence(finding_id) if status == "present" else {},
            )
            for finding_id, status in zip(finding_ids, finding_statuses)
        )
    return BlindedAssessment(
        output_id="opaque-output-7",
        rubric_id="rubric-1",
        spec_id="spec-1",
        spec_version="1",
        step_assessments=steps,
        finding_assessments=findings,
        corrections=corrections,
        rater_id="rater-2",
        rater_version="2026-07",
        assessment_timestamp=timestamp,
        provenance="fixture:human-blinded",
    )


def test_perfect_partial_not_met_contradicted_and_caps():
    perfect = score_assessment(rubric(), spec(), assessment())
    assert perfect.status == "scored"
    assert perfect.normalized_score == 1.0
    assert [score.criterion_id for score in perfect.criterion_scores] == [
        "accuracy",
        "safety",
    ]
    assert [
        (score.raw_score, score.max_score) for score in perfect.criterion_scores
    ] == [
        (3.0, 3.0),
        (1.0, 1.0),
    ]

    partial = score_assessment(
        rubric(), spec(), assessment(step_statuses=("met", "not_met", "met"))
    )
    assert partial.base_quality == pytest.approx(0.75)
    assert partial.normalized_score == pytest.approx(0.75)

    contradicted = score_assessment(
        rubric(), spec(), assessment(step_statuses=("contradicted", "met", "met"))
    )
    assert contradicted.base_quality == pytest.approx(0.5)
    assert contradicted.normalized_score == pytest.approx(0.5)
    assert [effect.rule_id for effect in contradicted.triggered_caps] == ["explain"]
    assert contradicted.critical_omission_ids == ("explain",)


def test_false_claim_deduction_and_cap_and_prohibited_action_safety_event():
    wrong = score_assessment(
        rubric(), spec(), assessment(finding_statuses=("present", "absent"))
    )
    assert wrong.criterion_scores[0].raw_score == 1.5
    assert wrong.base_quality == pytest.approx(0.625)
    assert wrong.normalized_score == 0.5
    assert [effect.rule_id for effect in wrong.triggered_deductions] == ["wrong"]

    dangerous = score_assessment(
        rubric(), spec(), assessment(finding_statuses=("absent", "present"))
    )
    assert dangerous.base_quality == 1.0
    assert dangerous.normalized_score == 0.2
    assert dangerous.severe_or_critical_event_ids == ("danger",)


def test_corrections_are_unique_descriptive_and_quality_independent():
    correction = CorrectionAssessment("correction-1", "Fixed unit", evidence("unit"))
    plain = score_assessment(rubric(), spec(), assessment())
    corrected = score_assessment(
        rubric(), spec(), assessment(corrections=(correction,))
    )
    assert corrected.correction_count == 1
    assert corrected.normalized_score == plain.normalized_score
    with pytest.raises(ValueError, match="correction IDs must be unique"):
        assessment(corrections=(correction, correction))


def test_exact_correspondence_coverage_and_assessment_rule_sets():
    with pytest.raises(ValueError, match="criterion IDs/order"):
        score_assessment(
            rubric(), spec(expected_criterion_ids=("safety", "accuracy")), assessment()
        )
    with pytest.raises(ValueError, match="positive required step"):
        score_assessment(
            rubric(),
            spec(required_steps=spec().required_steps[:2]),
            assessment(steps=assessment().step_assessments[:2]),
        )

    valid = assessment()
    for field, value, match in (
        ("step_assessments", valid.step_assessments[:-1], "step assessment IDs"),
        (
            "step_assessments",
            valid.step_assessments + (valid.step_assessments[0],),
            "step assessment IDs",
        ),
        (
            "step_assessments",
            valid.step_assessments[:-1] + (StepAssessment("unknown", "not_met", {}),),
            "step assessment IDs",
        ),
        (
            "finding_assessments",
            valid.finding_assessments[:-1],
            "finding assessment IDs",
        ),
    ):
        payload = valid.to_dict()
        payload[field] = [item.to_dict() for item in value]
        bad = BlindedAssessment.from_dict(payload)
        with pytest.raises(ValueError, match=match):
            score_assessment(rubric(), spec(), bad)


def test_unresolved_needs_adjudication_and_interprets_no_rules():
    result = score_assessment(
        rubric(),
        spec(),
        assessment(
            step_statuses=("unresolved", "met", "met"),
            finding_statuses=("absent", "unresolved"),
        ),
    )
    assert result.status == "needs_adjudication"
    assert result.criterion_scores == ()
    assert result.criterion_arithmetic == ()
    assert result.raw_score is result.max_score is result.normalized_score is None
    assert result.base_quality is None
    assert result.triggered_deductions == result.triggered_caps == ()
    assert result.unresolved_rule_ids == ("explain", "danger")


def test_permutation_invariance_canonical_hashes_and_timestamp_identity():
    canonical = assessment(
        corrections=(
            CorrectionAssessment("b", "B", evidence("b")),
            CorrectionAssessment("a", "A", evidence("a")),
        )
    )
    permuted = assessment(
        steps=tuple(reversed(canonical.step_assessments)),
        findings=tuple(reversed(canonical.finding_assessments)),
        corrections=tuple(reversed(canonical.corrections)),
    )
    first = score_assessment(rubric(), spec(), canonical)
    second = score_assessment(rubric(), spec(), permuted)
    assert first == second
    assert first.assessment_hash == second.assessment_hash
    assert (
        len(first.spec_hash)
        == len(first.assessment_hash)
        == len(first.scorer_hash)
        == 64
    )
    assert score_assessment(rubric(), spec(), canonical) == first

    later = score_assessment(
        rubric(), spec(), assessment(timestamp="2026-07-29T10:00:01Z")
    )
    assert later.assessment_hash != first.assessment_hash
    assert later.scorer_hash == first.scorer_hash


def test_recursive_immutability_and_tampered_records_are_rejected():
    item = assessment().step_assessments[0]
    with pytest.raises(TypeError):
        item.supporting_evidence["quote"] = "changed"
    with pytest.raises(TypeError):
        item.supporting_evidence["locations"][0]["start"] = 4
    with pytest.raises(FrozenInstanceError):
        item.status = "not_met"

    tampered = assessment()
    object.__setattr__(tampered.step_assessments[0], "status", "invented")
    with pytest.raises(ValueError, match="status"):
        score_assessment(rubric(), spec(), tampered)


def test_invalid_values_are_rejected():
    for value in (True, 0, -1):
        with pytest.raises(ValueError, match="decimal_precision"):
            spec(decimal_precision=value)
    for value in ("01", "1.0", "1e2", "NaN", "Infinity", "-1", True):
        with pytest.raises(ValueError, match="positive_points"):
            RequiredStepSpec("s", "accuracy", value, False, None)
    for value in ("-1", "0.0", "NaN", False):
        with pytest.raises(ValueError, match="deduction"):
            NegativeFindingSpec("f", "false_claim", "accuracy", value, "minor")
    with pytest.raises(ValueError, match="quality_cap"):
        RequiredStepSpec("s", "accuracy", "1", False, "1.1")
    with pytest.raises(ValueError, match="kind"):
        NegativeFindingSpec("f", "regex_judge", "accuracy", "0", "minor")
    with pytest.raises(ValueError, match="severity"):
        NegativeFindingSpec("f", "false_claim", "accuracy", "0", "unsafe")
    with pytest.raises(ValueError, match="status"):
        StepAssessment("s", "yes", evidence())
    with pytest.raises(ValueError, match="supporting_evidence"):
        StepAssessment("s", "met", {})
    with pytest.raises(ValueError, match="supporting_evidence"):
        FindingAssessment("f", "present", {})
    with pytest.raises(ValueError, match="assessment_timestamp"):
        assessment(timestamp="2026-07-29 10:00:00+00:00")
    with pytest.raises(ValueError, match="unsupported JSON"):
        StepAssessment("s", "met", {"bad": object()})


def test_api_is_structurally_blinded_and_poor_assessment_stays_poor():
    parameters = set(signature(score_assessment).parameters)
    assert parameters == {"rubric", "spec", "assessment"}
    forbidden = {
        "selection_decision",
        "selector_mode",
        "selected_ids",
        "retrieval_precision",
        "ndcg",
        "tokens",
        "latency",
        "condition",
        "context_labels",
    }
    annotations = repr(signature(score_assessment))
    assert not any(name in annotations.casefold() for name in forbidden)

    poor = score_assessment(
        rubric(),
        spec(),
        assessment(
            step_statuses=("contradicted", "not_met", "contradicted"),
            finding_statuses=("present", "present"),
        ),
    )
    assert poor.normalized_score == 0.0
    assert poor.severe_or_critical_event_ids == ("danger",)
    with pytest.raises(TypeError):
        score_assessment(rubric(), spec(), assessment(), retrieval_precision=1.0)


def test_artifact_round_trips_and_contains_complete_arithmetic():
    result = score_assessment(
        rubric(), spec(), assessment(finding_statuses=("present", "absent"))
    )
    payload = result.to_dict()
    assert payload["criterion_arithmetic"][0] == {
        "criterion_id": "accuracy",
        "met_step_ids": ["explain", "verify"],
        "no_credit_step_ids": [],
        "present_finding_ids": ["wrong"],
        "max_points": "3",
        "met_points": "3",
        "deduction_points": "1.5",
        "raw_points": "1.5",
        "normalized": "0.5",
    }
    assert payload["engine_version"] == "deterministic-blinded-v1"
    assert type(result).from_dict(payload) == result
