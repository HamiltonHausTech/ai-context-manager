import copy
from collections.abc import Sequence
from dataclasses import FrozenInstanceError, dataclass, replace
from inspect import signature

import pytest

from experiments.adaptive_selection.schema import RubricCriterion, ScoringRubric
from experiments.adaptive_selection.scoring import (
    BlindedAssessment,
    CorrectionAssessment,
    CriterionArithmetic,
    EvidenceSpan,
    FindingAssessment,
    NegativeFindingSpec,
    RequiredStepSpec,
    ScoringResult,
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
    return (EvidenceSpan(0, len(text), text),)


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
                evidence(step_id) if status in {"met", "contradicted"} else (),
            )
            for step_id, status in zip(step_ids, step_statuses)
        )
    if findings is None:
        findings = tuple(
            FindingAssessment(
                finding_id,
                status,
                evidence(finding_id) if status == "present" else (),
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
            valid.step_assessments[:-1] + (StepAssessment("unknown", "not_met", ()),),
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
    assert result.rubric == rubric()
    assert result.spec == spec()
    assert result.assessment.step_assessments[0].status == "unresolved"
    assert result.raw_score_decimal is result.normalized_score_decimal is None


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
    assert isinstance(item.supporting_evidence, tuple)
    with pytest.raises(FrozenInstanceError):
        item.supporting_evidence[0].quote = "changed"
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
        StepAssessment("s", "met", ())
    with pytest.raises(ValueError, match="supporting_evidence"):
        FindingAssessment("f", "present", ())
    with pytest.raises(ValueError, match="assessment_timestamp"):
        assessment(timestamp="2026-07-29 10:00:00+00:00")
    with pytest.raises(ValueError, match="EvidenceSpan"):
        StepAssessment(
            "s", "met", ({"start_offset": 0, "end_offset": 1, "quote": "x"},)
        )


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


@pytest.mark.parametrize(
    "smuggled",
    [
        {"mode": "adaptive"},
        {"arm": "treatment"},
        {"policy": "secret"},
        {"selected_context": ["ctx-1"]},
        {"selectionPolicy": "top-k"},
        {"nested": {"selector_mode": "adaptive"}},
    ],
)
def test_evidence_span_structurally_rejects_metadata_smuggling(smuggled):
    payload = {"start_offset": 0, "end_offset": 1, "quote": "x"}
    payload.update(smuggled)
    with pytest.raises(ValueError, match="invalid EvidenceSpan payload"):
        EvidenceSpan.from_dict(payload)
    with pytest.raises(ValueError, match="EvidenceSpan"):
        StepAssessment("s", "met", (payload,))


def test_evidence_span_subclass_metadata_smuggling_is_sealed():
    def plain():
        class Forged(EvidenceSpan):
            pass

    def data():
        @dataclass(frozen=True)
        class Forged(EvidenceSpan):
            selector_mode: str = "adaptive"

    def custom():
        class Forged(EvidenceSpan):
            def __init__(self, selector_mode):
                super().__init__(0, 1, "x")
                object.__setattr__(self, "selector_mode", selector_mode)

    for define in (plain, data, custom):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            define()

    original_hook = EvidenceSpan.__dict__["__init_subclass__"]
    try:
        setattr(
            EvidenceSpan,
            "__init_subclass__",
            classmethod(lambda cls, **kwargs: None),
        )

        class LegacyForged(EvidenceSpan):
            pass

    finally:
        setattr(EvidenceSpan, "__init_subclass__", original_hook)
    forged = LegacyForged(0, 1, "x")
    object.__setattr__(forged, "selector_mode", "adaptive")
    with pytest.raises(ValueError, match="exact EvidenceSpan"):
        StepAssessment("s", "met", (forged,))


def test_public_tuple_fields_reject_iterators_without_iteration():
    def forbidden_iterable():
        raise AssertionError("public tuple field consumed an iterator")
        yield None

    factories = (
        lambda value: spec(expected_criterion_ids=value),
        lambda value: spec(required_steps=value),
        lambda value: spec(negative_findings=value),
        lambda value: StepAssessment("s", "met", value),
        lambda value: FindingAssessment("f", "present", value),
        lambda value: CorrectionAssessment("c", "fixed", value),
        lambda value: assessment(steps=value),
        lambda value: assessment(findings=value),
        lambda value: assessment(corrections=value),
        lambda value: CriterionArithmetic(
            "accuracy", value, (), (), "1", "0", "0", "0", "0"
        ),
        lambda value: CriterionArithmetic(
            "accuracy", (), value, (), "1", "0", "0", "0", "0"
        ),
        lambda value: CriterionArithmetic(
            "accuracy", (), (), value, "1", "0", "0", "0", "0"
        ),
    )
    for factory in factories:
        with pytest.raises(ValueError, match="tuple or list"):
            factory(forbidden_iterable())

    class ArbitrarySequence(Sequence):
        def __len__(self):
            raise AssertionError("arbitrary Sequence length was inspected")

        def __getitem__(self, index):
            raise AssertionError("arbitrary Sequence was consumed")

    for invalid in ("scalar", ArbitrarySequence()):
        with pytest.raises(ValueError, match="tuple or list"):
            StepAssessment("s", "met", invalid)


@pytest.mark.parametrize(
    "factory, oversized, field",
    [
        (
            lambda value: spec(expected_criterion_ids=value),
            [object()] * 65,
            "expected_criterion_ids",
        ),
        (lambda value: spec(required_steps=value), [object()] * 129, "required_steps"),
        (
            lambda value: spec(negative_findings=value),
            [object()] * 129,
            "negative_findings",
        ),
        (
            lambda value: StepAssessment("s", "met", value),
            [object()] * 257,
            "supporting_evidence",
        ),
        (lambda value: assessment(steps=value), [object()] * 129, "step_assessments"),
        (
            lambda value: assessment(findings=value),
            [object()] * 129,
            "finding_assessments",
        ),
        (lambda value: assessment(corrections=value), [object()] * 257, "corrections"),
        (
            lambda value: CriterionArithmetic(
                "accuracy", value, (), (), "1", "0", "0", "0", "0"
            ),
            [object()] * 129,
            "met_step_ids",
        ),
    ],
)
def test_public_tuple_field_caps_precede_item_validation(factory, oversized, field):
    with pytest.raises(ValueError, match="{}.*at most".format(field)):
        factory(oversized)


def test_public_tuple_fields_accept_exact_lists_and_tuples():
    span = EvidenceSpan(0, 1, "x")
    assert StepAssessment("s", "met", [span]).supporting_evidence == (span,)
    assert FindingAssessment("f", "present", [span]).supporting_evidence == (span,)
    assert CorrectionAssessment("c", "fixed", [span]).supporting_evidence == (span,)
    assert TaskScoringSpec.from_dict(spec().to_dict()) == spec()
    assert BlindedAssessment.from_dict(assessment().to_dict()) == assessment()
    arithmetic = CriterionArithmetic(
        "accuracy", ["met"], ("missed",), [], "1", "1", "0", "1", "1"
    )
    assert arithmetic.met_step_ids == ("met",)
    assert arithmetic.no_credit_step_ids == ("missed",)
    assert arithmetic.present_finding_ids == ()


def test_evidence_span_bounds_and_status_evidence_exclusivity():
    for values in ((-1, 1, "x"), (0, 0, "x"), (2, 1, "x"), (0, 1, "")):
        with pytest.raises(ValueError):
            EvidenceSpan(*values)
    for status in ("not_met", "unresolved"):
        with pytest.raises(ValueError, match="must be empty"):
            StepAssessment("s", status, evidence())
    for status in ("absent", "unresolved"):
        with pytest.raises(ValueError, match="must be empty"):
            FindingAssessment("f", status, evidence())
    with pytest.raises(ValueError, match="nonempty"):
        CorrectionAssessment("c", "description", ())


def test_result_is_complete_canonical_and_recursively_immutable():
    source = assessment(
        steps=tuple(reversed(assessment().step_assessments)),
        findings=tuple(reversed(assessment().finding_assessments)),
        corrections=(
            CorrectionAssessment("z", "last", evidence("last")),
            CorrectionAssessment("a", "first", evidence("first")),
        ),
    )
    result = score_assessment(rubric(), spec(), source)
    payload = result.to_dict()
    assert result.rubric == rubric()
    assert result.spec == spec()
    assert [item.step_id for item in result.assessment.step_assessments] == [
        "explain",
        "verify",
        "safe",
    ]
    assert [item.finding_id for item in result.assessment.finding_assessments] == [
        "wrong",
        "danger",
    ]
    assert [item.correction_id for item in result.assessment.corrections] == ["a", "z"]
    assert payload["rubric"]["instructions"] == "Score only the frozen assessment."
    assert payload["rubric"]["criteria"][0]["description"] == "Technically correct"
    assert payload["spec"]["required_steps"][0]["positive_points"] == "2"
    assert payload["assessment"]["rater_id"] == "rater-2"
    assert payload["assessment"]["step_assessments"][0]["supporting_evidence"] == [
        {"start_offset": 0, "end_offset": 7, "quote": "explain"}
    ]
    assert result.raw_score_decimal == "4"
    assert result.max_score_decimal == "4"
    assert result.normalized_score_decimal == "1"
    assert result.base_quality_decimal == "1"
    with pytest.raises(FrozenInstanceError):
        result.assessment.step_assessments[0].supporting_evidence[0].quote = "forged"


def test_result_construction_and_subclass_forgery_are_sealed():
    result = score_assessment(rubric(), spec(), assessment())
    with pytest.raises(TypeError, match="score_assessment"):
        ScoringResult()
    with pytest.raises(TypeError, match="score_assessment"):
        ScoringResult(**result.to_dict())
    with pytest.raises(TypeError, match="score_assessment"):
        replace(result, normalized_score=0.0)

    def plain():
        class Forged(ScoringResult):
            pass

    def data():
        @dataclass(frozen=True)
        class Forged(ScoringResult):
            pass

    def custom():
        class Forged(ScoringResult):
            def __init__(self):
                pass

    for define in (plain, data, custom):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            define()
    assert copy.copy(result) == result
    assert copy.deepcopy(result) == result


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("rubric_hash", "0" * 64),
        lambda payload: payload.__setitem__("normalized_score", 0.1),
        lambda payload: payload.__setitem__("normalized_score_decimal", "0.1"),
        lambda payload: payload["criterion_scores"][0].__setitem__("raw_score", 0.0),
        lambda payload: payload["criterion_arithmetic"][0].__setitem__(
            "raw_points", "0"
        ),
        lambda payload: payload["triggered_caps"].append(
            {
                "rule_id": "explain",
                "criterion_id": "accuracy",
                "effect": "quality_cap",
                "value": "0.6",
                "reason": "forged",
            }
        ),
        lambda payload: payload["severe_or_critical_event_ids"].append("forged"),
        lambda payload: payload["critical_omission_ids"].append("forged"),
        lambda payload: payload["rubric"].__setitem__("instructions", "changed"),
        lambda payload: payload["spec"].__setitem__("provenance", "changed"),
        lambda payload: payload["assessment"].__setitem__("rater_id", "forged"),
    ],
)
def test_from_dict_rejects_any_noncanonical_or_underived_payload(mutate):
    payload = score_assessment(
        rubric(), spec(), assessment(step_statuses=("contradicted", "met", "met"))
    ).to_dict()
    mutate(payload)
    with pytest.raises(ValueError, match="canonical derived scoring result"):
        ScoringResult.from_dict(payload)


def test_global_rule_ids_and_all_not_met_caps_are_enforced():
    with pytest.raises(ValueError, match="rule IDs must be globally unique"):
        spec(
            negative_findings=(
                NegativeFindingSpec("explain", "false_claim", "accuracy", "1", "minor"),
            )
        )
    uncoupled = spec(
        required_steps=(
            RequiredStepSpec("explain", "accuracy", "2", False, "0.6"),
            *spec().required_steps[1:],
        )
    )
    result = score_assessment(
        rubric(), uncoupled, assessment(step_statuses=("not_met", "met", "met"))
    )
    assert [effect.rule_id for effect in result.triggered_caps] == ["explain"]
    assert result.critical_omission_ids == ()


def test_rubric_hash_binds_all_complete_rubric_content():
    baseline = score_assessment(rubric(), spec(), assessment())
    for changed in (
        ScoringRubric(
            "rubric-1", "Different instructions", rubric().criteria, "fixture:rubric"
        ),
        ScoringRubric(
            "rubric-1",
            rubric().instructions,
            (
                RubricCriterion("accuracy", "Different description", 3.0),
                rubric().criteria[1],
            ),
            "fixture:rubric",
        ),
        ScoringRubric(
            "rubric-1", rubric().instructions, rubric().criteria, "different:provenance"
        ),
    ):
        result = score_assessment(changed, spec(), assessment())
        assert result.rubric_hash != baseline.rubric_hash
        assert result.scorer_hash != baseline.scorer_hash


def test_deterministic_decimal_resource_bounds_reject_before_arithmetic():
    for precision in (15, 65, 10**9):
        with pytest.raises(ValueError, match="between 16 and 64"):
            spec(decimal_precision=precision)
    for value in ("1" * 65, "1000000000001", "0." + "1" * 65):
        with pytest.raises(ValueError, match="positive_points"):
            RequiredStepSpec("bounded", "accuracy", value, False)
    with pytest.raises(ValueError, match="at most"):
        spec(required_steps=spec().required_steps * 100)
    huge_weight = ScoringRubric(
        "rubric-1",
        rubric().instructions,
        (
            RubricCriterion("accuracy", "Technically correct", 1e308),
            RubricCriterion("safety", "Safe", 1e308),
        ),
        rubric().provenance,
    )
    with pytest.raises(ValueError, match="rubric weight"):
        score_assessment(huge_weight, spec(), assessment())

    repeating = spec(
        decimal_precision=64,
        required_steps=(
            RequiredStepSpec("one", "accuracy", "1", False),
            RequiredStepSpec("two", "accuracy", "2", False),
            RequiredStepSpec("safe", "safety", "1", False),
        ),
        negative_findings=(),
    )
    repeating_assessment = assessment(
        steps=(
            StepAssessment("one", "met", evidence("one")),
            StepAssessment("two", "not_met", ()),
            StepAssessment("safe", "met", evidence("safe")),
        ),
        findings=(),
    )
    result = score_assessment(rubric(), repeating, repeating_assessment)
    assert len(result.criterion_arithmetic[0].normalized) == 66
    assert result.normalized_score_decimal is not None
