import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from decimal import localcontext
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import cast

import pytest

import experiments.adaptive_selection.report as report_module
from experiments.adaptive_selection.dataset import load_tiny_fixture
from experiments.adaptive_selection.learning import LearningPolicy
from experiments.adaptive_selection.providers import (
    TOKEN_ACCOUNTING_VERSION,
    DeterministicFakeProvider,
    ProviderConfiguration,
    RawTransportResult,
)
from experiments.adaptive_selection.report import (
    INTERVAL_METHOD_VERSION,
    REPORT_VERSION,
    ExperimentReport,
    IntervalSpec,
    PriceRate,
    PricingSpec,
    ReportingSpec,
    build_experiment_report,
)
from experiments.adaptive_selection.runner import (
    ArmRuntime,
    ArmSpec,
    CanonicalPromptRenderer,
    ExperimentPlan,
    RepetitionSpec,
    RunnerClocks,
    Stage0OrderedDatasetSource,
    run_ordered_experiment,
)
from experiments.adaptive_selection.scoring import (
    BlindedAssessment,
    EvidenceSpan,
    FindingAssessment,
    NegativeFindingSpec,
    RequiredStepSpec,
    StepAssessment,
    TaskScoringSpec,
)
from experiments.adaptive_selection.selectors import (
    AdaptivePolicySelector,
    FullContextSelector,
    SimilarityTopKSelector,
    StaticPolicySelector,
)

UTC = "2026-07-29T12:00:00Z"


def _domain_hash(domain, payload):
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return (
        "sha256:"
        + hashlib.sha256(domain.encode("utf-8") + b"\0" + canonical).hexdigest()
    )


def _rehash_record(record, domain):
    unsigned = dict(record)
    unsigned.pop("record_hash")
    record["record_hash"] = _domain_hash(domain, unsigned)


def _rehash_report(payload):
    unsigned = dict(payload)
    unsigned.pop("report_hash")
    payload["report_hash"] = _domain_hash("adaptive-experiment-report-v1", unsigned)


def _artifact(
    response_text="answer",
    step_status="met",
    finding_severities=(),
    finding_status="absent",
    first_arm_classification="reference",
    repetition_count=2,
    bundle_transform=None,
):
    bundle = load_tiny_fixture(
        Path(__file__).parent / "fixtures" / "tiny_experiment.json"
    )
    if bundle_transform is not None:
        bundle = bundle_transform(bundle)
    specs = {}
    for case in bundle.cases:
        rubric = case.sealed_evaluation.scoring_rubric
        first_criterion_id = rubric.criteria[0].criterion_id
        specs[case.task_case_id] = TaskScoringSpec(
            "spec-" + case.task_case_id,
            "1",
            rubric.rubric_id,
            tuple(item.criterion_id for item in rubric.criteria),
            tuple(
                RequiredStepSpec(
                    "step-{}-{}".format(case.task_case_id, item.criterion_id),
                    item.criterion_id,
                    "1",
                    False,
                    None,
                )
                for item in rubric.criteria
            ),
            tuple(
                NegativeFindingSpec(
                    "finding-{}-{}".format(case.task_case_id, severity),
                    "false_claim",
                    first_criterion_id,
                    "0.1",
                    severity,
                    None,
                )
                for severity in finding_severities
            ),
            "fixture_only",
            "deterministic-v1",
            "weighted-v1",
            "rules-v1",
            28,
            "decimal-v1",
            "test:task10",
        )
    renderer = CanonicalPromptRenderer("Answer from supplied context only.")
    fixtures = {}
    for case in bundle.cases:
        candidates = case.inputs.candidate_context
        for count in range(len(candidates) + 1):
            for selected in permutations(candidates, count):
                if (
                    sum(item.token_count for item in selected)
                    > case.inputs.token_budget
                ):
                    continue
                request = renderer.render(case.inputs, selected)
                fixtures[request.request_hash] = RawTransportResult(
                    "recorded",
                    "model",
                    "rev",
                    response_text,
                    response_text.encode("utf-8") or b"empty-response",
                    10,
                    int(bool(response_text)),
                    "req",
                )
    arms = (
        ArmSpec(
            "full",
            "full_context",
            "1",
            "sha256:" + "1" * 64,
            False,
            first_arm_classification,
        ),
        ArmSpec(
            "topk", "similarity_top_k", "1", "sha256:" + "2" * 64, False, "secondary"
        ),
        ArmSpec(
            "static",
            "static_policy",
            "1",
            "sha256:" + "3" * 64,
            False,
            "primary_baseline",
        ),
        ArmSpec(
            "adaptive", "adaptive_policy", "1", "sha256:" + "4" * 64, True, "candidate"
        ),
    )

    class Assessor:
        def assess(self, packet):
            return BlindedAssessment(
                packet.output_id,
                packet.scoring_spec.rubric_id,
                packet.scoring_spec.spec_id,
                packet.scoring_spec.spec_version,
                tuple(
                    StepAssessment(
                        step.step_id,
                        step_status,
                        (
                            (EvidenceSpan(0, 6, "answer"),)
                            if step_status == "met"
                            else ()
                        ),
                    )
                    for step in packet.scoring_spec.required_steps
                ),
                tuple(
                    FindingAssessment(
                        item.finding_id,
                        finding_status,
                        (
                            (EvidenceSpan(0, 6, "answer"),)
                            if finding_status == "present"
                            else ()
                        ),
                    )
                    for item in packet.scoring_spec.negative_findings
                ),
                (),
                "fixture-rater",
                "1",
                UTC,
                "test:task10",
            )

    source = Stage0OrderedDatasetSource(bundle, specs)
    plan = ExperimentPlan(
        "ordered-v1",
        "experiment-v1",
        "protocol-v1",
        source.dataset_version,
        source.dataset_hash,
        "abc123",
        renderer.template_spec,
        renderer.template_hash,
        LearningPolicy(),
        99,
        source.family_order,
        arms,
        tuple(
            RepetitionSpec(index, 10 + index, None) for index in range(repetition_count)
        ),
        "test:task10",
    )
    runtimes = (
        ArmRuntime(arms[0], lambda _: FullContextSelector()),
        ArmRuntime(arms[1], lambda _: SimilarityTopKSelector(k=2)),
        ArmRuntime(arms[2], lambda _: StaticPolicySelector()),
        ArmRuntime(arms[3], lambda utilities: AdaptivePolicySelector(utilities)),
    )

    def provider_factory(repetition):
        config = ProviderConfiguration(
            "recorded",
            "model",
            "rev",
            0.0,
            repetition.provider_seed,
            True,
            (),
            TOKEN_ACCOUNTING_VERSION,
            {},
        )
        return DeterministicFakeProvider(
            config, tuple(fixtures.items()), lambda: UTC, lambda: 1.0
        )

    return run_ordered_experiment(
        plan,
        source,
        runtimes,
        provider_factory,
        renderer,
        Assessor(),
        RunnerClocks(
            lambda: UTC, lambda: 1.0, lambda: UTC, lambda value: "blind-" + value[-16:]
        ),
    )


def _spec(pricing=None, threshold=None, draw_count=101):
    return ReportingSpec(
        report_version=REPORT_VERSION,
        aggregation_version="family-balanced-repetition-v1",
        unscored_quality_value="0",
        pass_threshold=threshold,
        interval=IntervalSpec(INTERVAL_METHOD_VERSION, "0.95", draw_count, 7),
        decimal_precision=28,
        decimal_version="decimal-half-even-v1",
        pricing=pricing,
        claim_scope="Tiny deterministic control evidence only; no efficacy claim.",
    )


def test_report_round_trip_metrics_pairing_and_phase_separation():
    artifact = _artifact()
    report = build_experiment_report(artifact, _spec())
    assert (
        report.canonical_bytes()
        == build_experiment_report(artifact, _spec()).canonical_bytes()
    )
    assert (
        ExperimentReport.from_dict(report.to_dict()).canonical_bytes()
        == report.canonical_bytes()
    )
    assert report.primary_baseline_arm_id == "static"
    assert report.adaptive_candidate_arm_id == "adaptive"
    assert (
        report.reporting_spec.primary_estimand
        == "heldout-evaluation-adaptive-minus-primary-baseline-v1"
    )
    assert report.id_local_outcome_ablation_available is False
    assert all(
        item.orientation == "adaptive_minus_baseline" for item in report.pair_effects
    )
    assert {item.phase for item in report.primary_summaries} == {"evaluation"}
    assert {item.phase for item in report.adaptation_summaries} == {"adaptation"}
    assert {item.phase for item in report.arm_summaries} == {"adaptation", "evaluation"}
    assert any(
        item.effect.numerator == 0
        for item in report.pair_effects
        if item.effect.available
    )
    costs = [item for item in report.pair_effects if item.metric == "estimated_cost"]
    assert costs and all(
        not item.effect.available and item.effect.reason == "pricing_not_supplied"
        for item in costs
    )
    assert report.intervals and all(
        item.evidence_label == "coarse_control_evidence"
        for item in report.intervals
        if item.interval_available
    )
    assert all(item.record_hash.startswith("sha256:") for item in report.pair_effects)
    assert report.learning_evidence
    assert any(item.estimates for item in report.learning_evidence)
    assert all(
        estimate.provenance and estimate.source_event_ids
        for item in report.learning_evidence
        for estimate in item.estimates
    )
    with pytest.raises(FrozenInstanceError):
        report.primary_baseline_arm_id = "forged"
    with pytest.raises(TypeError):
        ExperimentReport()


def test_exact_pricing_pass_threshold_and_zero_denominators():
    pricing = PricingSpec(
        "USD",
        "rates-2026-07",
        (PriceRate("recorded", "model", "rev", TOKEN_ACCOUNTING_VERSION, "2.5", "10"),),
    )
    report = build_experiment_report(_artifact(), _spec(pricing, "0.5"))
    cost_pairs = [
        item for item in report.pair_effects if item.metric == "estimated_cost"
    ]
    assert cost_pairs and all(item.effect.available for item in cost_pairs)
    pass_pairs = [item for item in report.pair_effects if item.metric == "pass"]
    assert pass_pairs and all(item.effect.available for item in pass_pairs)
    zero_baselines = [
        item
        for item in report.pair_effects
        if item.metric == "critical_scoring_failure"
        and not item.relative_improvement.available
    ]
    assert zero_baselines
    assert all(
        item.relative_improvement.reason == "baseline_zero"
        and item.relative_improvement.denominator == 0
        for item in zero_baselines
    )


def test_roles_labels_pricing_and_recomputed_forgery_are_rejected():
    artifact = _artifact()
    payload = artifact.to_dict()
    payload["plan"]["arms"][2]["classification"] = "secondary"
    with pytest.raises(ValueError):
        type(artifact).from_dict(payload)

    bad_pricing = PricingSpec(
        "USD",
        "rates",
        (PriceRate("other", "model", "rev", TOKEN_ACCOUNTING_VERSION, "1", "1"),),
    )
    with pytest.raises(ValueError, match="pricing"):
        build_experiment_report(artifact, _spec(bad_pricing))

    report = build_experiment_report(artifact, _spec())
    forged = json.loads(report.canonical_bytes().decode("utf-8"))
    forged["pair_effects"][0]["effect"]["numerator"] += 1
    _rehash_record(forged["pair_effects"][0]["effect"], "adaptive-metric-value-v1")
    _rehash_record(forged["pair_effects"][0], "adaptive-pair-effect-v1")
    _rehash_report(forged)
    with pytest.raises(ValueError, match="canonical derived report"):
        ExperimentReport.from_dict(forged)


def test_failed_but_scored_outcome_is_separate_from_itt_quality_and_pass():
    report = build_experiment_report(_artifact(response_text=""), _spec(threshold="0"))
    evaluation = [item for item in report.pair_effects if item.phase == "evaluation"]
    execution = [item for item in evaluation if item.metric == "execution_success"]
    scored = [item for item in evaluation if item.metric == "scored_outcome"]
    quality = [item for item in evaluation if item.metric == "task_quality"]
    passed = [item for item in evaluation if item.metric == "pass"]
    assert execution and all(item.baseline_value.numerator == 0 for item in execution)
    assert scored and all(item.baseline_value.numerator == 1 for item in scored)
    assert quality and all(item.baseline_value.numerator == 0 for item in quality)
    assert passed and all(item.baseline_value.numerator == 1 for item in passed)


def test_unresolved_scoring_and_critical_severe_findings_are_reported_exactly():
    unresolved = build_experiment_report(_artifact(step_status="unresolved"), _spec())
    unresolved_evaluation = [
        item for item in unresolved.pair_effects if item.phase == "evaluation"
    ]
    assert all(
        item.baseline_value.numerator == 0
        for item in unresolved_evaluation
        if item.metric in {"scored_outcome", "task_quality"}
    )

    findings = build_experiment_report(
        _artifact(finding_severities=("critical", "severe"), finding_status="present"),
        _spec(),
    )
    for metric in {"critical_scoring_failure", "severe_scoring_failure"}:
        values = [
            item
            for item in findings.pair_effects
            if item.phase == "evaluation" and item.metric == metric
        ]
        assert values and all(item.baseline_value.numerator == 1 for item in values)


def test_interval_is_clustered_coarse_deterministic_and_requires_repetitions():
    artifact = _artifact()
    first = build_experiment_report(artifact, _spec())
    second = build_experiment_report(artifact, _spec())
    assert [item.to_dict() for item in first.intervals] == [
        item.to_dict() for item in second.intervals
    ]
    available = [item for item in first.intervals if item.interval_available]
    assert available
    for item in available:
        assert item.repetition_count == 2
        assert item.evidence_label == "coarse_control_evidence"
        assert item.lower is not None and item.upper is not None
        assert item.lower.numerator is not None and item.lower.denominator is not None
        assert item.upper.numerator is not None and item.upper.denominator is not None
        assert Fraction(item.lower.numerator, item.lower.denominator) <= Fraction(
            item.upper.numerator, item.upper.denominator
        )
    quality_interval = next(
        item
        for item in available
        if item.family_id is None and item.metric == "task_quality"
    )
    assert quality_interval.lower is not None and quality_interval.upper is not None
    assert quality_interval.lower.decimal == "0"
    assert quality_interval.upper.decimal == "0"

    single = build_experiment_report(_artifact(repetition_count=1), _spec())
    assert all(
        not item.interval_available and item.reason == "fewer_than_two_repetitions"
        for item in single.intervals
    )


def test_decimal_context_resource_and_pricing_payload_bounds_are_enforced():
    artifact = _artifact()
    pricing = PricingSpec(
        "USD",
        "rates",
        (
            PriceRate(
                "recorded",
                "model",
                "rev",
                TOKEN_ACCOUNTING_VERSION,
                "2.123456789123456789",
                "9.987654321987654321",
            ),
        ),
    )
    with localcontext() as context:
        context.prec = 6
        low_precision = build_experiment_report(artifact, _spec(pricing))
    with localcontext() as context:
        context.prec = 64
        high_precision = build_experiment_report(artifact, _spec(pricing))
    assert low_precision.canonical_bytes() == high_precision.canonical_bytes()

    with pytest.raises(ValueError, match="bootstrap work bound"):
        build_experiment_report(artifact, _spec(draw_count=100_000))

    payload = pricing.to_dict()
    payload["rates"] = (item for item in payload["rates"])
    with pytest.raises(ValueError, match="bounded exact list/tuple"):
        PricingSpec.from_dict(payload)
    oversized = pricing.to_dict()
    oversized["rates"] = oversized["rates"] * 257
    with pytest.raises(ValueError, match="bounded exact list/tuple"):
        PricingSpec.from_dict(oversized)
    with pytest.raises(ValueError, match="bounded exact PriceRate"):
        PricingSpec("USD", "oversized", cast(tuple, [pricing.rates[0]] * 257))


def test_ambiguous_roles_and_estimand_drift_are_rejected():
    artifact = _artifact(first_arm_classification="primary_baseline")
    with pytest.raises(ValueError, match="exactly one primary_baseline"):
        build_experiment_report(artifact, _spec())
    with pytest.raises(ValueError, match="estimand"):
        replace(_spec(), primary_estimand="post-hoc-estimand")


def test_family_balancing_and_repetition_cluster_bootstrap_differ_from_case_weighting():
    spec = _spec()
    zero = report_module._available(Fraction(0), spec)
    one = report_module._available(Fraction(1), spec)
    balanced = report_module._family_balanced_mean(
        {"large-family": (zero, zero, zero), "small-family": (one,)}, spec
    )
    assert balanced.numerator == 1 and balanced.denominator == 2
    assert Fraction(balanced.numerator, balanced.denominator) != Fraction(1, 4)

    def trajectory(repetition, value):
        return report_module._derived(
            report_module.TrajectoryEffect,
            phase="evaluation",
            family_id=None,
            repetition_index=repetition,
            metric="task_quality",
            favorable_direction="higher",
            nested_case_count=1,
            effect=value,
        )

    clustered = (trajectory(0, zero), trajectory(1, one))
    clustered_summary = report_module._metric_summary(
        "primary_paired",
        "evaluation",
        None,
        None,
        "task_quality",
        [item.effect for item in clustered],
        spec,
    )
    clustered_interval = report_module._interval(clustered_summary, clustered, spec)
    pseudo_cases = tuple(
        trajectory(index, value) for index, value in enumerate((zero, zero, zero, one))
    )
    pseudo_summary = report_module._metric_summary(
        "primary_paired",
        "evaluation",
        None,
        None,
        "task_quality",
        [item.effect for item in pseudo_cases],
        spec,
    )
    pseudo_interval = report_module._interval(pseudo_summary, pseudo_cases, spec)
    assert clustered_interval.repetition_count == 2
    assert pseudo_interval.repetition_count == 4
    assert clustered_interval.lower is not None and clustered_interval.upper is not None
    assert pseudo_interval.lower is not None and pseudo_interval.upper is not None
    assert (
        clustered_interval.lower.to_dict(),
        clustered_interval.upper.to_dict(),
    ) != (pseudo_interval.lower.to_dict(), pseudo_interval.upper.to_dict())


def test_nonexhaustive_labels_and_structurally_invalid_pairs_are_rejected():
    def omit_one_label(bundle):
        case = bundle.cases[0]
        sealed = case.sealed_evaluation
        assert sealed.irrelevant_context_item_ids
        incomplete = replace(
            sealed,
            irrelevant_context_item_ids=sealed.irrelevant_context_item_ids[:-1],
        )
        return replace(
            bundle,
            cases=(replace(case, sealed_evaluation=incomplete),) + bundle.cases[1:],
        )

    with pytest.raises(ValueError, match="exhaustive"):
        build_experiment_report(_artifact(bundle_transform=omit_one_label), _spec())

    original = _artifact()
    for mutation in (
        "missing",
        "duplicate",
        "cross_phase",
        "cross_family",
        "cross_repetition",
    ):
        artifact = type(original).from_dict(original.to_dict())
        run = next(
            item
            for item in artifact.arm_runs
            if item.arm_id == "static" and item.repetition_index == 0
        )
        if mutation == "missing":
            object.__setattr__(run, "task_records", run.task_records[:-1])
        elif mutation == "duplicate":
            object.__setattr__(
                run, "task_records", run.task_records + (run.task_records[-1],)
            )
        elif mutation == "cross_phase":
            object.__setattr__(run.task_records[0], "phase", "evaluation")
        elif mutation == "cross_family":
            object.__setattr__(run.task_records[0], "family_id", "wrong-family")
        else:
            object.__setattr__(run, "repetition_index", 1)
        with pytest.raises(ValueError):
            build_experiment_report(artifact, _spec())


def test_adverse_effects_and_each_derived_report_surface_survive_or_reject_forgery():
    report = build_experiment_report(_artifact(), _spec())
    assert report.contains_adverse_or_null_primary_effects is True
    adverse = [
        item
        for item in report.primary_summaries
        if item.family_id is None
        and item.mean.available
        and item.mean.numerator is not None
        and (
            (item.favorable_direction == "higher" and item.mean.numerator < 0)
            or (item.favorable_direction == "lower" and item.mean.numerator > 0)
        )
    ]
    assert adverse

    mutations = (
        ("arm_summaries", 0, "observation_count", "adaptive-metric-summary-v1"),
        (
            "adaptation_summaries",
            0,
            "observation_count",
            "adaptive-metric-summary-v1",
        ),
        (
            "primary_summaries",
            0,
            "observation_count",
            "adaptive-metric-summary-v1",
        ),
        ("intervals", 0, "draw_count", "adaptive-interval-estimate-v1"),
    )
    for collection, index, field, domain in mutations:
        forged = json.loads(report.canonical_bytes().decode("utf-8"))
        forged[collection][index][field] += 1
        _rehash_record(forged[collection][index], domain)
        _rehash_report(forged)
        with pytest.raises(ValueError, match="canonical derived report"):
            ExperimentReport.from_dict(forged)

    forged_flag = json.loads(report.canonical_bytes().decode("utf-8"))
    forged_flag["contains_adverse_or_null_primary_effects"] = False
    _rehash_report(forged_flag)
    with pytest.raises(ValueError, match="canonical derived report"):
        ExperimentReport.from_dict(forged_flag)
