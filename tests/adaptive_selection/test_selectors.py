import inspect
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.adaptive_selection.dataset import load_tiny_fixture
from experiments.adaptive_selection.schema import ContextItem, TaskInputs
from experiments.adaptive_selection.selectors import (
    AdaptivePolicySelector,
    FullContextSelector,
    SelectionResult,
    SimilarityTopKSelector,
    StaticPolicySelector,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_experiment.json"
SELECTOR_TYPES = (
    FullContextSelector,
    SimilarityTopKSelector,
    StaticPolicySelector,
    AdaptivePolicySelector,
)


def _selectors():
    return (
        FullContextSelector(),
        SimilarityTopKSelector(k=3),
        StaticPolicySelector(),
        AdaptivePolicySelector(),
    )


def _assert_complete_exact_result(result, inputs):
    assert isinstance(result, SelectionResult)
    assert result.token_budget == inputs.token_budget
    assert result.used_tokens == sum(item.token_count for item in result.selected_items)
    assert result.used_tokens <= inputs.token_budget
    assert result.eligible_context_item_ids == tuple(
        item.context_item_id for item in inputs.candidate_context
    )
    assert len(result.decisions) == len(inputs.candidate_context)
    assert tuple(decision.context_item_id for decision in result.decisions) == tuple(
        item.context_item_id for item in inputs.candidate_context
    )
    assert len({decision.context_item_id for decision in result.decisions}) == len(
        inputs.candidate_context
    )
    assert {
        decision.context_item_id for decision in result.decisions if decision.included
    } == {item.context_item_id for item in result.selected_items}
    assert all(
        decision.score_factors["selector_mode"] == result.selector_mode
        for decision in result.decisions
    )


def test_all_selectors_cover_every_tiny_case_with_same_pool_exact_budget_and_trace():
    bundle = load_tiny_fixture(FIXTURE)
    for case in bundle.cases:
        results = tuple(selector.select(case.inputs) for selector in _selectors())
        for result in results:
            _assert_complete_exact_result(result, case.inputs)
        assert len({result.eligible_context_item_ids for result in results}) == 1


def test_authoritative_counts_not_core_estimates_drive_budget_packing():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    candidates = inputs.candidate_context[:2]
    authoritative = tuple(replace(item, token_count=1) for item in candidates)
    tiny_budget = replace(inputs, candidate_context=authoritative, token_budget=1)

    result = FullContextSelector().select(tiny_budget)

    assert result.used_tokens == 1
    assert len(result.selected_items) == 1
    assert [decision.reason for decision in result.decisions] == [
        "selected",
        "budget_exclusion",
    ]


def test_full_context_uses_candidate_order_greedy_fit_and_continues_after_oversize():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    candidates = (
        replace(inputs.candidate_context[0], token_count=9),
        replace(inputs.candidate_context[1], token_count=2),
        replace(inputs.candidate_context[2], token_count=3),
    )
    constrained = replace(inputs, candidate_context=candidates, token_budget=5)

    result = FullContextSelector().select(constrained)

    assert tuple(item.context_item_id for item in result.selected_items) == tuple(
        item.context_item_id for item in candidates[1:]
    )
    assert tuple(decision.reason for decision in result.decisions) == (
        "budget_exclusion",
        "selected",
        "selected",
    )


def test_similarity_top_k_is_strict_and_reports_k_separately_from_budget():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    result = SimilarityTopKSelector(k=2).select(inputs)

    assert (
        sum(decision.reason == "k_exclusion" for decision in result.decisions)
        == len(inputs.candidate_context) - 2
    )
    assert (
        sum(
            decision.reason in {"selected", "budget_exclusion"}
            for decision in result.decisions
        )
        == 2
    )
    assert all("relevance" in decision.score_factors for decision in result.decisions)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "2"])
def test_similarity_top_k_rejects_invalid_k(value):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        SimilarityTopKSelector(k=value)


def test_stable_candidate_order_is_final_tie_breaker():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    tied = replace(inputs, task_prompt="terms absent everywhere", token_budget=10_000)
    expected = tuple(item.context_item_id for item in tied.candidate_context[:3])

    similarity = SimilarityTopKSelector(k=3).select(tied)
    static = StaticPolicySelector(
        feature_weights={}, relevance_weight=1.0, importance_weight=0.0
    ).select(tied)

    assert tuple(item.context_item_id for item in similarity.selected_items) == expected
    assert tuple(item.context_item_id for item in static.selected_items[:3]) == expected


def test_adaptive_without_utility_is_exactly_static():
    inputs = load_tiny_fixture(FIXTURE).cases[1].inputs
    static = StaticPolicySelector().select(inputs)

    assert (
        AdaptivePolicySelector().select(inputs).policy_signature
        == static.policy_signature
    )
    assert (
        AdaptivePolicySelector(utility_estimates={}).select(inputs).policy_signature
        == static.policy_signature
    )
    assert tuple(
        item.context_item_id
        for item in AdaptivePolicySelector().select(inputs).selected_items
    ) == tuple(item.context_item_id for item in static.selected_items)


def test_approved_reusable_feature_utility_changes_ranking_without_item_id_feature():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    target = inputs.candidate_context[-1]
    feature = target.metadata["learning_attributes"][0]
    selector = AdaptivePolicySelector(
        utility_estimates={feature: 10.0},
        learning_weight=1.0,
        feature_weights={},
        relevance_weight=0.0,
        importance_weight=1.0,
    )

    result = selector.select(replace(inputs, token_budget=target.token_count))

    assert result.selected_items[0].source == target.source
    target_trace = next(
        decision
        for decision in result.decisions
        if decision.context_item_id == target.context_item_id
    )
    assert (
        target_trace.score_factors["policy.adaptive.feature_utility.{}".format(feature)]
        == 10.0
    )
    assert not any(target.context_item_id in key for key in target_trace.score_factors)


def test_mapping_and_callback_utilities_are_defensively_snapshotted():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    feature = inputs.candidate_context[0].metadata["learning_attributes"][0]
    estimates = {feature: 2.0}
    selector = AdaptivePolicySelector(utility_estimates=estimates)
    estimates[feature] = -100.0

    first = selector.select(inputs)
    second = AdaptivePolicySelector(
        utility_estimates=lambda name: 2.0 if name == feature else 0.0
    ).select(inputs)

    assert first == selector.select(inputs)
    assert any(
        decision.score_factors.get("policy.adaptive.feature_utility.{}".format(feature))
        == 2.0
        for decision in first.decisions
    )
    assert isinstance(second, SelectionResult)


def test_utility_processing_error_has_one_complete_nonleaking_trace_decision():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    failing_feature = inputs.candidate_context[0].metadata["learning_attributes"][0]

    def utility(feature):
        if feature == failing_feature:
            raise RuntimeError("private feedback backend detail")
        return 0.0

    result = AdaptivePolicySelector(utility_estimates=utility).select(inputs)

    _assert_complete_exact_result(result, inputs)
    failures = [
        decision
        for decision in result.decisions
        if decision.reason == "processing_error"
    ]
    assert failures
    assert all(decision.detail == "processing_error" for decision in failures)
    assert "private feedback backend detail" not in repr(result)


def test_results_and_nested_score_factors_are_immutable():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    result = StaticPolicySelector().select(inputs)

    with pytest.raises((AttributeError, TypeError)):
        result.selected_items += (inputs.candidate_context[0],)
    with pytest.raises(TypeError):
        result.decisions[0].score_factors["changed"] = True


def test_public_select_api_accepts_only_task_inputs_and_rejects_other_records():
    case = load_tiny_fixture(FIXTURE).cases[0]
    for selector_type in SELECTOR_TYPES:
        annotation = (
            inspect.signature(selector_type.select).parameters["inputs"].annotation
        )
        assert annotation in (TaskInputs, "TaskInputs")
        with pytest.raises(TypeError, match="TaskInputs"):
            selector_type().select(case)
        with pytest.raises(TypeError, match="TaskInputs"):
            selector_type().select(case.sealed_evaluation)


def test_traces_contain_no_sealed_names_or_gold_values():
    case = load_tiny_fixture(FIXTURE).cases[0]
    forbidden_names = (
        "gold",
        "required_context_item_ids",
        "useful_context_item_ids",
        "misleading_context_item_ids",
        "irrelevant_context_item_ids",
        "feedback",
    )
    sealed_values = set(case.sealed_evaluation.required_context_item_ids)
    for selector in _selectors():
        result = selector.select(case.inputs)
        rendered = repr(result).casefold()
        assert not any(name in rendered for name in forbidden_names)
        assert not sealed_values.intersection(result.__dict__)


def test_invalid_policy_configuration_and_nonfinite_utilities_are_rejected():
    for kwargs in (
        {"relevance_weight": -1.0},
        {"importance_weight": float("nan")},
        {"feature_weights": {"confidence": float("inf")}},
    ):
        with pytest.raises(ValueError):
            StaticPolicySelector(**kwargs)
    with pytest.raises(ValueError):
        AdaptivePolicySelector(learning_weight=-1.0)
    with pytest.raises(ValueError):
        AdaptivePolicySelector(utility_estimates={"signal:test": float("nan")})


@pytest.mark.parametrize(
    "key",
    ["candidate-123", "source:private", "provenance:private", "metadata.path:value"],
)
def test_policy_mappings_reject_non_reusable_feature_keys(key):
    with pytest.raises(ValueError, match="feature|namespace"):
        StaticPolicySelector(feature_weights={key: 1.0})
    with pytest.raises(ValueError, match="feature|namespace"):
        AdaptivePolicySelector(utility_estimates={key: 1.0})


def test_sources_and_arbitrary_metadata_never_reach_callbacks_or_factors():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    candidate_id = inputs.candidate_context[0].context_item_id
    poisoned = tuple(
        replace(
            item,
            source="source-with-{}".format(candidate_id),
            provenance="provenance-with-{}".format(candidate_id),
            metadata=dict(item.metadata, arbitrary="private:{}".format(candidate_id)),
        )
        for item in inputs.candidate_context
    )
    seen = []
    selector = AdaptivePolicySelector(
        utility_estimates=lambda feature: seen.append(feature) or 0.0
    )
    result = selector.select(replace(inputs, candidate_context=poisoned))

    assert seen
    assert all(
        "source" not in feature and candidate_id not in feature for feature in seen
    )
    for decision in result.decisions:
        assert all(
            "source" not in key and "arbitrary" not in key and candidate_id not in key
            for key in decision.score_factors
        )


@pytest.mark.parametrize(
    "field", ["learning_attributes", "control_attributes", "format"]
)
@pytest.mark.parametrize("unsafe_kind", ["candidate_id", "evaluation_label"])
def test_unsafe_approved_metadata_values_are_explicitly_rejected(field, unsafe_kind):
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    item = inputs.candidate_context[0]
    unsafe = (
        "signal:contains-{}".format(item.context_item_id)
        if unsafe_kind == "candidate_id"
        else "signal:gold-required-label"
    )
    if field == "format":
        unsafe = unsafe.replace("signal:", "format:")
    value = unsafe if field == "format" else (unsafe,)
    poisoned = replace(item, metadata=dict(item.metadata, **{field: value}))
    seen = []
    result = AdaptivePolicySelector(
        utility_estimates=lambda feature: seen.append(feature) or 0.0
    ).select(replace(inputs, candidate_context=(poisoned,)))

    assert result.decisions[0].reason == "processing_error"
    assert unsafe not in seen
    assert unsafe not in repr(result.decisions[0].score_factors)


def test_stateful_callback_is_snapshotted_once_per_unique_shared_feature():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    metadata = {
        "learning_attributes": ("signal:shared",),
        "format": "format:note",
    }
    candidates = tuple(
        replace(item, metadata=metadata) for item in inputs.candidate_context[:2]
    )
    constrained = replace(inputs, candidate_context=candidates, token_budget=10_000)
    calls = []

    def monotonically_stateful(feature):
        calls.append(feature)
        return float(len(calls))

    selector = AdaptivePolicySelector(
        utility_estimates=monotonically_stateful,
        feature_weights={},
        relevance_weight=0.0,
        importance_weight=1.0,
    )
    first = selector.select(constrained)
    first_call_count = len(calls)
    second = selector.select(constrained)

    assert first == second
    assert first_call_count == len(set(calls)) == 3
    assert len(calls) == first_call_count
    assert (
        len(
            {
                decision.score_factors["policy.adaptive.feature_utility.signal:shared"]
                for decision in first.decisions
            }
        )
        == 1
    )


@pytest.mark.parametrize("nonfinite", [False, True])
def test_callback_failure_is_cached_once_and_repeats_deterministically(nonfinite):
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    item = replace(
        inputs.candidate_context[0],
        metadata={"learning_attributes": ("signal:failing",)},
    )
    constrained = replace(inputs, candidate_context=(item,))
    calls = []

    def callback(feature):
        calls.append(feature)
        if feature != "signal:failing":
            return 0.0
        if nonfinite:
            return float("nan")
        raise RuntimeError("backend detail")

    selector = AdaptivePolicySelector(utility_estimates=callback)
    first = selector.select(constrained)
    second = selector.select(constrained)

    assert first == second
    assert calls.count("signal:failing") == 1
    assert first.decisions[0].reason == "processing_error"
    assert "backend detail" not in repr(first)


def test_raw_scores_are_logistically_normalized_and_trace_pipeline_math():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    features_and_weights = (
        ("group:above-one-a", 2.0),
        ("group:above-one-b", 3.0),
        ("group:below-zero", -1.0),
    )
    candidates = tuple(
        replace(
            inputs.candidate_context[index],
            confidence=0.5,
            metadata={"learning_attributes": (feature,)},
        )
        for index, (feature, _weight) in enumerate(features_and_weights)
    )
    selector = StaticPolicySelector(
        feature_weights=dict(features_and_weights),
        relevance_weight=0.0,
        importance_weight=1.0,
    )
    result = selector.select(
        replace(
            inputs,
            task_prompt="terms absent everywhere",
            candidate_context=candidates,
            token_budget=10_000,
        )
    )

    assert [item.context_item_id for item in result.selected_items] == [
        candidates[1].context_item_id,
        candidates[0].context_item_id,
        candidates[2].context_item_id,
    ]
    by_id = {decision.context_item_id: decision for decision in result.decisions}
    for candidate, (_feature, raw_score) in zip(candidates, features_and_weights):
        decision = by_id[candidate.context_item_id]
        factors = decision.score_factors
        assert factors["policy.static.raw_score"] == raw_score
        assert factors["policy.raw_score"] == raw_score
        assert factors["policy.normalization_method"] == "logistic"
        assert 0.0 < factors["policy.effective_importance"] < 1.0
        assert factors["retrieval.importance"] == factors["policy.effective_importance"]
        assert (
            factors["retrieval.weighted_importance"]
            == factors["policy.effective_importance"]
        )
        assert factors["retrieval.final_score"] == decision.score
        assert decision.score == factors["policy.effective_importance"]
        assert "static.score" not in factors


def test_negative_adaptive_utility_trace_separates_raw_and_effective_scores():
    inputs = load_tiny_fixture(FIXTURE).cases[0].inputs
    feature = inputs.candidate_context[0].metadata["learning_attributes"][0]
    result = AdaptivePolicySelector(
        utility_estimates={feature: -2.0},
        learning_weight=0.5,
        feature_weights={},
        relevance_weight=0.0,
        importance_weight=1.0,
    ).select(replace(inputs, token_budget=10_000))
    decision = result.decisions[0]
    factors = decision.score_factors

    assert factors["policy.adaptive.raw_utility"] < 0.0
    assert factors["policy.adaptive.weighted_utility_contribution"] < 0.0
    assert factors["policy.adaptive.raw_score"] == factors["policy.raw_score"]
    assert factors["policy.raw_score"] < 0.0
    assert factors["retrieval.importance"] == factors["policy.effective_importance"]
    assert factors["retrieval.final_score"] == decision.score


def test_static_configuration_is_frozen_and_has_no_fixture_specific_defaults():
    selector = StaticPolicySelector()
    assert all(
        marker not in repr(selector).casefold()
        for marker in ("net-", "tf-", "required", "useful", "misleading", "irrelevant")
    )
    with pytest.raises(TypeError):
        selector.feature_weights["confidence"] = 99.0
