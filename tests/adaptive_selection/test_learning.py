from collections.abc import Mapping
from dataclasses import FrozenInstanceError, dataclass, replace
import ast
import inspect
import json
from pathlib import Path

import pytest

from experiments.adaptive_selection.dataset import load_tiny_fixture
from experiments.adaptive_selection.learning import (
    IDLocalUtilityEstimate,
    LearningPolicy,
    LearningSnapshot,
    learn_utilities,
)
from experiments.adaptive_selection.schema import (
    FeedbackEvent,
    TaskInputs,
    UtilityEstimate,
)
from experiments.adaptive_selection.selectors import (
    AdaptivePolicySelector,
    StaticPolicySelector,
    reusable_features,
)

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_experiment.json"
CLOCK = lambda: "2026-07-29T10:00:00Z"


def _bundle_parts():
    bundle = load_tiny_fixture(FIXTURE)
    inputs = {case.task_case_id: case.inputs for case in bundle.cases}
    return bundle, inputs


def _learn(events=None, inputs=None, policy=None, clock=CLOCK):
    bundle, fixture_inputs = _bundle_parts()
    return learn_utilities(
        bundle.adaptation_feedback if events is None else events,
        fixture_inputs if inputs is None else inputs,
        LearningPolicy() if policy is None else policy,
        clock,
    )


def _numeric(event, *, event_id, value, affected=None, signal_type="context_utility"):
    correction = signal_type == "correction"
    return replace(
        event,
        event_id=event_id,
        signal_type=signal_type,
        numeric_value=value,
        structured_value=None,
        affected_context_item_ids=(
            (event.affected_context_item_ids[0],)
            if affected is None
            else tuple(affected)
        ),
        correction_category="factual" if correction else None,
        correction_text="Replace the incorrect claim." if correction else None,
    )


def _estimate(snapshot, family, feature):
    return next(
        estimate
        for estimate in snapshot.feature_estimates
        if estimate.task_family_id == family
        and estimate.context_attributes == (feature,)
    )


def test_no_feedback_is_empty_and_adaptive_equals_static():
    _, inputs = _bundle_parts()
    task_inputs = next(iter(inputs.values()))
    snapshot = _learn(events=())

    assert snapshot.feature_estimates == ()
    assert snapshot.id_local_estimates == ()
    assert dict(snapshot.feature_utilities) == {}
    assert dict(snapshot.id_local_utilities) == {}
    assert (
        dict(snapshot.feature_utilities_for(task_inputs.profile.task_family_id)) == {}
    )
    assert (
        AdaptivePolicySelector(
            utility_estimates=snapshot.feature_utilities_for(
                task_inputs.profile.task_family_id
            )
        )
        .select(task_inputs)
        .policy_signature
        == StaticPolicySelector().select(task_inputs).policy_signature
    )


def test_one_event_is_provisional_and_second_consistent_event_is_active():
    bundle, _ = _bundle_parts()
    family = bundle.adaptation_feedback[0].task_family_id
    feature = "basis:observed"

    first = _learn(events=bundle.adaptation_feedback[:1])
    second = _learn(events=bundle.adaptation_feedback[:2])

    provisional = _estimate(first, family, feature)
    active = _estimate(second, family, feature)
    assert provisional.estimated_utility == pytest.approx(1.0 / 3.0)
    assert provisional.confidence == pytest.approx(1.0 / 3.0)
    assert provisional.source_event_ids == ("feedback-01",)
    assert feature not in first.feature_utilities_for(family)
    assert active.estimated_utility == pytest.approx(0.5)
    assert active.confidence == pytest.approx(0.5)
    assert active.source_event_ids == ("feedback-01", "feedback-02")
    assert second.feature_utilities_for(family)[feature] == active.estimated_utility


def test_useful_raises_harmful_and_numeric_correction_lower_with_smoothing():
    bundle, inputs = _bundle_parts()
    first, second = bundle.adaptation_feedback[:2]
    family = first.task_family_id
    structured = _learn(events=(first, second))
    assert structured.feature_utilities_for(family)["basis:observed"] > 0
    assert structured.feature_utilities_for(family)["basis:assumed"] < 0

    item_id = first.affected_context_item_ids[0]
    negative_events = (
        _numeric(first, event_id="numeric-negative", value=-0.5, affected=(item_id,)),
        _numeric(
            first,
            event_id="correction-negative",
            value=-1.0,
            affected=(item_id,),
            signal_type="correction",
        ),
    )
    snapshot = _learn(events=negative_events, inputs=inputs)
    target_features = reusable_features(
        inputs[first.task_case_id].candidate_context[0],
        tuple(
            item.context_item_id
            for item in inputs[first.task_case_id].candidate_context
        ),
    )
    assert all(
        snapshot.feature_utilities_for(family)[feature] < 0
        for feature in target_features
    )
    estimate = _estimate(snapshot, family, target_features[0])
    assert -1.0 <= estimate.estimated_utility <= 1.0
    assert estimate.estimated_utility == pytest.approx(-0.375)


def test_contradiction_preserves_ordered_source_ids_and_averages_by_event():
    bundle, _ = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    item_id = event.affected_context_item_ids[0]
    _, inputs = _bundle_parts()
    task_inputs = inputs[event.task_case_id]
    feature = reusable_features(
        task_inputs.candidate_context[0],
        tuple(item.context_item_id for item in task_inputs.candidate_context),
    )[0]
    events = (
        _numeric(event, event_id="positive", value=1.0, affected=(item_id,)),
        _numeric(event, event_id="negative", value=-1.0, affected=(item_id,)),
    )

    estimate = _estimate(_learn(events=events), event.task_family_id, feature)
    assert estimate.estimated_utility == 0.0
    assert estimate.source_event_ids == ("positive", "negative")


def test_repeated_feature_occurrence_counts_once_per_event():
    bundle, _ = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    estimate = _estimate(
        _learn(events=(event,)), event.task_family_id, "basis:observed"
    )
    assert estimate.source_event_ids == (event.event_id,)
    assert estimate.confidence == pytest.approx(1.0 / 3.0)
    assert estimate.estimated_utility == pytest.approx(1.0 / 3.0)


def test_family_isolation_and_ambiguous_feature_gets_mixed_evidence():
    snapshot = _learn()
    assert set(snapshot.feature_utilities) == {
        "hybrid-network-return-routing",
        "terraform-drift-state",
    }
    for family in snapshot.feature_utilities:
        family_map = snapshot.feature_utilities_for(family)
        assert family_map["basis:observed"] > 0
        assert family_map.get("presentation:teal-header", 0.0) <= 0.0

    mixed = _estimate(
        snapshot,
        "hybrid-network-return-routing",
        "signal:path-correlation",
    )
    assert len(mixed.source_event_ids) == 2
    assert -1.0 < mixed.estimated_utility < 1.0


def test_controls_and_format_trap_do_not_gain_under_balanced_feedback():
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    task_inputs = inputs[event.task_case_id]
    target = next(
        item
        for item in task_inputs.candidate_context
        if "format:runbook"
        in reusable_features(
            item,
            tuple(member.context_item_id for member in task_inputs.candidate_context),
        )
        and "presentation:teal-header"
        in reusable_features(
            item,
            tuple(member.context_item_id for member in task_inputs.candidate_context),
        )
    )
    events = (
        _numeric(
            event,
            event_id="balanced-positive",
            value=1.0,
            affected=(target.context_item_id,),
        ),
        _numeric(
            event,
            event_id="balanced-negative",
            value=-1.0,
            affected=(target.context_item_id,),
        ),
    )
    snapshot = _learn(events=events, inputs=inputs)
    family_map = snapshot.feature_utilities_for(event.task_family_id)
    assert family_map["format:runbook"] == 0.0
    assert family_map["presentation:teal-header"] == 0.0


def test_heldout_ids_have_no_id_local_estimate_but_shared_feature_affects_selection():
    bundle, _ = _bundle_parts()
    snapshot = _learn()
    adaptation_ids = {
        context_id
        for event in bundle.adaptation_feedback
        for context_id in event.affected_context_item_ids
    }
    heldout = bundle.cases[2].inputs
    heldout_ids = {item.context_item_id for item in heldout.candidate_context}
    family = heldout.profile.task_family_id
    assert heldout_ids.isdisjoint(adaptation_ids)
    assert heldout_ids.isdisjoint(snapshot.id_local_utilities_for(family))

    feature_map = snapshot.feature_utilities_for(family)
    candidate_ids = tuple(item.context_item_id for item in heldout.candidate_context)
    shared = next(
        feature
        for item in heldout.candidate_context
        for feature in reusable_features(item, candidate_ids)
        if feature in feature_map and feature_map[feature] != 0
    )
    normalized = replace(
        heldout,
        candidate_context=tuple(
            replace(item, token_count=1) for item in heldout.candidate_context
        ),
        token_budget=1,
    )
    selector_kwargs = {
        "feature_weights": {},
        "relevance_weight": 0.0,
        "importance_weight": 1.0,
    }
    static = StaticPolicySelector(**selector_kwargs).select(normalized)
    result = AdaptivePolicySelector(
        utility_estimates=feature_map, **selector_kwargs
    ).select(normalized)
    assert tuple(item.context_item_id for item in result.selected_items) != tuple(
        item.context_item_id for item in static.selected_items
    )
    assert any(
        "policy.adaptive.feature_utility.{}".format(shared) in decision.score_factors
        for decision in result.decisions
    )


def test_renaming_adaptation_ids_preserves_feature_estimates_and_changes_only_id_keys():
    bundle, inputs = _bundle_parts()
    events = bundle.adaptation_feedback[:2]
    original = _learn(events=events, inputs=inputs)
    rename = {
        context_id: "renamed-{}".format(index)
        for index, context_id in enumerate(
            dict.fromkeys(
                context_id
                for event in events
                for context_id in event.affected_context_item_ids
            )
        )
    }
    renamed_inputs = dict(inputs)
    renamed_events = []
    for event in events:
        case_inputs = renamed_inputs[event.task_case_id]
        renamed_inputs[event.task_case_id] = replace(
            case_inputs,
            candidate_context=tuple(
                replace(
                    item,
                    context_item_id=rename.get(
                        item.context_item_id, item.context_item_id
                    ),
                )
                for item in case_inputs.candidate_context
            ),
        )
        payload = dict(event.structured_value)
        for key in ("useful_context_item_ids", "harmful_context_item_ids"):
            payload[key] = tuple(rename[value] for value in payload[key])
        renamed_events.append(
            replace(
                event,
                affected_context_item_ids=tuple(
                    rename[value] for value in event.affected_context_item_ids
                ),
                structured_value=payload,
            )
        )

    renamed = _learn(events=renamed_events, inputs=renamed_inputs)
    assert tuple(item.to_dict() for item in renamed.feature_estimates) == tuple(
        item.to_dict() for item in original.feature_estimates
    )
    assert {item.context_item_id for item in renamed.id_local_estimates} != {
        item.context_item_id for item in original.id_local_estimates
    }


def test_semantic_candidate_id_collision_remains_intentionally_invalid():
    """ID renaming is invariant only for opaque IDs outside the feature vocabulary."""

    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    old_id = event.affected_context_item_ids[0]
    collision = "observed"
    case_inputs = inputs[event.task_case_id]
    collided_inputs = dict(inputs)
    collided_inputs[event.task_case_id] = replace(
        case_inputs,
        candidate_context=tuple(
            (
                replace(item, context_item_id=collision)
                if item.context_item_id == old_id
                else item
            )
            for item in case_inputs.candidate_context
        ),
    )
    payload = dict(event.structured_value)
    for key in ("useful_context_item_ids", "harmful_context_item_ids"):
        payload[key] = tuple(
            collision if value == old_id else value for value in payload[key]
        )
    collided_event = replace(
        event,
        affected_context_item_ids=tuple(
            collision if value == old_id else value
            for value in event.affected_context_item_ids
        ),
        structured_value=payload,
    )

    with pytest.raises(ValueError, match="candidate ID"):
        _learn(events=(collided_event,), inputs=collided_inputs)


def test_exact_reveal_prefix_and_caller_order_drive_provenance():
    bundle, _ = _bundle_parts()
    events = bundle.adaptation_feedback[:2]
    first = _learn(events=events[:1])
    reversed_snapshot = _learn(events=tuple(reversed(events)))
    estimate = _estimate(reversed_snapshot, events[0].task_family_id, "basis:observed")
    assert estimate.source_event_ids == (events[1].event_id, events[0].event_id)
    assert events[1].event_id not in repr(first.to_dict())


@pytest.mark.parametrize(
    "policy_kwargs",
    [
        {"prior_mean": float("nan")},
        {"prior_mean": 1.1},
        {"prior_strength": 0},
        {"prior_strength": float("inf")},
        {"minimum_evidence_count": 0},
        {"minimum_evidence_count": True},
        {"accepted_signal_types": {"task_score"}},
        {"id_local_enabled": 1},
        {"estimator_version": ""},
        {"credit_assignment_version": ""},
    ],
)
def test_policy_validation(policy_kwargs):
    with pytest.raises(ValueError):
        LearningPolicy(**policy_kwargs)


def test_empty_accepted_signal_set_is_valid_and_rejects_every_event():
    policy = LearningPolicy(accepted_signal_types=frozenset())
    assert policy.accepted_signal_types == frozenset()
    assert _learn(events=(), policy=policy).to_dict() == {
        "feature_estimates": [],
        "feature_utilities": {},
        "id_local_estimates": [],
        "id_local_utilities": {},
        "policy": policy.to_dict(),
    }
    bundle, inputs = _bundle_parts()
    with pytest.raises(ValueError, match="unsupported|accepted"):
        _learn(events=(bundle.adaptation_feedback[0],), inputs=inputs, policy=policy)


def test_recursive_immutability_and_defensive_copies():
    accepted = {"context_utility", "correction"}
    policy = LearningPolicy(accepted_signal_types=accepted)
    accepted.clear()
    snapshot = _learn(policy=policy)
    assert policy.accepted_signal_types == frozenset({"context_utility", "correction"})
    with pytest.raises(FrozenInstanceError):
        policy.prior_mean = 0.2
    with pytest.raises(TypeError):
        snapshot.feature_utilities["new"] = {}
    family = next(iter(snapshot.feature_utilities))
    with pytest.raises(TypeError):
        snapshot.feature_utilities[family]["basis:observed"] = 0.0
    with pytest.raises(FrozenInstanceError):
        snapshot.id_local_estimates[0].estimated_utility = 0.0


def test_id_local_can_be_disabled_without_affecting_feature_learning():
    enabled = _learn()
    disabled = _learn(policy=LearningPolicy(id_local_enabled=False))
    assert disabled.id_local_estimates == ()
    assert dict(disabled.id_local_utilities) == {}
    assert [item.estimated_utility for item in disabled.feature_estimates] == [
        item.estimated_utility for item in enabled.feature_estimates
    ]


def test_all_learned_feature_keys_are_accepted_unchanged_by_selector():
    snapshot = _learn()
    for family, utility_map in snapshot.feature_utilities.items():
        assert AdaptivePolicySelector(utility_estimates=utility_map)
        assert all(
            reusable_features(
                next(
                    case.inputs.candidate_context[0]
                    for case in load_tiny_fixture(FIXTURE).cases
                    if case.inputs.profile.task_family_id == family
                ),
                (),
            )
            is not None
            for _feature in utility_map
        )


def test_material_policy_changes_recompute_feature_estimate_ids():
    bundle, _ = _bundle_parts()
    events = bundle.adaptation_feedback[:2]
    baseline = _learn(events=events)
    variants = (
        LearningPolicy(estimator_version="estimator-v2"),
        LearningPolicy(credit_assignment_version="credit-v2"),
        LearningPolicy(prior_mean=0.1),
        LearningPolicy(prior_strength=3.0),
        LearningPolicy(minimum_evidence_count=3),
        LearningPolicy(accepted_signal_types={"context_utility"}),
        LearningPolicy(id_local_enabled=False),
    )
    baseline_ids = {item.utility_estimate_id for item in baseline.feature_estimates}
    for policy in variants:
        assert {
            item.utility_estimate_id
            for item in _learn(events=events, policy=policy).feature_estimates
        }.isdisjoint(baseline_ids)


def test_reward_changes_recompute_feature_and_id_local_estimate_ids():
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    context_item_id = event.affected_context_item_ids[0]
    positive_event = _numeric(
        event,
        event_id="same-event-id",
        value=1.0,
        affected=(context_item_id,),
    )
    negative_event = replace(positive_event, numeric_value=-1.0)

    positive = _learn(events=(positive_event,), inputs=inputs)
    negative = _learn(events=(negative_event,), inputs=inputs)

    assert {
        estimate.utility_estimate_id for estimate in positive.feature_estimates
    }.isdisjoint(
        estimate.utility_estimate_id for estimate in negative.feature_estimates
    )
    assert {
        estimate.id_local_utility_estimate_id
        for estimate in positive.id_local_estimates
    }.isdisjoint(
        estimate.id_local_utility_estimate_id
        for estimate in negative.id_local_estimates
    )


def test_estimate_identity_includes_derived_utility_and_confidence():
    import experiments.adaptive_selection.learning as learning

    policy = LearningPolicy()
    args = ("feature-utility", policy, "family", "basis:observed", ("event",))
    baseline = learning._identity(*args, 0.25, 0.5)

    assert learning._identity(*args, -0.25, 0.5) != baseline
    assert learning._identity(*args, 0.25, 0.25) != baseline
    assert learning._identity(*args, 0.25, 0.5) == baseline


def test_same_inputs_policy_and_clock_are_byte_equivalent():
    first = _learn()
    second = _learn()
    assert json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    ) == json.dumps(second.to_dict(), sort_keys=True, separators=(",", ":"))


def test_learning_snapshot_rejects_all_public_construction():
    snapshot = _learn()
    forged_fields = {
        "policy": snapshot.policy,
        "feature_estimates": snapshot.feature_estimates,
        "id_local_estimates": snapshot.id_local_estimates,
        "feature_utilities": snapshot.feature_utilities,
        "id_local_utilities": snapshot.id_local_utilities,
    }

    with pytest.raises(TypeError, match="learn_utilities"):
        LearningSnapshot()
    with pytest.raises(TypeError, match="learn_utilities"):
        LearningSnapshot(**forged_fields)
    with pytest.raises(TypeError, match="learn_utilities"):
        LearningSnapshot._from_learning(
            object(), estimated_timestamp=CLOCK(), **forged_fields
        )
    active = next(
        item
        for item in snapshot.feature_estimates
        if len(item.source_event_ids) >= snapshot.policy.minimum_evidence_count
    )
    for forged_estimate in (
        replace(active, confidence=0.0),
        replace(active, estimated_timestamp="2026-07-30T10:00:00Z"),
        replace(active, estimated_utility=-active.estimated_utility),
    ):
        with pytest.raises(TypeError, match="learn_utilities"):
            LearningSnapshot(
                **dict(
                    forged_fields,
                    feature_estimates=(forged_estimate,),
                    feature_utilities={
                        forged_estimate.task_family_id: {
                            forged_estimate.context_attributes[0]: (
                                forged_estimate.estimated_utility
                            )
                        }
                    },
                )
            )
    with pytest.raises(TypeError, match="learn_utilities"):
        LearningSnapshot(
            **dict(
                forged_fields,
                feature_utilities={"family": {"basis:observed": 0.5}},
            )
        )


def test_learning_snapshot_rejects_subclasses_before_they_can_forge_instances():
    created_types = []

    def define_plain_subclass():
        class PlainSnapshot(LearningSnapshot):
            pass

        created_types.append(PlainSnapshot)

    def define_dataclass_subclass():
        @dataclass(frozen=True)
        class DataclassSnapshot(LearningSnapshot):
            pass

        created_types.append(DataclassSnapshot)

    def define_custom_init_subclass():
        class CustomInitSnapshot(LearningSnapshot):
            def __init__(self):
                pass

        created_types.append(CustomInitSnapshot)

    for define_subclass in (
        define_plain_subclass,
        define_dataclass_subclass,
        define_custom_init_subclass,
    ):
        with pytest.raises(TypeError, match="cannot be subclassed"):
            define_subclass()

    assert created_types == []
    snapshot = _learn()
    assert type(snapshot) is LearningSnapshot
    assert isinstance(snapshot, LearningSnapshot)


def test_learner_uses_one_clock_and_formula_correct_confidence():
    calls = []

    def clock():
        calls.append(None)
        return "2026-07-29T10:00:00Z"

    snapshot = _learn(clock=clock)
    estimates = snapshot.feature_estimates + snapshot.id_local_estimates

    assert len(calls) == 1
    assert {item.estimated_timestamp for item in estimates} == {"2026-07-29T10:00:00Z"}
    for estimate in estimates:
        count = len(estimate.source_event_ids)
        assert estimate.confidence == count / (snapshot.policy.prior_strength + count)


def test_estimate_identity_is_stable_when_only_clock_changes():
    first = _learn(clock=lambda: "2026-07-29T10:00:00Z")
    second = _learn(clock=lambda: "2026-07-30T10:00:00Z")

    assert [item.utility_estimate_id for item in first.feature_estimates] == [
        item.utility_estimate_id for item in second.feature_estimates
    ]
    assert [item.id_local_utility_estimate_id for item in first.id_local_estimates] == [
        item.id_local_utility_estimate_id for item in second.id_local_estimates
    ]


def test_output_public_types_are_frozen_and_well_typed():
    snapshot = _learn()
    assert isinstance(snapshot, LearningSnapshot)
    assert all(isinstance(item, UtilityEstimate) for item in snapshot.feature_estimates)
    assert all(
        isinstance(item, IDLocalUtilityEstimate) for item in snapshot.id_local_estimates
    )
    assert isinstance(snapshot.feature_utilities, Mapping)


def test_public_boundary_and_module_contain_no_sealed_or_repository_types():
    import experiments.adaptive_selection.learning as learning

    source = inspect.getsource(learning)
    tree = ast.parse(source)
    forbidden = {
        "TaskCase",
        "SealedEvaluation",
        "DatasetBundle",
        "ExperimentRepository",
    }
    assert forbidden.isdisjoint(source.split())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("repository" in name for name in imported)
    signature = inspect.signature(learn_utilities)
    assert signature.parameters["events"].annotation in (
        "Sequence[FeedbackEvent]",
        "Iterable[FeedbackEvent]",
    )
    assert (
        signature.parameters["inputs_by_task_case_id"].annotation
        == "Mapping[str, TaskInputs]"
    )


def test_rejects_duplicate_event_ids_missing_case_family_mismatch_and_unknown_ids():
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    with pytest.raises(ValueError, match="event IDs.*unique"):
        _learn(events=(event, event), inputs=inputs)
    with pytest.raises(ValueError, match="task_case_id"):
        _learn(events=(event,), inputs={})
    with pytest.raises(ValueError, match="family"):
        _learn(events=(replace(event, task_family_id="other-family"),), inputs=inputs)
    unknown = replace(event, affected_context_item_ids=("unknown-id",))
    payload = dict(event.structured_value)
    payload["useful_context_item_ids"] = ("unknown-id",)
    payload["harmful_context_item_ids"] = ()
    unknown = replace(unknown, structured_value=payload)
    with pytest.raises(ValueError, match="unknown"):
        _learn(events=(unknown,), inputs=inputs)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda payload: {
                key: value for key, value in payload.items() if key != "locked"
            },
            "keys",
        ),
        (lambda payload: dict(payload, locked=False), "locked"),
        (
            lambda payload: dict(payload, selector_independent=False),
            "selector_independent",
        ),
        (
            lambda payload: dict(payload, useful_context_item_ids="not-a-list"),
            "useful_context_item_ids",
        ),
        (
            lambda payload: dict(
                payload,
                harmful_context_item_ids=payload["useful_context_item_ids"][:1],
            ),
            "disjoint",
        ),
        (
            lambda payload: dict(payload, extra_target_label="forbidden"),
            "keys",
        ),
    ],
)
def test_rejects_malformed_structured_context_utility(mutation, match):
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    with pytest.raises(ValueError, match=match):
        _learn(
            events=(
                replace(event, structured_value=mutation(dict(event.structured_value))),
            ),
            inputs=inputs,
        )


@pytest.mark.parametrize(
    "signal_type", ["task_score", "selection_quality", "preference"]
)
def test_rejects_unsupported_feedback_types(signal_type):
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    value = 0.5
    unsupported = replace(
        event,
        signal_type=signal_type,
        numeric_value=value,
        structured_value=None,
    )
    with pytest.raises(ValueError, match="unsupported|accepted"):
        _learn(events=(unsupported,), inputs=inputs)


def test_rejects_structured_correction_nonnegative_correction_and_wrong_record_types():
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    structured = replace(
        event,
        event_id="structured-correction",
        signal_type="correction",
        numeric_value=None,
        structured_value={"severity": "major"},
        correction_category="factual",
        correction_text="Correct it.",
    )
    with pytest.raises(ValueError, match="structured correction"):
        _learn(events=(structured,), inputs=inputs)
    for value in (0.0, 0.5):
        with pytest.raises(ValueError, match="strictly negative"):
            _learn(
                events=(
                    _numeric(
                        event,
                        event_id="correction-{}".format(value),
                        value=value,
                        signal_type="correction",
                    ),
                ),
                inputs=inputs,
            )
    with pytest.raises(TypeError, match="FeedbackEvent"):
        _learn(events=(object(),), inputs=inputs)
    with pytest.raises(TypeError, match="TaskInputs"):
        _learn(events=(event,), inputs={event.task_case_id: object()})


@pytest.mark.parametrize("value", [True, -1.01, 1.01, float("nan"), float("inf")])
def test_learner_revalidates_tampered_numeric_context_utility(value):
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    object.__setattr__(event, "structured_value", None)
    object.__setattr__(event, "numeric_value", value)
    with pytest.raises(ValueError, match="numeric_value"):
        _learn(events=(event,), inputs=inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("numeric_value", True),
        ("numeric_value", -1.01),
        ("numeric_value", 1.01),
        ("numeric_value", float("nan")),
        ("numeric_value", float("inf")),
        ("correction_category", None),
        ("correction_category", ""),
        ("correction_text", None),
        ("correction_text", "   "),
    ],
)
def test_learner_revalidates_tampered_numeric_correction(field, value):
    bundle, inputs = _bundle_parts()
    original = bundle.adaptation_feedback[0]
    event = _numeric(
        original,
        event_id="tampered-correction",
        value=-0.5,
        signal_type="correction",
    )
    object.__setattr__(event, field, value)
    with pytest.raises(ValueError, match="numeric_value|correction"):
        _learn(events=(event,), inputs=inputs)


def test_learner_rejects_tampered_context_utility_with_correction_fields():
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    object.__setattr__(event, "structured_value", None)
    object.__setattr__(event, "numeric_value", 0.5)
    object.__setattr__(event, "correction_category", "factual")
    object.__setattr__(event, "correction_text", "Injected correction metadata")
    with pytest.raises(ValueError, match="correction.*null"):
        _learn(events=(event,), inputs=inputs)


@pytest.mark.parametrize(
    "affected",
    [
        lambda context_id: (context_id, context_id),
        lambda context_id: ("",),
        lambda context_id: (42,),
    ],
)
def test_learner_completely_revalidates_tampered_affected_ids(affected):
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    object.__setattr__(
        event,
        "affected_context_item_ids",
        affected(event.affected_context_item_ids[0]),
    )

    with pytest.raises(ValueError, match="affected_context_item_ids"):
        _learn(events=(event,), inputs=inputs)


@pytest.mark.parametrize("tamper", ["duplicate-candidates", "blank-family"])
def test_learner_completely_revalidates_tampered_task_inputs(tamper):
    bundle, inputs = _bundle_parts()
    event = bundle.adaptation_feedback[0]
    task_inputs = inputs[event.task_case_id]
    if tamper == "duplicate-candidates":
        object.__setattr__(
            task_inputs,
            "candidate_context",
            (task_inputs.candidate_context[0], task_inputs.candidate_context[0]),
        )
        match = "candidate context IDs must be unique"
    else:
        object.__setattr__(task_inputs.profile, "task_family_id", "")
        match = "task_family_id must be nonempty"

    with pytest.raises(ValueError, match=match):
        _learn(events=(event,), inputs=inputs)
