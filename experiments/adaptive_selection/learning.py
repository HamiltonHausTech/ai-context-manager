"""Deterministic utility learning from an exact ordered feedback reveal prefix.

The learner is stateless and consumes only feedback records, selector-visible inputs, a
frozen policy, and an injected clock. Reusable-feature estimates and the ID-local
ablation are deliberately separate outputs; selectors receive only a family feature map.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, DefaultDict, Dict, FrozenSet, List, Tuple

from .schema import FeedbackEvent, TaskInputs, UtilityEstimate
from .selectors import reusable_features, validate_reusable_feature

_ALLOWED_SIGNAL_TYPES = frozenset({"context_utility", "correction"})
_STRUCTURED_CONTEXT_UTILITY_KEYS = frozenset(
    {
        "ambiguous_attributes",
        "harmful_attributes",
        "harmful_context_item_ids",
        "locked",
        "no_effect_attributes",
        "selector_independent",
        "shared_feature_trap",
        "useful_attributes",
        "useful_context_item_ids",
    }
)
_ATTRIBUTE_DIAGNOSTIC_KEYS = (
    "ambiguous_attributes",
    "harmful_attributes",
    "no_effect_attributes",
    "useful_attributes",
)


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be nonempty".format(name))
    return value


def _finite(name: str, value: Any) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError("{} must be finite".format(name))
    return float(value)


def _string_tuple(name: str, value: Any) -> Tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("{} must be a sequence of strings".format(name))
    copied = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in copied):
        raise ValueError("{} must be a sequence of nonempty strings".format(name))
    if len(set(copied)) != len(copied):
        raise ValueError("{} must be unique".format(name))
    return copied


def _freeze_nested_float_mapping(
    values: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType(
        {
            family: MappingProxyType(
                {key: float(values[family][key]) for key in sorted(values[family])}
            )
            for family in sorted(values)
        }
    )


def _validate_context_item_id(context_item_id: Any) -> str:
    return _nonempty("context_item_id", context_item_id)


def _validated_utility_mapping(
    name: str,
    values: Any,
    key_validator: Callable[[Any], str],
) -> Dict[str, Dict[str, float]]:
    if not isinstance(values, Mapping):
        raise ValueError("{} must be a mapping".format(name))
    copied: Dict[str, Dict[str, float]] = {}
    for family, family_values in values.items():
        family = _nonempty("{} family".format(name), family)
        if not isinstance(family_values, Mapping):
            raise ValueError("{} family values must be mappings".format(name))
        copied[family] = {}
        for key, value in family_values.items():
            validated_key = key_validator(key)
            utility = _finite("{} value".format(name), value)
            if not -1.0 <= utility <= 1.0:
                raise ValueError("{} value must be between -1 and 1".format(name))
            copied[family][validated_key] = utility
    return copied


def _validate_estimate_common(
    task_family_id: Any,
    estimated_utility: Any,
    confidence: Any,
    source_event_ids: Any,
    estimator_version: Any,
    provenance: Any,
    policy: "LearningPolicy",
    expected_provenance: str,
) -> None:
    _nonempty("task_family_id", task_family_id)
    utility = _finite("estimated_utility", estimated_utility)
    if not -1.0 <= utility <= 1.0:
        raise ValueError("estimated_utility must be between -1 and 1")
    confidence_value = _finite("confidence", confidence)
    if not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    source_ids = _string_tuple("source_event_ids", source_event_ids)
    if not source_ids:
        raise ValueError("source_event_ids must not be empty")
    if estimator_version != policy.estimator_version:
        raise ValueError("estimate estimator_version must match learning policy")
    if provenance != expected_provenance:
        raise ValueError("estimate provenance must match learning policy")


def _eligible_utilities(
    targets: Mapping[Tuple[str, str], Any], minimum_evidence_count: int
) -> Dict[str, Dict[str, float]]:
    eligible: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    for (family, target), estimate in targets.items():
        if len(estimate.source_event_ids) >= minimum_evidence_count:
            eligible[family][target] = float(estimate.estimated_utility)
    return dict(eligible)


@dataclass(frozen=True)
class LearningPolicy:
    """Frozen mechanics for Task 6 utility estimation."""

    estimator_version: str = "feature-utility-v1"
    credit_assignment_version: str = "stage0-item-credit-v1"
    prior_mean: float = 0.0
    prior_strength: float = 2.0
    minimum_evidence_count: int = 2
    accepted_signal_types: FrozenSet[str] = field(
        default_factory=lambda: frozenset(_ALLOWED_SIGNAL_TYPES)
    )
    id_local_enabled: bool = True

    def __post_init__(self) -> None:
        _nonempty("estimator_version", self.estimator_version)
        _nonempty("credit_assignment_version", self.credit_assignment_version)
        prior_mean = _finite("prior_mean", self.prior_mean)
        if not -1.0 <= prior_mean <= 1.0:
            raise ValueError("prior_mean must be between -1 and 1")
        prior_strength = _finite("prior_strength", self.prior_strength)
        if prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        if (
            not isinstance(self.minimum_evidence_count, int)
            or isinstance(self.minimum_evidence_count, bool)
            or self.minimum_evidence_count <= 0
        ):
            raise ValueError("minimum_evidence_count must be a positive integer")
        try:
            accepted = frozenset(self.accepted_signal_types)
        except TypeError:
            raise ValueError("accepted_signal_types must be a collection")
        if not accepted.issubset(_ALLOWED_SIGNAL_TYPES):
            raise ValueError(
                "accepted_signal_types may contain only context_utility and correction"
            )
        if any(not isinstance(signal, str) for signal in accepted):
            raise ValueError("accepted_signal_types must contain strings")
        if not isinstance(self.id_local_enabled, bool):
            raise ValueError("id_local_enabled must be boolean")
        object.__setattr__(self, "prior_mean", prior_mean)
        object.__setattr__(self, "prior_strength", prior_strength)
        object.__setattr__(self, "accepted_signal_types", accepted)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_signal_types": sorted(self.accepted_signal_types),
            "credit_assignment_version": self.credit_assignment_version,
            "estimator_version": self.estimator_version,
            "id_local_enabled": self.id_local_enabled,
            "minimum_evidence_count": self.minimum_evidence_count,
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
        }


@dataclass(frozen=True)
class IDLocalUtilityEstimate:
    """Immutable diagnostic estimate for one family-local context-item ID."""

    id_local_utility_estimate_id: str
    task_family_id: str
    context_item_id: str
    estimated_utility: float
    confidence: float
    source_event_ids: Tuple[str, ...]
    estimator_version: str
    estimated_timestamp: str
    provenance: str

    def __post_init__(self) -> None:
        for name in (
            "id_local_utility_estimate_id",
            "task_family_id",
            "context_item_id",
            "estimator_version",
            "estimated_timestamp",
            "provenance",
        ):
            _nonempty(name, getattr(self, name))
        utility = _finite("estimated_utility", self.estimated_utility)
        confidence = _finite("confidence", self.confidence)
        if not -1.0 <= utility <= 1.0:
            raise ValueError("estimated_utility must be between -1 and 1")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        source_ids = _string_tuple("source_event_ids", self.source_event_ids)
        if not source_ids:
            raise ValueError("source_event_ids must not be empty")
        # Reuse the canonical timestamp validator on the existing artifact type.
        UtilityEstimate(
            utility_estimate_id="timestamp-validation",
            task_family_id=self.task_family_id,
            context_attributes=("tag:timestamp-validation",),
            estimated_utility=utility,
            confidence=confidence,
            source_event_ids=source_ids,
            estimator_version=self.estimator_version,
            estimated_timestamp=self.estimated_timestamp,
            provenance=self.provenance,
        )
        object.__setattr__(self, "estimated_utility", utility)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_event_ids", source_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence": self.confidence,
            "context_item_id": self.context_item_id,
            "estimated_timestamp": self.estimated_timestamp,
            "estimated_utility": self.estimated_utility,
            "estimator_version": self.estimator_version,
            "id_local_utility_estimate_id": self.id_local_utility_estimate_id,
            "provenance": self.provenance,
            "source_event_ids": list(self.source_event_ids),
            "task_family_id": self.task_family_id,
        }


_LEARNING_SNAPSHOT_TOKEN = object()


@dataclass(frozen=True, init=False)
class LearningSnapshot:
    """One immutable result for an exact caller-ordered reveal prefix.

    The outer mapping key is always ``task_family_id``. Inner feature mappings are the
    only mappings intended for ``AdaptivePolicySelector``. ID-local mappings remain a
    separate ablation and can never enter the selector through this API accidentally.

    Snapshots have no public construction path. ``learn_utilities`` is the trusted
    derivation boundary because validating an arbitrary snapshot without its candidate
    context and reward evidence would necessarily be incomplete.
    """

    policy: LearningPolicy
    feature_estimates: Tuple[UtilityEstimate, ...]
    id_local_estimates: Tuple[IDLocalUtilityEstimate, ...]
    feature_utilities: Mapping[str, Mapping[str, float]]
    id_local_utilities: Mapping[str, Mapping[str, float]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        raise TypeError("LearningSnapshot cannot be subclassed")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "LearningSnapshot instances are created only by learn_utilities"
        )

    @classmethod
    def _from_learning(
        cls,
        token: object,
        *,
        policy: LearningPolicy,
        feature_estimates: Tuple[UtilityEstimate, ...],
        id_local_estimates: Tuple[IDLocalUtilityEstimate, ...],
        feature_utilities: Mapping[str, Mapping[str, float]],
        id_local_utilities: Mapping[str, Mapping[str, float]],
        estimated_timestamp: Any,
    ) -> "LearningSnapshot":
        if token is not _LEARNING_SNAPSHOT_TOKEN:
            raise TypeError(
                "LearningSnapshot instances are created only by learn_utilities"
            )
        snapshot = object.__new__(cls)
        object.__setattr__(snapshot, "policy", policy)
        object.__setattr__(snapshot, "feature_estimates", feature_estimates)
        object.__setattr__(snapshot, "id_local_estimates", id_local_estimates)
        object.__setattr__(snapshot, "feature_utilities", feature_utilities)
        object.__setattr__(snapshot, "id_local_utilities", id_local_utilities)
        snapshot._validate(estimated_timestamp)
        return snapshot

    def _validate(self, estimated_timestamp: Any) -> None:
        if not isinstance(self.policy, LearningPolicy):
            raise TypeError("policy must be a LearningPolicy")
        feature_estimates = tuple(self.feature_estimates)
        id_local_estimates = tuple(self.id_local_estimates)
        if any(not isinstance(item, UtilityEstimate) for item in feature_estimates):
            raise TypeError("feature_estimates must contain UtilityEstimate records")
        if any(
            not isinstance(item, IDLocalUtilityEstimate) for item in id_local_estimates
        ):
            raise TypeError(
                "id_local_estimates must contain IDLocalUtilityEstimate records"
            )
        if not self.policy.id_local_enabled and (
            id_local_estimates or self.id_local_utilities
        ):
            raise ValueError(
                "ID-local estimates and utilities must be empty when disabled"
            )
        all_estimates = feature_estimates + id_local_estimates
        if all_estimates:
            if not isinstance(estimated_timestamp, str):
                raise ValueError("nonempty snapshots require one learning timestamp")
            if any(
                estimate.estimated_timestamp != estimated_timestamp
                for estimate in all_estimates
            ):
                raise ValueError(
                    "all snapshot estimates must share the injected learning timestamp"
                )
        elif estimated_timestamp is not None:
            raise ValueError("empty snapshots must not consume a learning timestamp")
        feature_targets: Dict[Tuple[str, str], UtilityEstimate] = {}
        expected_provenance = "learning:{}".format(
            self.policy.credit_assignment_version
        )
        for estimate in feature_estimates:
            UtilityEstimate(
                utility_estimate_id=estimate.utility_estimate_id,
                task_family_id=estimate.task_family_id,
                context_attributes=estimate.context_attributes,
                estimated_utility=estimate.estimated_utility,
                confidence=estimate.confidence,
                source_event_ids=estimate.source_event_ids,
                estimator_version=estimate.estimator_version,
                estimated_timestamp=estimate.estimated_timestamp,
                provenance=estimate.provenance,
                schema_version=estimate.schema_version,
            )
            if len(estimate.context_attributes) != 1:
                raise ValueError(
                    "feature estimates must contain exactly one context attribute"
                )
            feature = validate_reusable_feature(estimate.context_attributes[0])
            _validate_estimate_common(
                estimate.task_family_id,
                estimate.estimated_utility,
                estimate.confidence,
                estimate.source_event_ids,
                estimate.estimator_version,
                estimate.provenance,
                self.policy,
                expected_provenance,
            )
            expected_confidence = len(estimate.source_event_ids) / (
                self.policy.prior_strength + len(estimate.source_event_ids)
            )
            if estimate.confidence != expected_confidence:
                raise ValueError(
                    "feature estimate confidence does not match evidence count"
                )
            target = (estimate.task_family_id, feature)
            if target in feature_targets:
                raise ValueError("duplicate feature estimate target")
            if estimate.utility_estimate_id != _identity(
                "feature-utility",
                self.policy,
                estimate.task_family_id,
                feature,
                tuple(estimate.source_event_ids),
            ):
                raise ValueError(
                    "feature estimate identity does not match its artifact"
                )
            feature_targets[target] = estimate

        id_targets: Dict[Tuple[str, str], IDLocalUtilityEstimate] = {}
        for estimate in id_local_estimates:
            IDLocalUtilityEstimate(
                id_local_utility_estimate_id=estimate.id_local_utility_estimate_id,
                task_family_id=estimate.task_family_id,
                context_item_id=estimate.context_item_id,
                estimated_utility=estimate.estimated_utility,
                confidence=estimate.confidence,
                source_event_ids=estimate.source_event_ids,
                estimator_version=estimate.estimator_version,
                estimated_timestamp=estimate.estimated_timestamp,
                provenance=estimate.provenance,
            )
            _nonempty("context_item_id", estimate.context_item_id)
            _validate_estimate_common(
                estimate.task_family_id,
                estimate.estimated_utility,
                estimate.confidence,
                estimate.source_event_ids,
                estimate.estimator_version,
                estimate.provenance,
                self.policy,
                expected_provenance,
            )
            expected_confidence = len(estimate.source_event_ids) / (
                self.policy.prior_strength + len(estimate.source_event_ids)
            )
            if estimate.confidence != expected_confidence:
                raise ValueError(
                    "ID-local estimate confidence does not match evidence count"
                )
            target = (estimate.task_family_id, estimate.context_item_id)
            if target in id_targets:
                raise ValueError("duplicate ID-local estimate target")
            if estimate.id_local_utility_estimate_id != _identity(
                "id-local-utility",
                self.policy,
                estimate.task_family_id,
                estimate.context_item_id,
                tuple(estimate.source_event_ids),
            ):
                raise ValueError(
                    "ID-local estimate identity does not match its artifact"
                )
            id_targets[target] = estimate

        feature_utilities = _validated_utility_mapping(
            "feature_utilities", self.feature_utilities, validate_reusable_feature
        )
        id_local_utilities = _validated_utility_mapping(
            "id_local_utilities", self.id_local_utilities, _validate_context_item_id
        )
        expected_features = _eligible_utilities(
            feature_targets, self.policy.minimum_evidence_count
        )
        expected_ids = _eligible_utilities(
            id_targets, self.policy.minimum_evidence_count
        )
        if feature_utilities != expected_features:
            raise ValueError(
                "feature_utilities must correspond exactly to eligible feature estimates"
            )
        if id_local_utilities != expected_ids:
            raise ValueError(
                "id_local_utilities must correspond exactly to eligible ID-local estimates"
            )
        object.__setattr__(self, "feature_estimates", feature_estimates)
        object.__setattr__(self, "id_local_estimates", id_local_estimates)
        object.__setattr__(
            self,
            "feature_utilities",
            _freeze_nested_float_mapping(feature_utilities),
        )
        object.__setattr__(
            self,
            "id_local_utilities",
            _freeze_nested_float_mapping(id_local_utilities),
        )

    def feature_utilities_for(self, task_family_id: str) -> Mapping[str, float]:
        _nonempty("task_family_id", task_family_id)
        return self.feature_utilities.get(task_family_id, MappingProxyType({}))

    def id_local_utilities_for(self, task_family_id: str) -> Mapping[str, float]:
        _nonempty("task_family_id", task_family_id)
        return self.id_local_utilities.get(task_family_id, MappingProxyType({}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_estimates": [item.to_dict() for item in self.feature_estimates],
            "feature_utilities": {
                family: dict(self.feature_utilities[family])
                for family in sorted(self.feature_utilities)
            },
            "id_local_estimates": [item.to_dict() for item in self.id_local_estimates],
            "id_local_utilities": {
                family: dict(self.id_local_utilities[family])
                for family in sorted(self.id_local_utilities)
            },
            "policy": self.policy.to_dict(),
        }


@dataclass
class _Evidence:
    rewards: List[float] = field(default_factory=list)
    source_event_ids: List[str] = field(default_factory=list)


def _identity(
    kind: str,
    policy: LearningPolicy,
    family: str,
    target: str,
    source_event_ids: Tuple[str, ...],
) -> str:
    payload = {
        "family": family,
        "kind": kind,
        "policy": policy.to_dict(),
        "source_event_ids": list(source_event_ids),
        "target": target,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "{}-{}".format(kind, hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def _structured_item_rewards(
    event: FeedbackEvent, known_ids: FrozenSet[str]
) -> Dict[str, float]:
    payload = event.structured_value
    if not isinstance(payload, Mapping):
        raise ValueError("structured context_utility must be an object")
    if frozenset(payload) != _STRUCTURED_CONTEXT_UTILITY_KEYS:
        raise ValueError("structured context_utility has incorrect keys")
    if payload["locked"] is not True:
        raise ValueError("structured context_utility must be locked")
    if payload["selector_independent"] is not True:
        raise ValueError("structured context_utility must be selector_independent")
    for key in _ATTRIBUTE_DIAGNOSTIC_KEYS:
        _string_tuple(key, payload[key])
    _nonempty("shared_feature_trap", payload["shared_feature_trap"])
    useful = _string_tuple(
        "useful_context_item_ids", payload["useful_context_item_ids"]
    )
    harmful = _string_tuple(
        "harmful_context_item_ids", payload["harmful_context_item_ids"]
    )
    if set(useful).intersection(harmful):
        raise ValueError("useful and harmful context item IDs must be disjoint")
    structured_ids = set(useful).union(harmful)
    if structured_ids != set(event.affected_context_item_ids):
        raise ValueError(
            "affected_context_item_ids must correspond exactly to structured IDs"
        )
    unknown = structured_ids.difference(known_ids)
    if unknown:
        raise ValueError("structured context_utility contains unknown context item IDs")
    return dict(
        [(context_id, 1.0) for context_id in useful]
        + [(context_id, -1.0) for context_id in harmful]
    )


def _item_rewards(
    event: FeedbackEvent, known_ids: FrozenSet[str], policy: LearningPolicy
) -> Dict[str, float]:
    if event.signal_type not in policy.accepted_signal_types:
        raise ValueError(
            "feedback signal_type is unsupported or not accepted by learning policy"
        )
    affected = tuple(event.affected_context_item_ids)
    if not affected:
        raise ValueError("affected_context_item_ids must not be empty")
    unknown = set(affected).difference(known_ids)
    if unknown:
        raise ValueError("feedback contains unknown affected context item IDs")

    if event.signal_type == "context_utility":
        if event.correction_category is not None or event.correction_text is not None:
            raise ValueError("correction-only fields must be null for context_utility")
        if event.structured_value is not None:
            if event.numeric_value is not None:
                raise ValueError(
                    "context_utility requires exactly one feedback value encoding"
                )
            return _structured_item_rewards(event, known_ids)
        if event.numeric_value is None:
            raise ValueError("numeric context_utility requires numeric_value")
        value = _finite("numeric_value", event.numeric_value)
        if not -1.0 <= value <= 1.0:
            raise ValueError("numeric_value must be between -1 and 1")
        return {context_id: value for context_id in affected}

    _nonempty("correction_category", event.correction_category)
    _nonempty("correction_text", event.correction_text)
    if event.structured_value is not None:
        raise ValueError("structured correction is unsupported")
    if event.numeric_value is None:
        raise ValueError("numeric correction requires numeric_value")
    value = _finite("numeric_value", event.numeric_value)
    if not -1.0 <= value <= 1.0:
        raise ValueError("numeric_value must be between -1 and 1")
    if value >= 0.0:
        raise ValueError("numeric correction must be strictly negative")
    return {context_id: value for context_id in affected}


def _smoothed(evidence: _Evidence, policy: LearningPolicy) -> Tuple[float, float]:
    count = len(evidence.rewards)
    denominator = policy.prior_strength + count
    utility = (
        policy.prior_mean * policy.prior_strength + math.fsum(evidence.rewards)
    ) / denominator
    confidence = count / denominator
    if not math.isfinite(utility) or not -1.0 <= utility <= 1.0:
        raise ValueError("smoothed utility must be finite and bounded")
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("estimate confidence must be finite and bounded")
    return utility, confidence


def learn_utilities(
    events: Sequence[FeedbackEvent],
    inputs_by_task_case_id: Mapping[str, TaskInputs],
    policy: LearningPolicy,
    clock: Callable[[], str],
) -> LearningSnapshot:
    """Learn feature and ID-local estimates from the exact supplied reveal prefix."""

    if not isinstance(policy, LearningPolicy):
        raise TypeError("policy must be a LearningPolicy")
    if not isinstance(inputs_by_task_case_id, Mapping):
        raise TypeError("inputs_by_task_case_id must be a mapping")
    if not callable(clock):
        raise TypeError("clock must be callable")
    try:
        supplied_events = tuple(events)
    except TypeError:
        raise TypeError("events must be a sequence of FeedbackEvent records")
    if any(not isinstance(event, FeedbackEvent) for event in supplied_events):
        raise TypeError("events must contain FeedbackEvent records")
    supplied_inputs = dict(inputs_by_task_case_id)
    if any(not isinstance(value, TaskInputs) for value in supplied_inputs.values()):
        raise TypeError("inputs_by_task_case_id must contain TaskInputs records")

    # Frozen records can still be altered through object.__setattr__. Round-trip every
    # caller record through its canonical schema boundary before assigning credit. This
    # preserves event order, does not mutate caller objects, and yields only the two
    # explicitly accepted public record types.
    copied_events = tuple(
        FeedbackEvent.from_dict(FeedbackEvent.to_dict(event))
        for event in supplied_events
    )
    copied_inputs = {
        task_case_id: TaskInputs.from_dict(TaskInputs.to_dict(value))
        for task_case_id, value in supplied_inputs.items()
    }
    event_ids = tuple(event.event_id for event in copied_events)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("feedback event IDs must be unique")

    feature_evidence: DefaultDict[Tuple[str, str], _Evidence] = defaultdict(_Evidence)
    id_evidence: DefaultDict[Tuple[str, str], _Evidence] = defaultdict(_Evidence)

    for event in copied_events:
        if event.task_case_id not in copied_inputs:
            raise ValueError(
                "exact task_case_id input is required for every feedback event"
            )
        inputs = copied_inputs[event.task_case_id]
        family = inputs.profile.task_family_id
        if event.task_family_id != family:
            raise ValueError(
                "feedback task family must match TaskInputs profile family"
            )
        candidate_ids = tuple(item.context_item_id for item in inputs.candidate_context)
        known_ids = frozenset(candidate_ids)
        items_by_id = {item.context_item_id: item for item in inputs.candidate_context}
        rewards = _item_rewards(event, known_ids, policy)

        event_feature_rewards: DefaultDict[str, List[float]] = defaultdict(list)
        for context_item_id, reward in rewards.items():
            for feature in reusable_features(
                items_by_id[context_item_id], candidate_ids
            ):
                event_feature_rewards[feature].append(reward)
            if policy.id_local_enabled:
                evidence = id_evidence[(family, context_item_id)]
                evidence.rewards.append(reward)
                evidence.source_event_ids.append(event.event_id)
        for feature, item_rewards in event_feature_rewards.items():
            evidence = feature_evidence[(family, feature)]
            evidence.rewards.append(math.fsum(item_rewards) / len(item_rewards))
            evidence.source_event_ids.append(event.event_id)

    if not feature_evidence and not id_evidence:
        return LearningSnapshot._from_learning(
            _LEARNING_SNAPSHOT_TOKEN,
            policy=policy,
            feature_estimates=(),
            id_local_estimates=(),
            feature_utilities={},
            id_local_utilities={},
            estimated_timestamp=None,
        )

    estimated_timestamp = clock()
    feature_estimates = []
    feature_utilities: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    provenance = "learning:{}".format(policy.credit_assignment_version)
    for (family, feature), evidence in sorted(feature_evidence.items()):
        utility, confidence = _smoothed(evidence, policy)
        source_ids = tuple(evidence.source_event_ids)
        estimate = UtilityEstimate(
            utility_estimate_id=_identity(
                "feature-utility", policy, family, feature, source_ids
            ),
            task_family_id=family,
            context_attributes=(feature,),
            estimated_utility=utility,
            confidence=confidence,
            source_event_ids=source_ids,
            estimator_version=policy.estimator_version,
            estimated_timestamp=estimated_timestamp,
            provenance=provenance,
        )
        feature_estimates.append(estimate)
        if len(source_ids) >= policy.minimum_evidence_count:
            feature_utilities[family][feature] = utility

    id_local_estimates = []
    id_local_utilities: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
    for (family, context_item_id), evidence in sorted(id_evidence.items()):
        utility, confidence = _smoothed(evidence, policy)
        source_ids = tuple(evidence.source_event_ids)
        estimate = IDLocalUtilityEstimate(
            id_local_utility_estimate_id=_identity(
                "id-local-utility", policy, family, context_item_id, source_ids
            ),
            task_family_id=family,
            context_item_id=context_item_id,
            estimated_utility=utility,
            confidence=confidence,
            source_event_ids=source_ids,
            estimator_version=policy.estimator_version,
            estimated_timestamp=estimated_timestamp,
            provenance=provenance,
        )
        id_local_estimates.append(estimate)
        if len(source_ids) >= policy.minimum_evidence_count:
            id_local_utilities[family][context_item_id] = utility

    return LearningSnapshot._from_learning(
        _LEARNING_SNAPSHOT_TOKEN,
        policy=policy,
        feature_estimates=tuple(feature_estimates),
        id_local_estimates=tuple(id_local_estimates),
        feature_utilities=feature_utilities,
        id_local_utilities=id_local_utilities,
        estimated_timestamp=estimated_timestamp,
    )


__all__ = [
    "IDLocalUtilityEstimate",
    "LearningPolicy",
    "LearningSnapshot",
    "learn_utilities",
]
