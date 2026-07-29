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
from .selectors import reusable_features

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
        if not accepted or not accepted.issubset(_ALLOWED_SIGNAL_TYPES):
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


@dataclass(frozen=True)
class LearningSnapshot:
    """One immutable result for an exact caller-ordered reveal prefix.

    The outer mapping key is always ``task_family_id``. Inner feature mappings are the
    only mappings intended for ``AdaptivePolicySelector``. ID-local mappings remain a
    separate ablation and can never enter the selector through this API accidentally.
    """

    policy: LearningPolicy
    feature_estimates: Tuple[UtilityEstimate, ...]
    id_local_estimates: Tuple[IDLocalUtilityEstimate, ...]
    feature_utilities: Mapping[str, Mapping[str, float]]
    id_local_utilities: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "feature_estimates", feature_estimates)
        object.__setattr__(self, "id_local_estimates", id_local_estimates)
        object.__setattr__(
            self,
            "feature_utilities",
            _freeze_nested_float_mapping(self.feature_utilities),
        )
        object.__setattr__(
            self,
            "id_local_utilities",
            _freeze_nested_float_mapping(self.id_local_utilities),
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
        if event.structured_value is not None:
            return _structured_item_rewards(event, known_ids)
        if event.numeric_value is None:
            raise ValueError("numeric context_utility requires numeric_value")
        return {context_id: float(event.numeric_value) for context_id in affected}

    if event.structured_value is not None:
        raise ValueError("structured correction is unsupported")
    if event.numeric_value is None:
        raise ValueError("numeric correction requires numeric_value")
    value = float(event.numeric_value)
    if value >= 0.0:
        raise ValueError("numeric correction must be strictly negative")
    # FeedbackEvent construction already requires correction fields. Their text is never
    # interpreted; only the signed numeric value participates in credit assignment.
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
        copied_events = tuple(events)
    except TypeError:
        raise TypeError("events must be a sequence of FeedbackEvent records")
    if any(not isinstance(event, FeedbackEvent) for event in copied_events):
        raise TypeError("events must contain FeedbackEvent records")
    copied_inputs = dict(inputs_by_task_case_id)
    if any(not isinstance(value, TaskInputs) for value in copied_inputs.values()):
        raise TypeError("inputs_by_task_case_id must contain TaskInputs records")
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
        return LearningSnapshot(policy, (), (), {}, {})

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

    return LearningSnapshot(
        policy=policy,
        feature_estimates=tuple(feature_estimates),
        id_local_estimates=tuple(id_local_estimates),
        feature_utilities=feature_utilities,
        id_local_utilities=id_local_utilities,
    )


__all__ = [
    "IDLocalUtilityEstimate",
    "LearningPolicy",
    "LearningSnapshot",
    "learn_utilities",
]
