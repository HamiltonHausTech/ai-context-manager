"""Comparable, leakage-resistant policies for adaptive-selection experiments.

Every public selector accepts only :class:`TaskInputs`.  All policies adapt the same
candidate sequence to ``RetrievalPipeline`` components, run its eligibility stage, and
use its budget packer with each ``ContextItem.token_count`` as the authoritative count.
The full-context reference uses deterministic candidate-order greedy packing: an item
that does not fit is excluded and later, smaller items may still be selected.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ai_context_manager.components import ContextComponent
from ai_context_manager.retrieval import (
    RetrievalDecision,
    RetrievalPipeline,
    RetrievalRequest,
)

from .schema import ContextItem, TaskInputs

ScoreValue = Union[bool, int, float, str]
UtilityProvider = Union[Mapping[str, float], Callable[[str], float]]


def _finite(name: str, value: Any, *, nonnegative: bool = False) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (nonnegative and float(value) < 0.0)
    ):
        suffix = " nonnegative and" if nonnegative else ""
        raise ValueError("{} must be{} finite".format(name, suffix))
    return float(value)


def _freeze_factors(values: Mapping[str, ScoreValue]) -> Mapping[str, ScoreValue]:
    return MappingProxyType({key: values[key] for key in sorted(values)})


def _freeze_weights(values: Optional[Mapping[str, float]]) -> Mapping[str, float]:
    if values is None:
        values = {"confidence": 1.0}
    if not isinstance(values, Mapping):
        raise ValueError("feature_weights must be a mapping")
    frozen: Dict[str, float] = {}
    for key, value in values.items():
        if key != "confidence":
            _validate_feature_name(key, "feature weight")
        frozen[key] = _finite("feature weight", value)
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


_FEATURE_RE = re.compile(r"^[a-z][a-z0-9_-]*:[a-z0-9][a-z0-9_-]*$")
_RESERVED_FEATURE_TERMS = frozenset(
    {
        "adaptation",
        "feedback",
        "gold",
        "heldout",
        "irrelevant",
        "label",
        "misleading",
        "required",
        "useful",
    }
)
_FORBIDDEN_IDENTITY_TERMS = frozenset(
    {"candidate", "contextitemid", "metadata", "provenance", "secret", "source"}
)
_APPROVED_METADATA_FIELDS = (
    "control_attributes",
    "format",
    "learning_attributes",
)
_APPROVED_FEATURE_NAMESPACES = frozenset(
    {
        "action",
        "basis",
        "capability",
        "confidence",
        "context_role",
        "format",
        "memory_kind",
        "presentation",
        "recency",
        "relevance",
        "scope",
        "signal",
        "tag",
        "task_family",
    }
)


def _validate_feature_name(
    feature: Any, name: str = "reusable feature", candidate_ids: Sequence[str] = ()
) -> str:
    """Validate one corpus-independent, namespaced policy feature.

    Reusable features have a strict ``namespace:value`` grammar.  Evaluation-label
    vocabulary and strings containing any candidate ID are forbidden even in approved
    metadata fields.  The namespace allowlist is the frozen reusable-feature vocabulary
    for the Stage 0 ontology and planned policy features; all other namespaces are
    rejected rather than inferred.  Rejecting ambiguous values is safer than
    manufacturing a policy key from them.
    """

    if not isinstance(feature, str):
        raise ValueError("{} must be a safe namespaced feature".format(name))
    normalized = feature.casefold()
    screening_text = re.sub(r"[^a-z0-9]+", "", normalized)
    if any(term in screening_text for term in _RESERVED_FEATURE_TERMS):
        raise ValueError("{} contains reserved evaluation vocabulary".format(name))
    if any(term in screening_text for term in _FORBIDDEN_IDENTITY_TERMS) or re.search(
        r"(?:^|[^a-z0-9])i[^a-z0-9]*d(?:$|[^a-z0-9])", normalized
    ):
        raise ValueError("{} contains forbidden identity vocabulary".format(name))
    if not _FEATURE_RE.fullmatch(feature):
        raise ValueError("{} must be a safe namespaced feature".format(name))
    if normalized.split(":", 1)[0] not in _APPROVED_FEATURE_NAMESPACES:
        raise ValueError(
            "{} uses an unapproved reusable-feature namespace".format(name)
        )
    if any(candidate_id.casefold() in normalized for candidate_id in candidate_ids):
        raise ValueError("{} contains a candidate ID".format(name))
    return feature


def _visible_features(
    item: ContextItem, candidate_ids: Sequence[str]
) -> Tuple[str, ...]:
    """Extract only approved, reusable metadata values and confidence bins.

    Source, provenance, IDs, arbitrary metadata paths/scalars, and path-prefixed aliases
    are intentionally invisible.  Approved values must already be safe namespaced
    strings; malformed or evaluation-like values are rejected explicitly.
    """

    features = []
    if item.confidence < 1.0 / 3.0:
        features.append("confidence:low")
    elif item.confidence < 2.0 / 3.0:
        features.append("confidence:medium")
    else:
        features.append("confidence:high")

    for field in _APPROVED_METADATA_FIELDS:
        if field not in item.metadata:
            continue
        value = item.metadata[field]
        values = value if isinstance(value, tuple) else (value,)
        if not values or any(not isinstance(member, str) for member in values):
            raise ValueError("metadata.{} must contain feature strings".format(field))
        for member in values:
            validated = _validate_feature_name(
                member, "metadata.{} feature".format(field), candidate_ids
            )
            if field == "format" and not validated.startswith("format:"):
                raise ValueError("metadata.format must use the format namespace")
            features.append(validated)
    return tuple(sorted(set(features)))


def _normalize_pool_scores(raw_scores: Sequence[float]) -> Tuple[float, ...]:
    """Dense-rank a complete finite candidate pool onto ``[0, 1]``.

    Ordinal normalization intentionally preserves ordering rather than score magnitude:
    equal raw scores receive equal importance and distinct raw scores receive strictly
    increasing representable values, even when finite extrema erase min-max differences.
    A single unique value maps to neutral ``0.5``; stable retrieval sorting preserves
    candidate order for ties.
    """

    if not raw_scores:
        return ()
    unique_scores = sorted(set(raw_scores))
    if len(unique_scores) == 1:
        return tuple(0.5 for _score in raw_scores)
    denominator = len(unique_scores) - 1
    importance_by_score = {
        raw_score: rank / denominator for rank, raw_score in enumerate(unique_scores)
    }
    return tuple(importance_by_score[raw_score] for raw_score in raw_scores)


class _ContextItemComponent(ContextComponent):
    def __init__(self, item: ContextItem):
        super().__init__(item.context_item_id)
        self.item = item

    def load_content(self) -> str:
        return self.item.content


@dataclass(frozen=True)
class SelectorDecision:
    """One immutable trace decision for one input candidate."""

    context_item_id: str
    included: bool
    reason: str
    detail: str
    score: float
    token_count: int
    score_factors: Mapping[str, ScoreValue]


@dataclass(frozen=True)
class SelectionResult:
    """Selected records and a complete candidate-order decision trace."""

    selector_mode: str
    selected_items: Tuple[ContextItem, ...]
    decisions: Tuple[SelectorDecision, ...]
    eligible_context_item_ids: Tuple[str, ...]
    used_tokens: int
    token_budget: int

    @property
    def policy_signature(self) -> Tuple[Any, ...]:
        """Mode-independent mechanics signature used to compare equivalent policies."""

        return (
            tuple(item.context_item_id for item in self.selected_items),
            tuple(
                (decision.context_item_id, decision.included, decision.reason)
                for decision in self.decisions
            ),
            self.used_tokens,
            self.token_budget,
        )


class _BaseSelector:
    mode = "base"

    @staticmethod
    def _require_inputs(inputs: TaskInputs) -> None:
        if not isinstance(inputs, TaskInputs):
            raise TypeError("selector input must be a TaskInputs record")

    @staticmethod
    def _token_counter(component: ContextComponent, content: str) -> int:
        adapted = component
        if not isinstance(adapted, _ContextItemComponent):
            raise TypeError("unexpected component type")
        if content != adapted.item.content:
            raise ValueError("authoritative count applies only to original content")
        return adapted.item.token_count

    def _pipeline(
        self, score: Callable[[ContextComponent], float]
    ) -> RetrievalPipeline:
        return RetrievalPipeline(score, token_counter=self._token_counter)

    def _finalize(
        self,
        inputs: TaskInputs,
        components: Sequence[_ContextItemComponent],
        eligible: Sequence[ContextComponent],
        retrieval_result: Any,
    ) -> SelectionResult:
        decisions = []
        selected_items = tuple(
            component.item
            for component in (item.component for item in retrieval_result.items)
            if isinstance(component, _ContextItemComponent)
        )
        for component in components:
            matches = [
                decision
                for decision in retrieval_result.decisions
                if decision.component_id == component.id
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "every candidate must have exactly one retrieval decision"
                )
            raw = matches[0]
            if raw.included:
                reason = "selected"
            elif raw.reason in ("over_budget", "budget_exhausted"):
                reason = "budget_exclusion"
            elif raw.reason == "max_components":
                reason = "k_exclusion"
            elif raw.reason == "processing_error":
                reason = "processing_error"
            else:
                reason = "filtered"
            factors = dict(raw.score_factors)
            factors["selector_mode"] = self.mode
            decisions.append(
                SelectorDecision(
                    context_item_id=component.id,
                    included=raw.included,
                    reason=reason,
                    detail=raw.reason,
                    score=float(raw.score or 0.0),
                    token_count=component.item.token_count,
                    score_factors=_freeze_factors(factors),
                )
            )
        if retrieval_result.used_tokens != sum(
            item.token_count for item in selected_items
        ):
            raise RuntimeError(
                "retrieval result violated authoritative token accounting"
            )
        return SelectionResult(
            selector_mode=self.mode,
            selected_items=selected_items,
            decisions=tuple(decisions),
            eligible_context_item_ids=tuple(component.id for component in eligible),
            used_tokens=retrieval_result.used_tokens,
            token_budget=inputs.token_budget,
        )


class FullContextSelector(_BaseSelector):
    """High-context reference with candidate-order greedy exact-budget packing."""

    mode = "full_context"

    def select(self, inputs: TaskInputs) -> SelectionResult:
        self._require_inputs(inputs)
        components = tuple(
            _ContextItemComponent(item) for item in inputs.candidate_context
        )
        pipeline = self._pipeline(lambda _component: 0.0)
        request = RetrievalRequest(token_budget=inputs.token_budget)
        eligible, initial = pipeline.select_candidates(components, request)
        ranked = [
            (component, 0.0, {"packing_order": float(index)})
            for index, component in enumerate(eligible)
        ]
        result = pipeline.pack_budget(ranked, request, initial)
        return self._finalize(inputs, components, eligible, result)


class SimilarityTopKSelector(_BaseSelector):
    """Lexical-similarity top-K policy followed by exact-budget packing."""

    mode = "similarity_top_k"

    def __init__(self, k: int = 5):
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k

    def select(self, inputs: TaskInputs) -> SelectionResult:
        self._require_inputs(inputs)
        components = tuple(
            _ContextItemComponent(item) for item in inputs.candidate_context
        )
        pipeline = self._pipeline(lambda _component: 0.0)
        request = RetrievalRequest(
            query=inputs.task_prompt,
            token_budget=inputs.token_budget,
            relevance_weight=1.0,
            importance_weight=0.0,
            recency_weight=0.0,
        )
        eligible, initial = pipeline.select_candidates(components, request)
        ranked = pipeline.rank_candidates(eligible, request)
        top = ranked[: self.k]
        for component, score, factors in ranked[self.k :]:
            initial.append(
                RetrievalDecision(
                    component.id,
                    component.__class__.__name__,
                    False,
                    "max_components",
                    score,
                    self._token_counter(component, component.get_content()),
                    dict(factors),
                )
            )
        result = pipeline.pack_budget(top, request, initial)
        return self._finalize(inputs, components, eligible, result)


class StaticPolicySelector(_BaseSelector):
    """Frozen configurable policy using only prompt and visible item features."""

    mode = "static_policy"

    def __init__(
        self,
        feature_weights: Optional[Mapping[str, float]] = None,
        relevance_weight: float = 0.7,
        importance_weight: float = 0.3,
    ):
        self.feature_weights = _freeze_weights(feature_weights)
        self.relevance_weight = _finite(
            "relevance_weight", relevance_weight, nonnegative=True
        )
        self.importance_weight = _finite(
            "importance_weight", importance_weight, nonnegative=True
        )
        if self.relevance_weight + self.importance_weight <= 0.0:
            raise ValueError("at least one policy weight must be positive")

    def _score_details(
        self, item: ContextItem, candidate_ids: Sequence[str]
    ) -> Tuple[float, Dict[str, ScoreValue]]:
        factors: Dict[str, ScoreValue] = {}
        contributions = [self.feature_weights.get("confidence", 0.0) * item.confidence]
        factors["policy.static.confidence"] = item.confidence
        factors["policy.static.confidence_weight"] = self.feature_weights.get(
            "confidence", 0.0
        )
        for feature in _visible_features(item, candidate_ids):
            weight = self.feature_weights.get(feature, 0.0)
            if weight:
                factors["policy.static.feature_weight.{}".format(feature)] = weight
                contributions.append(weight)
        try:
            raw_score = _finite("raw policy score", math.fsum(contributions))
        except OverflowError as exc:
            raise ValueError("raw policy score must be finite") from exc
        factors["policy.static.raw_score"] = raw_score
        factors["policy.raw_score"] = raw_score
        return raw_score, factors

    def _rank(
        self,
        pipeline: RetrievalPipeline,
        eligible: Sequence[ContextComponent],
        request: RetrievalRequest,
        initial: list,
        candidate_ids: Sequence[str],
    ) -> Sequence[tuple]:
        details: Dict[int, Tuple[float, Dict[str, ScoreValue]]] = {}
        validated = []
        for component in eligible:
            try:
                if not isinstance(component, _ContextItemComponent):
                    raise TypeError("unexpected component type")
                _visible_features(component.item, candidate_ids)
                validated.append(component)
            except Exception:
                initial.append(
                    RetrievalDecision(
                        component.id,
                        component.__class__.__name__,
                        False,
                        "processing_error",
                        0.0,
                        None,
                        {},
                    )
                )
        rankable = []
        for component in validated:
            try:
                details[id(component)] = self._score_details(
                    component.item, candidate_ids
                )
                rankable.append(component)
            except Exception:
                initial.append(
                    RetrievalDecision(
                        component.id,
                        component.__class__.__name__,
                        False,
                        "processing_error",
                        0.0,
                        None,
                        {},
                    )
                )
        raw_scores = tuple(details[id(component)][0] for component in rankable)
        effective_importance = _normalize_pool_scores(raw_scores)
        pool_min = min(raw_scores) if raw_scores else 0.0
        pool_max = max(raw_scores) if raw_scores else 0.0
        normalized_by_component = {
            id(component): normalized
            for component, normalized in zip(rankable, effective_importance)
        }
        for component in rankable:
            factors = details[id(component)][1]
            factors["policy.normalization_method"] = "candidate_pool_dense_rank"
            factors["policy.pool_raw_min"] = pool_min
            factors["policy.pool_raw_max"] = pool_max
            factors["policy.effective_importance"] = normalized_by_component[
                id(component)
            ]
        pipeline.score_component = lambda component: normalized_by_component[
            id(component)
        ]
        ranked = pipeline.rank_candidates(rankable, request)
        enriched = []
        for ranked_item in ranked:
            component, score, factors = ranked_item
            merged = {}
            merged.update(details[id(component)][1])
            total_weight = (
                request.relevance_weight
                + request.importance_weight
                + request.recency_weight
            )
            merged["retrieval.relevance"] = factors["relevance"]
            merged["retrieval.relevance_method"] = factors["relevance_method"]
            merged["retrieval.importance"] = factors["importance"]
            merged["retrieval.recency"] = factors["recency"]
            merged["retrieval.relevance_weight"] = request.relevance_weight
            merged["retrieval.importance_weight"] = request.importance_weight
            merged["retrieval.recency_weight"] = request.recency_weight
            merged["retrieval.weighted_relevance"] = (
                factors["relevance"] * request.relevance_weight
            )
            merged["retrieval.weighted_importance"] = (
                factors["importance"] * request.importance_weight
            )
            merged["retrieval.weighted_recency"] = (
                factors["recency"] * request.recency_weight
            )
            merged["retrieval.total_weight"] = total_weight
            merged["retrieval.final_score"] = score
            enriched.append((component, score, merged))
        return enriched

    def select(self, inputs: TaskInputs) -> SelectionResult:
        self._require_inputs(inputs)
        components = tuple(
            _ContextItemComponent(item) for item in inputs.candidate_context
        )
        candidate_ids = tuple(item.context_item_id for item in inputs.candidate_context)
        pipeline = self._pipeline(
            lambda component: self._score_details(  # type: ignore[attr-defined]
                component.item, candidate_ids
            )[0]
        )
        request = RetrievalRequest(
            query=inputs.task_prompt,
            token_budget=inputs.token_budget,
            relevance_weight=self.relevance_weight,
            importance_weight=self.importance_weight,
            recency_weight=0.0,
        )
        eligible, initial = pipeline.select_candidates(components, request)
        ranked = self._rank(pipeline, eligible, request, initial, candidate_ids)
        result = pipeline.pack_budget(ranked, request, initial)
        return self._finalize(inputs, components, eligible, result)


class AdaptivePolicySelector(StaticPolicySelector):
    """Static selection plus caller-supplied reusable-feature utility estimates.

    Learning and feedback processing are intentionally out of scope.  A mapping is
    copied at construction; a callback is queried only once per reusable visible feature
    for the selector instance, with both successful and failed results cached. Context-item
    IDs are never supplied to either utility interface.
    """

    mode = "adaptive_policy"

    def __init__(
        self,
        utility_estimates: Optional[UtilityProvider] = None,
        learning_weight: float = 1.0,
        feature_weights: Optional[Mapping[str, float]] = None,
        relevance_weight: float = 0.7,
        importance_weight: float = 0.3,
    ):
        super().__init__(feature_weights, relevance_weight, importance_weight)
        self.learning_weight = _finite(
            "learning_weight", learning_weight, nonnegative=True
        )
        if utility_estimates is None:
            self._utility_estimates: Optional[UtilityProvider] = None
        elif isinstance(utility_estimates, Mapping):
            copied = {}
            for key, value in utility_estimates.items():
                validated = _validate_feature_name(key, "utility feature")
                copied[validated] = _finite("utility estimate", value)
            self._utility_estimates = MappingProxyType(
                {key: copied[key] for key in sorted(copied)}
            )
        elif callable(utility_estimates):
            self._utility_estimates = utility_estimates
        else:
            raise ValueError("utility_estimates must be a mapping or callback")
        self._utility_cache: Dict[str, Tuple[bool, float]] = {}

    def _utility(self, feature: str) -> float:
        if self._utility_estimates is None:
            return 0.0
        if isinstance(self._utility_estimates, Mapping):
            return self._utility_estimates.get(feature, 0.0)
        cached = self._utility_cache.get(feature)
        if cached is None:
            try:
                value = _finite(
                    "utility callback result", self._utility_estimates(feature)
                )
                cached = (True, value)
            except Exception:
                cached = (False, 0.0)
            self._utility_cache[feature] = cached
        if not cached[0]:
            raise ValueError("utility callback failed for reusable feature")
        return cached[1]

    def _score_details(
        self, item: ContextItem, candidate_ids: Sequence[str]
    ) -> Tuple[float, Dict[str, ScoreValue]]:
        static_raw_score, factors = super()._score_details(item, candidate_ids)
        utility_contributions = []
        for feature in _visible_features(item, candidate_ids):
            estimate = self._utility(feature)
            if estimate:
                factors["policy.adaptive.feature_utility.{}".format(feature)] = estimate
                utility_contributions.append(estimate)
        try:
            utility = _finite("adaptive raw utility", math.fsum(utility_contributions))
        except OverflowError as exc:
            raise ValueError("adaptive raw utility must be finite") from exc
        contribution = _finite(
            "weighted utility contribution", self.learning_weight * utility
        )
        try:
            raw_score = _finite(
                "raw policy score", math.fsum((static_raw_score, contribution))
            )
        except OverflowError as exc:
            raise ValueError("raw policy score must be finite") from exc
        factors["policy.adaptive.raw_utility"] = utility
        factors["policy.adaptive.learning_weight"] = self.learning_weight
        factors["policy.adaptive.weighted_utility_contribution"] = contribution
        factors["policy.adaptive.raw_score"] = raw_score
        factors["policy.raw_score"] = raw_score
        return raw_score, factors


__all__ = [
    "AdaptivePolicySelector",
    "FullContextSelector",
    "SelectionResult",
    "SelectorDecision",
    "SimilarityTopKSelector",
    "StaticPolicySelector",
]
