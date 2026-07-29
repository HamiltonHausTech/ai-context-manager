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
        if not isinstance(key, str) or not key.strip():
            raise ValueError("feature weight names must be nonempty strings")
        frozen[key] = _finite("feature weight", value)
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _visible_features(item: ContextItem) -> Tuple[str, ...]:
    """Extract reusable visible features, deliberately excluding item IDs/provenance."""

    features = ["source:{}".format(item.source)]
    if item.confidence < 1.0 / 3.0:
        features.append("confidence:low")
    elif item.confidence < 2.0 / 3.0:
        features.append("confidence:medium")
    else:
        features.append("confidence:high")

    def visit(path: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit("{}.{}".format(path, key) if path else key, value[key])
        elif isinstance(value, tuple):
            for member in value:
                visit(path, member)
        elif value is not None:
            rendered = str(value).casefold() if isinstance(value, bool) else str(value)
            features.append("metadata.{}:{}".format(path, rendered))
            if isinstance(value, str):
                # Ontology values are useful as policy keys independent of storage path.
                features.append(value)

    visit("", item.metadata)
    return tuple(sorted(set(features)))


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

    def _score_details(self, item: ContextItem) -> Tuple[float, Dict[str, ScoreValue]]:
        factors: Dict[str, ScoreValue] = {}
        score = self.feature_weights.get("confidence", 0.0) * item.confidence
        factors["static.confidence"] = item.confidence
        factors["static.weight.confidence"] = self.feature_weights.get(
            "confidence", 0.0
        )
        for feature in _visible_features(item):
            weight = self.feature_weights.get(feature, 0.0)
            if weight:
                factors["static.feature.{}".format(feature)] = weight
                score += weight
        factors["static.score"] = score
        return score, factors

    def _rank(
        self,
        pipeline: RetrievalPipeline,
        eligible: Sequence[ContextComponent],
        request: RetrievalRequest,
        initial: list,
    ) -> Sequence[tuple]:
        details = {}
        rankable = []
        for component in eligible:
            try:
                if not isinstance(component, _ContextItemComponent):
                    raise TypeError("unexpected component type")
                details[id(component)] = self._score_details(component.item)
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
        pipeline.score_component = lambda component: details[id(component)][0]
        ranked = pipeline.rank_candidates(rankable, request)
        enriched = []
        for ranked_item in ranked:
            component, score, factors = ranked_item
            merged = dict(factors)
            merged.update(details[id(component)][1])
            enriched.append((component, score, merged))
        return enriched

    def select(self, inputs: TaskInputs) -> SelectionResult:
        self._require_inputs(inputs)
        components = tuple(
            _ContextItemComponent(item) for item in inputs.candidate_context
        )
        pipeline = self._pipeline(
            lambda component: self._score_details(component.item)[0]  # type: ignore[attr-defined]
        )
        request = RetrievalRequest(
            query=inputs.task_prompt,
            token_budget=inputs.token_budget,
            relevance_weight=self.relevance_weight,
            importance_weight=self.importance_weight,
            recency_weight=0.0,
        )
        eligible, initial = pipeline.select_candidates(components, request)
        ranked = self._rank(pipeline, eligible, request, initial)
        result = pipeline.pack_budget(ranked, request, initial)
        return self._finalize(inputs, components, eligible, result)


class AdaptivePolicySelector(StaticPolicySelector):
    """Static selection plus caller-supplied reusable-feature utility estimates.

    Learning and feedback processing are intentionally out of scope.  A mapping is
    copied at construction; a callback is queried only by reusable visible feature name.
    Context-item IDs are never supplied to either utility interface.
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
            copied = {
                key: _finite("utility estimate", value)
                for key, value in utility_estimates.items()
                if isinstance(key, str) and key.strip()
            }
            if len(copied) != len(utility_estimates):
                raise ValueError("utility feature names must be nonempty strings")
            self._utility_estimates = MappingProxyType(
                {key: copied[key] for key in sorted(copied)}
            )
        elif callable(utility_estimates):
            self._utility_estimates = utility_estimates
        else:
            raise ValueError("utility_estimates must be a mapping or callback")

    def _utility(self, feature: str) -> float:
        if self._utility_estimates is None:
            return 0.0
        if isinstance(self._utility_estimates, Mapping):
            return self._utility_estimates.get(feature, 0.0)
        return _finite("utility callback result", self._utility_estimates(feature))

    def _score_details(self, item: ContextItem) -> Tuple[float, Dict[str, ScoreValue]]:
        static_score, factors = super()._score_details(item)
        utility = 0.0
        for feature in _visible_features(item):
            estimate = self._utility(feature)
            if estimate:
                factors["adaptive.utility.{}".format(feature)] = estimate
                utility += estimate
        factors["adaptive.utility_total"] = utility
        factors["adaptive.learning_weight"] = self.learning_weight
        score = static_score + self.learning_weight * utility
        factors["adaptive.score"] = score
        return score, factors


__all__ = [
    "AdaptivePolicySelector",
    "FullContextSelector",
    "SelectionResult",
    "SelectorDecision",
    "SimilarityTopKSelector",
    "StaticPolicySelector",
]
