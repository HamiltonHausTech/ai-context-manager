"""Explicit, explainable stages for assembling agent context."""

from dataclasses import dataclass, field
import logging
import math
import re
from typing import Callable, List, Optional, Sequence

from ai_context_manager.components import ContextComponent, TaskSummaryComponent
from ai_context_manager.summarizers import Summarizer
from ai_context_manager.tokenization import estimate_tokens, truncate_to_token_budget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalRequest:
    query: Optional[str] = None
    required_terms: Optional[List[str]] = None
    include_tags: Optional[List[str]] = None
    component_types: Optional[List[str]] = None
    summarize_if_needed: bool = False
    token_budget: Optional[int] = None
    tag_match_mode: str = "any"
    task_id: Optional[str] = None
    include_inactive: bool = False
    min_relevance: float = 0.0
    relevance_weight: float = 0.70
    importance_weight: float = 0.25
    recency_weight: float = 0.05
    deduplicate: bool = False
    redundancy_threshold: float = 0.88
    max_components: Optional[int] = None

    def validate(self) -> None:
        if self.token_budget is not None and self.token_budget <= 0:
            raise ValueError("Token budget must be positive")
        if self.include_tags is not None and not isinstance(self.include_tags, list):
            raise ValueError("include_tags must be a list")
        if self.component_types is not None and not isinstance(self.component_types, list):
            raise ValueError("component_types must be a list")
        if self.required_terms is not None and not isinstance(self.required_terms, list):
            raise ValueError("required_terms must be a list")
        if self.tag_match_mode not in ("any", "all"):
            raise ValueError("tag_match_mode must be 'any' or 'all'")
        if not 0.0 <= self.min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")
        if not 0.0 <= self.redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be between 0 and 1")
        if self.max_components is not None and self.max_components <= 0:
            raise ValueError("max_components must be positive")
        if any(weight < 0 for weight in (
            self.relevance_weight, self.importance_weight, self.recency_weight
        )):
            raise ValueError("retrieval weights cannot be negative")
        if self.query and (
            self.relevance_weight + self.importance_weight + self.recency_weight <= 0
        ):
            raise ValueError("at least one retrieval weight must be positive")


@dataclass
class RetrievalItem:
    component: ContextComponent
    score: float
    content: str
    tokens: int
    summarized: bool = False
    score_factors: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        return {
            "id": self.component.id,
            "type": self.component.__class__.__name__,
            "tags": list(self.component.tags),
            "score": self.score,
            "tokens": self.tokens,
            "summarized": self.summarized,
            "content": self.content,
            "score_factors": dict(self.score_factors),
        }


@dataclass
class RetrievalDecision:
    component_id: str
    component_type: str
    included: bool
    reason: str
    score: Optional[float] = None
    tokens: Optional[int] = None
    score_factors: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    request: RetrievalRequest
    items: List[RetrievalItem] = field(default_factory=list)
    decisions: List[RetrievalDecision] = field(default_factory=list)
    used_tokens: int = 0

    @property
    def context(self) -> str:
        return "\n\n".join(item.content for item in self.items)

    def metadata(self) -> List[dict]:
        return [item.to_metadata() for item in self.items]


class RetrievalPipeline:
    """Candidate selection, ranking, and budget packing as separate stages."""

    def __init__(
        self,
        score_component: Callable[[ContextComponent], float],
        summarizer: Optional[Summarizer] = None,
        relevance_component: Optional[Callable[[str, ContextComponent], float]] = None,
        token_counter: Optional[Callable[[ContextComponent, str], int]] = None,
    ):
        self.score_component = score_component
        self.summarizer = summarizer
        self.relevance_component = relevance_component
        self.token_counter = token_counter

    def _count_tokens(self, component: ContextComponent, content: str) -> int:
        """Count tokens, allowing callers with authoritative accounting to opt in."""
        count = (
            self.token_counter(component, content)
            if self.token_counter is not None
            else estimate_tokens(content)
        )
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("token counter must return a nonnegative integer")
        return count

    _STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "was", "what", "when", "where", "which", "who", "with",
    }

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {
            term for term in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(term) > 1 and term not in cls._STOP_WORDS
        }

    @classmethod
    def lexical_relevance(cls, query: str, content: str) -> float:
        query_terms, content_terms = cls._terms(query), cls._terms(content)
        if not query_terms or not content_terms:
            return 0.0
        overlap = len(query_terms & content_terms)
        cosine = overlap / math.sqrt(len(query_terms) * len(content_terms))
        phrase_bonus = 0.15 if query.lower() in content.lower() else 0.0
        return min(1.0, cosine + phrase_bonus)

    @classmethod
    def _redundancy(cls, left: str, right: str) -> float:
        left_terms, right_terms = cls._terms(left), cls._terms(right)
        union = left_terms | right_terms
        return len(left_terms & right_terms) / len(union) if union else 0.0

    @staticmethod
    def _recency(component: ContextComponent) -> float:
        from ai_context_manager.hybrid import normalize_recency

        return normalize_recency(getattr(component, "timestamp", None))

    def select_candidates(
        self, components: Sequence[ContextComponent], request: RetrievalRequest
    ) -> tuple[List[ContextComponent], List[RetrievalDecision]]:
        candidates = []
        decisions = []
        for component in components:
            reason = None
            if not request.include_inactive and not component.memory.is_active():
                reason = f"lifecycle_{component.memory.status}"
                if component.memory.is_expired():
                    reason = "lifecycle_expired"
            elif request.include_tags and not component.matches_tags(
                request.include_tags, request.tag_match_mode
            ):
                reason = "tag_filter"
            elif request.component_types and component.__class__.__name__ not in request.component_types:
                reason = "type_filter"
            elif (
                request.task_id is not None
                and isinstance(component, TaskSummaryComponent)
                and component.id != request.task_id
            ):
                reason = "task_filter"

            if reason:
                decisions.append(
                    RetrievalDecision(
                        component.id, component.__class__.__name__, False, reason
                    )
                )
            else:
                candidates.append(component)
        return candidates, decisions

    def rank_candidates(
        self, components: Sequence[ContextComponent], request: Optional[RetrievalRequest] = None
    ) -> List[tuple]:
        request = request or RetrievalRequest()
        scored = []
        for component in components:
            try:
                importance = max(0.0, min(1.0, self.score_component(component)))
            except Exception as exc:
                logger.warning("Failed to score component %s: %s", component.id, exc)
                importance = 0.0
            if request.query:
                try:
                    relevance = (
                        self.relevance_component(request.query, component)
                        if self.relevance_component
                        else self.lexical_relevance(request.query, component.get_content())
                    )
                    relevance = max(0.0, min(1.0, float(relevance)))
                except Exception:
                    relevance = 0.0
                recency = self._recency(component)
                total_weight = (
                    request.relevance_weight
                    + request.importance_weight
                    + request.recency_weight
                )
                score = (
                    relevance * request.relevance_weight
                    + importance * request.importance_weight
                    + recency * request.recency_weight
                ) / total_weight
                factors = {
                    "relevance": relevance,
                    "relevance_method": (
                        "custom" if self.relevance_component else "lexical"
                    ),
                    "importance": importance,
                    "recency": recency,
                }
            else:
                score = self.score_component(component)
                factors = {"importance": importance}
            scored.append(
                (component, score, factors) if request.query else (component, score)
            )
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def pack_budget(
        self,
        ranked: Sequence[tuple],
        request: RetrievalRequest,
        initial_decisions: Optional[List[RetrievalDecision]] = None,
    ) -> RetrievalResult:
        result = RetrievalResult(request=request, decisions=list(initial_decisions or []))
        selected_content = []
        for ranked_item in ranked:
            component, score = ranked_item[:2]
            factors = ranked_item[2] if len(ranked_item) > 2 else {}
            try:
                content = component.get_content()
                token_count = self._count_tokens(component, content)
                summarized = False
                if request.required_terms:
                    content_terms = self._terms(content)
                    required = {
                        term for value in request.required_terms for term in self._terms(value)
                    }
                    required_match = bool(required & content_terms)
                    factors["required_match"] = 1.0 if required_match else 0.0
                    if not required_match:
                        result.decisions.append(
                            RetrievalDecision(
                                component.id, component.__class__.__name__, False,
                                "required_term_miss", score, token_count, factors,
                            )
                        )
                        continue
                relevance = factors.get("relevance")
                if relevance is not None and relevance < request.min_relevance:
                    result.decisions.append(
                        RetrievalDecision(
                            component.id, component.__class__.__name__, False,
                            "below_relevance_threshold", score, token_count, factors,
                        )
                    )
                    continue
                if request.deduplicate and any(
                    self._redundancy(content, previous) >= request.redundancy_threshold
                    for previous in selected_content
                ):
                    result.decisions.append(
                        RetrievalDecision(
                            component.id, component.__class__.__name__, False,
                            "redundant", score, token_count, factors,
                        )
                    )
                    continue
                if request.max_components is not None and len(result.items) >= request.max_components:
                    result.decisions.append(
                        RetrievalDecision(
                            component.id, component.__class__.__name__, False,
                            "max_components", score, token_count, factors,
                        )
                    )
                    continue
                remaining = (
                    request.token_budget - result.used_tokens
                    if request.token_budget is not None
                    else None
                )

                if remaining is not None and token_count > remaining:
                    if remaining <= 0:
                        result.decisions.append(
                            RetrievalDecision(
                                component.id,
                                component.__class__.__name__,
                                False,
                                "budget_exhausted",
                                score,
                                token_count,
                                factors,
                            )
                        )
                        continue
                    if not request.summarize_if_needed or self.summarizer is None:
                        result.decisions.append(
                            RetrievalDecision(
                                component.id,
                                component.__class__.__name__,
                                False,
                                "over_budget",
                                score,
                                token_count,
                                factors,
                            )
                        )
                        continue
                    compression_input = (
                        f"Current task: {request.query}\n"
                        "Compress the following while retaining only information useful "
                        f"to the current task:\n\n{content}"
                        if request.query else content
                    )
                    content = self.summarizer.summarize(compression_input, remaining)
                    content = truncate_to_token_budget(content, remaining)
                    token_count = self._count_tokens(component, content)
                    summarized = True
                    if not content or token_count > remaining:
                        result.decisions.append(
                            RetrievalDecision(
                                component.id,
                                component.__class__.__name__,
                                False,
                                "summarization_failed_budget",
                                score,
                                token_count,
                                factors,
                            )
                        )
                        continue

                item = RetrievalItem(
                    component, score, content, token_count, summarized, factors
                )
                result.items.append(item)
                selected_content.append(content)
                result.used_tokens += token_count
                result.decisions.append(
                    RetrievalDecision(
                        component.id,
                        component.__class__.__name__,
                        True,
                        "included_summarized" if summarized else "included",
                        score,
                        token_count,
                        factors,
                    )
                )
            except Exception as exc:
                logger.error("Failed to process component %s: %s", component.id, exc)
                result.decisions.append(
                    RetrievalDecision(
                        component.id,
                        component.__class__.__name__,
                        False,
                        "processing_error",
                        score,
                        score_factors=factors,
                    )
                )
        return result

    def retrieve(
        self, components: Sequence[ContextComponent], request: RetrievalRequest
    ) -> RetrievalResult:
        request.validate()
        candidates, decisions = self.select_candidates(components, request)
        ranked = self.rank_candidates(candidates, request)
        return self.pack_budget(ranked, request, decisions)
