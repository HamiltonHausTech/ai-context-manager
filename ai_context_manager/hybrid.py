"""Backend-independent score normalization and hybrid ranking."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Dict, Iterable, List, Optional


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class HybridWeights:
    semantic: float = 0.65
    importance: float = 0.20
    recency: float = 0.10
    feedback: float = 0.05

    def normalized(self) -> "HybridWeights":
        values = [self.semantic, self.importance, self.recency, self.feedback]
        if any(value < 0 for value in values):
            raise ValueError("Hybrid weights cannot be negative")
        total = sum(values)
        if total <= 0:
            raise ValueError("At least one hybrid weight must be positive")
        return HybridWeights(*(value / total for value in values))


def normalize_similarity(value: float) -> float:
    """Normalize cosine similarity to the shared [0, 1] score range."""
    return _clamp(value)


def normalize_importance(value: float) -> float:
    value = max(0.0, float(value))
    return value / (1.0 + value)


def normalize_feedback(value: float) -> float:
    # Feedback is allowed to be signed; map it monotonically onto [0, 1].
    return 0.5 + (math.tanh(float(value)) / 2.0)


def normalize_recency(
    timestamp: Optional[str], now: Optional[datetime] = None, half_life_days: float = 30.0
) -> float:
    if not timestamp or half_life_days <= 0:
        return 0.0
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (current - parsed).total_seconds() / 86400.0)
        return 0.5 ** (age_days / half_life_days)
    except (TypeError, ValueError):
        return 0.0


def hybrid_score(
    record: Dict[str, Any],
    weights: Optional[HybridWeights] = None,
    now: Optional[datetime] = None,
) -> float:
    normalized = (weights or HybridWeights()).normalized()
    semantic = normalize_similarity(record.get("similarity_score", 0.0))
    importance = normalize_importance(record.get("score", 0.0))
    recency = normalize_recency(record.get("timestamp"), now=now)
    feedback = normalize_feedback(record.get("feedback_score", 0.0))
    return _clamp(
        semantic * normalized.semantic
        + importance * normalized.importance
        + recency * normalized.recency
        + feedback * normalized.feedback
    )


def rank_hybrid(
    records: Iterable[Dict[str, Any]],
    weights: Optional[HybridWeights] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    ranked = []
    for record in records:
        enriched = dict(record)
        enriched["hybrid_score"] = hybrid_score(enriched, weights, now)
        ranked.append(enriched)
    return sorted(ranked, key=lambda item: item["hybrid_score"], reverse=True)
