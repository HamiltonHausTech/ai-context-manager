"""Repeatable retrieval evaluation with relevance and efficiency metrics."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    query: str
    relevant_ids: List[str]
    excluded_ids: List[str] = field(default_factory=list)
    relevance: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationDataset:
    name: str
    cases: List[EvaluationCase]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        return cls(
            name=data["name"],
            cases=[EvaluationCase(**case) for case in data["cases"]],
        )

    @classmethod
    def load(cls, path: str) -> "EvaluationDataset":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class CaseMetrics:
    precision: float
    recall: float
    reciprocal_rank: float
    ndcg: float
    token_efficiency: float
    exclusion_accuracy: float
    context_stability: float = 1.0
    downstream_utility: float = 0.0


@dataclass(frozen=True)
class EvaluationReport:
    dataset: str
    cases: Dict[str, CaseMetrics]

    @property
    def averages(self) -> Dict[str, float]:
        if not self.cases:
            return {}
        fields = CaseMetrics.__dataclass_fields__
        return {
            field_name: mean(
                getattr(metrics, field_name) for metrics in self.cases.values()
            )
            for field_name in fields
        }

    @property
    def utility_score(self) -> float:
        """Measured downstream scorer, or NDCG when no task scorer is supplied."""
        return self.averages.get("downstream_utility", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "cases": {key: asdict(value) for key, value in self.cases.items()},
            "averages": self.averages,
            "utility_score": self.utility_score,
        }


def _result_id(result: Any) -> str:
    return result if isinstance(result, str) else result["id"]


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def evaluate_ranking(
    case: EvaluationCase,
    results: List[Any],
    k: Optional[int] = None,
    context_stability: float = 1.0,
    downstream_utility: Optional[float] = None,
) -> CaseMetrics:
    selected = results[: k or len(results)]
    selected_ids = [_result_id(result) for result in selected]
    relevant = set(case.relevant_ids)
    hits = [result_id for result_id in selected_ids if result_id in relevant]
    precision = len(hits) / len(selected_ids) if selected_ids else 0.0
    recall = len(hits) / len(relevant) if relevant else 1.0
    reciprocal_rank = next(
        (1.0 / index for index, result_id in enumerate(selected_ids, 1) if result_id in relevant),
        0.0,
    )

    gains = [
        case.relevance.get(result_id, 1.0 if result_id in relevant else 0.0)
        for result_id in selected_ids
    ]
    dcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(gains, 1))
    ideal_gains = sorted(
        [case.relevance.get(item, 1.0) for item in case.relevant_ids], reverse=True
    )[: len(selected_ids)]
    ideal_dcg = sum(
        gain / math.log2(index + 1) for index, gain in enumerate(ideal_gains, 1)
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else 1.0

    token_counts = [
        float(result.get("tokens", 1.0)) if isinstance(result, dict) else 1.0
        for result in selected
    ]
    total_tokens = sum(token_counts)
    relevant_tokens = sum(
        tokens
        for result_id, tokens in zip(selected_ids, token_counts)
        if result_id in relevant
    )
    token_efficiency = relevant_tokens / total_tokens if total_tokens else 0.0
    excluded = set(case.excluded_ids)
    exclusion_accuracy = (
        len(excluded - set(selected_ids)) / len(excluded) if excluded else 1.0
    )
    return CaseMetrics(
        precision,
        recall,
        reciprocal_rank,
        ndcg,
        token_efficiency,
        exclusion_accuracy,
        context_stability,
        ndcg if downstream_utility is None else downstream_utility,
    )


class RetrievalEvaluator:
    def __init__(
        self,
        search: Callable[[str, int], List[Any]],
        utility_scorer: Optional[Callable[[EvaluationCase, List[Any]], float]] = None,
    ):
        self.search = search
        self.utility_scorer = utility_scorer

    def evaluate(
        self,
        dataset: EvaluationDataset,
        k: int = 5,
        stability_runs: int = 2,
    ) -> EvaluationReport:
        case_metrics = {}
        for case in dataset.cases:
            runs = [self.search(case.query, k) for _ in range(max(1, stability_runs))]
            first_ids = [_result_id(result) for result in runs[0]]
            stability = mean(
                _jaccard(first_ids, [_result_id(result) for result in run])
                for run in runs[1:]
            ) if len(runs) > 1 else 1.0
            case_metrics[case.id] = evaluate_ranking(
                case,
                runs[0],
                k=k,
                context_stability=stability,
                downstream_utility=(
                    self.utility_scorer(case, runs[0])
                    if self.utility_scorer
                    else None
                ),
            )
        return EvaluationReport(dataset.name, case_metrics)
