from pathlib import Path

import pytest

from ai_context_manager.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    RetrievalEvaluator,
    evaluate_ranking,
)
from ai_context_manager.hybrid import HybridWeights, rank_hybrid


def test_checked_in_evaluation_dataset_loads():
    path = Path(__file__).parents[1] / "evaluations" / "agent_memory_retrieval.json"

    dataset = EvaluationDataset.load(str(path))

    assert dataset.name == "agent-memory-retrieval-v1"
    assert len(dataset.cases) == 3
    assert all(case.relevant_ids for case in dataset.cases)


def test_ranking_metrics_cover_relevance_cost_and_exclusions():
    case = EvaluationCase(
        "case",
        "query",
        relevant_ids=["best", "useful"],
        excluded_ids=["distractor"],
        relevance={"best": 2.0, "useful": 1.0},
    )
    results = [
        {"id": "best", "tokens": 10},
        {"id": "distractor", "tokens": 30},
        {"id": "useful", "tokens": 10},
    ]

    metrics = evaluate_ranking(case, results, k=3)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == 1.0
    assert metrics.reciprocal_rank == 1.0
    assert metrics.token_efficiency == pytest.approx(0.4)
    assert metrics.exclusion_accuracy == 0.0
    assert 0.0 < metrics.ndcg < 1.0


def test_evaluator_reports_context_stability():
    dataset = EvaluationDataset(
        "stability",
        [EvaluationCase("case", "query", ["a"])],
    )
    runs = iter([
        [{"id": "a"}, {"id": "b"}],
        [{"id": "a"}, {"id": "c"}],
    ])
    evaluator = RetrievalEvaluator(lambda _query, _k: next(runs))

    report = evaluator.evaluate(dataset, k=2, stability_runs=2)

    assert report.cases["case"].context_stability == pytest.approx(1 / 3)
    assert report.to_dict()["utility_score"] == report.utility_score


def test_feedback_improves_measured_retrieval_quality():
    dataset = EvaluationDataset(
        "feedback-comparison",
        [EvaluationCase("case", "bank rules", ["relevant"], ["distractor"])],
    )
    records = [
        {
            "id": "distractor",
            "similarity_score": 0.9,
            "score": 1.0,
            "feedback_score": 0.0,
        },
        {
            "id": "relevant",
            "similarity_score": 0.8,
            "score": 1.0,
            "feedback_score": 2.0,
        },
    ]
    baseline_weights = HybridWeights(1.0, 0.0, 0.0, 0.0)
    learned_weights = HybridWeights(0.5, 0.0, 0.0, 0.5)
    baseline = RetrievalEvaluator(
        lambda _query, _k: rank_hybrid(records, baseline_weights)
    ).evaluate(dataset, k=2, stability_runs=1)
    learned = RetrievalEvaluator(
        lambda _query, _k: rank_hybrid(records, learned_weights)
    ).evaluate(dataset, k=2, stability_runs=1)

    assert learned.cases["case"].reciprocal_rank > baseline.cases["case"].reciprocal_rank
    assert learned.utility_score > baseline.utility_score


def test_custom_downstream_utility_scorer_is_reported():
    dataset = EvaluationDataset(
        "task-quality", [EvaluationCase("case", "query", ["relevant"])]
    )
    evaluator = RetrievalEvaluator(
        lambda _query, _k: [{"id": "relevant"}],
        utility_scorer=lambda _case, _results: 0.73,
    )

    report = evaluator.evaluate(dataset, k=1, stability_runs=1)

    assert report.cases["case"].downstream_utility == 0.73
    assert report.utility_score == 0.73
