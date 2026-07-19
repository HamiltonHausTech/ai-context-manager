"""Offline demonstration of memory consolidation and evaluation."""

from ai_context_manager import (
    ConsolidationEngine,
    ContextManager,
    EvaluationDataset,
    RetrievalEvaluator,
    RetrievalRequest,
)
from ai_context_manager.components import LongTermMemoryComponent, TaskSummaryComponent
from ai_context_manager.evaluation import EvaluationCase
from ai_context_manager.tokenization import estimate_tokens


manager = ContextManager()
manager.register_component(
    TaskSummaryComponent(
        "episode-bank-1",
        "Review checking account",
        "Recurring utility payments should use the Utilities category.",
        tags=["task", "banking"],
    )
)
manager.register_component(
    TaskSummaryComponent(
        "episode-bank-2",
        "Review credit card",
        "Merchant rules should be applied before broad category rules.",
        tags=["task", "banking"],
    )
)
manager.register_component(
    LongTermMemoryComponent(
        "preference-old",
        "The user prefers CSV reports.",
        "session-1",
        "2026-01-01T00:00:00",
        tags=["memory", "preference"],
    )
)
manager.register_component(
    LongTermMemoryComponent(
        "preference-new",
        "The user prefers XLSX reports with a summary tab.",
        "session-2",
        "2026-02-01T00:00:00",
        tags=["memory", "preference"],
    )
)

engine = ConsolidationEngine(manager)
bank_rules = engine.merge(
    "fact-bank-rules",
    ["episode-bank-1", "episode-bank-2"],
    content=(
        "Apply merchant-specific rules first; categorize recurring utility "
        "payments as Utilities."
    ),
    confidence=0.95,
    tags=["memory", "banking", "derived"],
)
engine.record_contradiction("preference-old", "preference-new")
engine.resolve_contradiction("preference-new", "preference-old")

result = manager.retrieve(
    RetrievalRequest(include_tags=["memory", "banking"], token_budget=200)
)
print("ACTIVE CONTEXT")
print(result.context)
print("\nPROVENANCE")
print(bank_rules.id, "derived from", bank_rules.memory.provenance_ids)
print("\nDECISIONS")
for decision in result.decisions:
    print(decision.component_id, decision.reason)


dataset = EvaluationDataset(
    "offline-consolidation-demo",
    [
        EvaluationCase(
            "bank-rules",
            "bank utility merchant category rules",
            relevant_ids=["fact-bank-rules"],
            excluded_ids=["preference-new"],
        )
    ],
)


def simple_active_search(query, limit):
    query_terms = set(query.lower().split())
    candidates = []
    for component in engine.active_components():
        content = component.get_content()
        overlap = len(query_terms & set(content.lower().split()))
        candidates.append(
            {"id": component.id, "tokens": estimate_tokens(content), "overlap": overlap}
        )
    return sorted(candidates, key=lambda item: item["overlap"], reverse=True)[:limit]


report = RetrievalEvaluator(simple_active_search).evaluate(dataset, k=1)
print("\nEVALUATION")
print(report.to_dict()["averages"])
