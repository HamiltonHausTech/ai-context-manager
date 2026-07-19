from datetime import datetime, timezone

import pytest

from ai_context_manager.components import (
    AgentGoalComponent,
    AgentSessionComponent,
    DerivedMemoryComponent,
    LongTermMemoryComponent,
    TaskSummaryComponent,
    UserProfileComponent,
)
from ai_context_manager.consolidation import ConsolidationEngine, ConsolidationError
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.memory import MemoryKind, MemoryStatus
from ai_context_manager.retrieval import RetrievalRequest
from ai_context_manager.store.json_memory import JSONMemoryStore


def test_builtin_components_have_explicit_memory_categories():
    components = [
        (TaskSummaryComponent("task", "Task", "result"), MemoryKind.EPISODE.value),
        (
            AgentSessionComponent("session", "agent", "work", "done", 1.0),
            MemoryKind.EPISODE.value,
        ),
        (
            LongTermMemoryComponent("fact", "content", "source", "2026-01-01"),
            MemoryKind.DURABLE_FACT.value,
        ),
        (
            UserProfileComponent("user", "Ada", {"style": "brief"}),
            MemoryKind.PREFERENCE.value,
        ),
        (
            AgentGoalComponent("goal", "Ship it", "agent"),
            MemoryKind.GOAL.value,
        ),
        (
            DerivedMemoryComponent("derived", "summary"),
            MemoryKind.DERIVED_SUMMARY.value,
        ),
    ]

    assert [(component.memory.kind) for component, _kind in components] == [
        kind for _component, kind in components
    ]


def test_merge_preserves_provenance_and_supersedes_sources(tmp_path):
    path = tmp_path / "memory.json"
    manager = ContextManager(memory_store=JSONMemoryStore(str(path)))
    manager.register_component(TaskSummaryComponent("episode-1", "One", "first"))
    manager.register_component(TaskSummaryComponent("episode-2", "Two", "second"))

    merged = ConsolidationEngine(manager).merge(
        "summary",
        ["episode-1", "episode-2"],
        content="Both sessions established the durable rule.",
        confidence=0.9,
    )

    assert merged.memory.provenance_ids == ["episode-1", "episode-2"]
    assert merged.memory.supersedes_ids == ["episode-1", "episode-2"]
    assert merged.memory.confidence == 0.9
    assert manager.components["episode-1"].memory.status == MemoryStatus.SUPERSEDED.value

    recovered = ContextManager(memory_store=JSONMemoryStore(str(path)))
    assert recovered.components["summary"].memory.provenance_ids == [
        "episode-1",
        "episode-2",
    ]
    assert recovered.components["episode-2"].memory.status == MemoryStatus.SUPERSEDED.value


def test_contradictions_are_preserved_until_explicit_resolution():
    manager = ContextManager()
    manager.register_component(
        LongTermMemoryComponent("old", "User prefers PDF", "session-1", "2026-01-01")
    )
    manager.register_component(
        LongTermMemoryComponent("new", "User prefers DOCX", "session-2", "2026-02-01")
    )
    engine = ConsolidationEngine(manager)

    engine.record_contradiction("old", "new")

    assert manager.components["old"].memory.is_active()
    assert manager.components["new"].memory.is_active()
    assert manager.components["old"].memory.contradiction_ids == ["new"]

    engine.resolve_contradiction("new", "old")

    assert manager.components["old"].memory.status == MemoryStatus.SUPERSEDED.value
    assert manager.components["new"].memory.supersedes_ids == ["old"]
    assert manager.components["new"].memory.contradiction_ids == []


def test_expiration_is_applied_and_explained_by_retrieval():
    manager = ContextManager()
    temporary = LongTermMemoryComponent(
        "temporary", "Short-lived fact", "test", "2026-01-01"
    )
    temporary.memory.expires_at = "2026-02-01T00:00:00+00:00"
    manager.register_component(temporary)

    expired = ConsolidationEngine(manager).expire_due(
        datetime(2026, 3, 1, tzinfo=timezone.utc)
    )
    result = manager.retrieve(RetrievalRequest())
    included = manager.retrieve(RetrievalRequest(include_inactive=True))

    assert expired == ["temporary"]
    assert result.items == []
    assert result.decisions[0].reason == "lifecycle_expired"
    assert [item.component.id for item in included.items] == ["temporary"]


def test_derive_rejects_missing_sources_and_empty_content():
    manager = ContextManager()
    manager.register_component(TaskSummaryComponent("source", "Task", "result"))
    engine = ConsolidationEngine(manager)

    with pytest.raises(ConsolidationError, match="Unknown source"):
        engine.derive("derived", "content", ["missing"])
    with pytest.raises(ConsolidationError, match="cannot be empty"):
        engine.derive("derived", " ", ["source"])
