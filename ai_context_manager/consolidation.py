"""Deterministic policies for deriving and maintaining durable memories."""

from datetime import datetime, timezone
from typing import Iterable, List, Optional

from ai_context_manager.components import DerivedMemoryComponent
from ai_context_manager.context_manager import ContextManager
from ai_context_manager.memory import MemoryKind, MemoryLifecycle, MemoryStatus


class ConsolidationError(ValueError):
    pass


class ConsolidationEngine:
    def __init__(self, context_manager: ContextManager):
        self.context_manager = context_manager

    def _get_sources(self, source_ids: Iterable[str]):
        ordered_ids = list(dict.fromkeys(source_ids))
        if not ordered_ids:
            raise ConsolidationError("At least one source memory is required")
        missing = [
            source_id
            for source_id in ordered_ids
            if source_id not in self.context_manager.components
        ]
        if missing:
            raise ConsolidationError(
                f"Unknown source memories: {', '.join(sorted(missing))}"
            )
        return [self.context_manager.components[source_id] for source_id in ordered_ids]

    def _persist_lifecycle(self, component) -> None:
        if self.context_manager.memory_store:
            self.context_manager.save_component_to_memory(component)

    def derive(
        self,
        derived_id: str,
        content: str,
        source_ids: Iterable[str],
        *,
        derivation: str = "consolidation",
        score: float = 1.0,
        confidence: float = 1.0,
        expires_at: Optional[str] = None,
        supersede_sources: bool = False,
        tags: Optional[List[str]] = None,
    ) -> DerivedMemoryComponent:
        sources = self._get_sources(source_ids)
        if not content.strip():
            raise ConsolidationError("Derived memory content cannot be empty")
        if derived_id in self.context_manager.components:
            raise ConsolidationError(f"Memory {derived_id} already exists")

        derived = DerivedMemoryComponent(
            derived_id, content, derivation=derivation, score=score, tags=tags
        )
        derived.set_memory_lifecycle(
            MemoryLifecycle(
                kind=MemoryKind.DERIVED_SUMMARY.value,
                provenance_ids=[source.id for source in sources],
                supersedes_ids=(
                    [source.id for source in sources] if supersede_sources else []
                ),
                expires_at=expires_at,
                confidence=confidence,
            )
        )
        self.context_manager.register_component(derived, on_duplicate="error")

        if supersede_sources:
            for source in sources:
                source.memory.status = MemoryStatus.SUPERSEDED.value
                self._persist_lifecycle(source)
        return derived

    def merge(
        self,
        merged_id: str,
        source_ids: Iterable[str],
        content: Optional[str] = None,
        **kwargs,
    ) -> DerivedMemoryComponent:
        sources = self._get_sources(source_ids)
        merged_content = content or "\n\n".join(
            source.get_content() for source in sources
        )
        return self.derive(
            merged_id,
            merged_content,
            [source.id for source in sources],
            derivation="merge",
            supersede_sources=True,
            **kwargs,
        )

    def record_contradiction(self, left_id: str, right_id: str) -> None:
        if left_id == right_id:
            raise ConsolidationError("A memory cannot contradict itself")
        left, right = self._get_sources([left_id, right_id])
        if right.id not in left.memory.contradiction_ids:
            left.memory.contradiction_ids.append(right.id)
        if left.id not in right.memory.contradiction_ids:
            right.memory.contradiction_ids.append(left.id)
        self._persist_lifecycle(left)
        self._persist_lifecycle(right)

    def resolve_contradiction(self, winner_id: str, superseded_id: str) -> None:
        winner, superseded = self._get_sources([winner_id, superseded_id])
        superseded.memory.status = MemoryStatus.SUPERSEDED.value
        if superseded.id not in winner.memory.supersedes_ids:
            winner.memory.supersedes_ids.append(superseded.id)
        winner.memory.contradiction_ids = [
            item for item in winner.memory.contradiction_ids if item != superseded.id
        ]
        superseded.memory.contradiction_ids = [
            item for item in superseded.memory.contradiction_ids if item != winner.id
        ]
        self._persist_lifecycle(winner)
        self._persist_lifecycle(superseded)

    def expire_due(self, now: Optional[datetime] = None) -> List[str]:
        current = now or datetime.now(timezone.utc)
        expired = []
        for component in self.context_manager.components.values():
            if (
                component.memory.status == MemoryStatus.ACTIVE.value
                and component.memory.is_expired(current)
            ):
                component.memory.status = MemoryStatus.EXPIRED.value
                self._persist_lifecycle(component)
                expired.append(component.id)
        return expired

    def active_components(self, now: Optional[datetime] = None):
        return [
            component
            for component in self.context_manager.components.values()
            if component.memory.is_active(now)
        ]
