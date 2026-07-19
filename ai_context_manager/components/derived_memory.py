from typing import List, Optional

from ai_context_manager.components.base import ContextComponent
from ai_context_manager.memory import MemoryKind


class DerivedMemoryComponent(ContextComponent):
    """A consolidated memory whose provenance points to source components."""

    def __init__(
        self,
        id: str,
        content: str,
        derivation: str = "consolidation",
        score: float = 1.0,
        tags: Optional[List[str]] = None,
    ):
        super().__init__(
            id,
            tags or ["memory", "derived"],
            memory_kind=MemoryKind.DERIVED_SUMMARY.value,
        )
        self.content = content
        self.derivation = derivation
        self._score = score

    def load_content(self) -> str:
        return f"Derived Memory ({self.derivation}): {self.content}"

    def score(self) -> float:
        return self._score
