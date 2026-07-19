from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

from ai_context_manager.tokenization import estimate_tokens
from ai_context_manager.memory import MemoryKind, MemoryLifecycle

# --- Base Component Class ---
class ContextComponent(ABC):
    def __init__(
        self,
        id: str,
        tags: Optional[List[str]] = None,
        lazy: bool = False,
        memory_kind: str = MemoryKind.GENERIC.value,
    ):
        self.id = id
        self.tags = tags or []
        self.lazy = lazy
        self._content_cache = None
        self.memory = MemoryLifecycle(kind=memory_kind)

    @abstractmethod
    def load_content(self) -> str:
        pass

    def get_content(self) -> str:
        if self.lazy:
            if self._content_cache is None:
                self._content_cache = self.load_content()
            return self._content_cache
        return self.load_content()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tags": self.tags,
            "type": self.__class__.__name__,
            "memory": self.memory.to_dict(),
        }

    def set_memory_lifecycle(self, lifecycle: MemoryLifecycle) -> None:
        lifecycle.validate()
        self.memory = lifecycle

    def matches_tags(self, include: List[str], mode: str = "any") -> bool:
        if mode == "all":
            return all(tag in self.tags for tag in include)
        if mode == "any":
            return any(tag in self.tags for tag in include)
        raise ValueError("Tag match mode must be 'any' or 'all'")

    def score(self) -> float:
        return 1.0

    def summarize(self, max_tokens: int) -> str:
        """Returns a shortened version of the component's content."""
        content = self.get_content()
        if estimate_tokens(content) <= max_tokens:
            return content
        return content[:max_tokens * 4]  # naive fallback; 4 chars/token approx

    def render_preview(self, score: float, token_count: int, summarized: bool = False) -> str:
        flags = " (summarized)" if summarized else ""
        content_preview = self.get_content()[:80].replace('\n', ' ')
        return (
            f"[{self.id}] {self.__class__.__name__}{flags}\n"
            f"  Score: {score:.2f} | Tokens: {token_count}\n"
            f"  Tags: {', '.join(self.tags)}\n"
            f"  Preview: {content_preview}...\n"
        )
