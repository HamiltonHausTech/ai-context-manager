from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any

try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
except ImportError:
    enc = None

def estimate_tokens(text: str) -> int:
    if enc:
        return len(enc.encode(text))
    return len(text.split())  # Rough fallback

# --- Base Component Class ---
class ContextComponent(ABC):
    def __init__(self, id: str, tags: Optional[List[str]] = None, lazy: bool = False):
        self.id = id
        self.tags = tags or []
        self.lazy = lazy
        self._content_cache = None

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
            "type": self.__class__.__name__
        }

    def matches_tags(self, include: List[str]) -> bool:
        return any(tag in self.tags for tag in include)

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