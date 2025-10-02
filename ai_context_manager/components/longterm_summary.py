from ai_context_manager.components import ContextComponent

class LongTermMemoryComponent(ContextComponent):
    def __init__(self, id: str, content: str, source: str, timestamp: str, score=0.5, tags=None):
        super().__init__(id, tags or ["memory", "longterm"])
        self.content = content
        self.source = source
        self.timestamp = timestamp
        self._score = score

    def load_content(self) -> str:
        return f"[{self.timestamp}] From {self.source}: {self.content}"

    def score(self) -> float:
        return self._score