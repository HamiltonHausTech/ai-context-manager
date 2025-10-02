from ai_context_manager.components import ContextComponent

class TaskSummaryComponent(ContextComponent):
    def __init__(self, id: str, task_name: str, summary: str, score=1.0, tags=None):
        super().__init__(id, tags or ["task", "summary"])
        self.task_name = task_name
        self.summary = summary
        self._score = score

    def load_content(self) -> str:
        return f"Task: {self.task_name}\nSummary: {self.summary}"

    def score(self) -> float:
        return self._score


