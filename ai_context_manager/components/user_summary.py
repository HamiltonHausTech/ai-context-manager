from ai_context_manager.components import ContextComponent
from typing import Dict, Any

class UserProfileComponent(ContextComponent):
    def __init__(self, id: str, name: str, preferences: Dict[str, Any], score=1.0, tags=None):
        super().__init__(id, tags or ["user", "profile"])
        self.name = name
        self.preferences = preferences
        self._score = score

    def load_content(self) -> str:
        prefs = ", ".join(f"{k}: {v}" for k, v in self.preferences.items())
        return f"User: {self.name}\nPreferences: {prefs}"

    def score(self) -> float:
        return self._score
