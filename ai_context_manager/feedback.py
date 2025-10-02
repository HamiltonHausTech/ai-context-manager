# ai_context_manager/store/feedback.py
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from ai_context_manager.store.base import FeedbackStore
from ai_context_manager.store.json_store import JSONFeedbackStore  # Default

class Feedback:
    def __init__(self, store: Optional[FeedbackStore] = None, decay_half_life_minutes: int = 60):
        self.store = store or JSONFeedbackStore()
        self.decay_half_life = timedelta(minutes=decay_half_life_minutes)

    def add_feedback(self, component_id: str, score: float, component_type: Optional[str] = None):
        if not component_type:
            component_type = "Unknown"
        self.store.add_feedback(component_id, score, component_type)

    def get_average_score(self, component_id: str) -> float:
        return self._weighted_average(self.store.get_scores(component_id))

    def get_average_score_by_type(self, component_type: str) -> float:
        return self._weighted_average(self.store.get_scores_by_type(component_type))

    def _weighted_average(self, entries: List[Dict]) -> float:
        if not entries:
            return 0.0

        now = datetime.utcnow()
        weighted_sum = 0.0
        total_weight = 0.0

        for entry in entries:
            score = entry["score"]
            timestamp = datetime.fromisoformat(entry["timestamp"])
            age = (now - timestamp).total_seconds() / 60.0
            weight = 0.5 ** (age / self.decay_half_life.total_seconds() * 60)
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
    def render_feedback(self, limit: int = 10) -> str:
        lines = ["[FEEDBACK] Feedback Summary"]
    
        lines.append("\n[COMPONENTS] Component Scores:")
        for comp_id in self.store.get_tracked_component_ids()[:limit]:
            entries = self.store.get_scores(comp_id)
            avg = self._weighted_average(entries)
            lines.append(f"  - {comp_id}: avg={avg:.2f} ({len(entries)} entries)")
    
        lines.append("\n[TYPES] Type Scores:")
        for comp_type in self.store.get_tracked_component_types()[:limit]:
            entries = self.store.get_scores_by_type(comp_type)
            avg = self._weighted_average(entries)
            lines.append(f"  - {comp_type}: avg={avg:.2f} ({len(entries)} entries)")
    
        return "\n".join(lines)
